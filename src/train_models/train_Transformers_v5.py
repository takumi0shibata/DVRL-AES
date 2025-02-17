'''
This script is used to train the model using the Transformer-based model.
    - BERT
    - DeBERTa-v3-large
'''
import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
import wandb

import polars as pl

# my packages
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dvrl.dataset import EssayDataset
from utils.dvrl_utils import remove_top_p_sample 
from utils.create_embedding_feautres import create_data_loader
from utils.evaluation import train_epoch, evaluate_epoch
from models.transfomer_enc import BERT_Regressor
from utils.general_utils import set_seed


def train_and_evaluate(
    train_data,
    dev_data,
    test_data,
    pseudo_dict,
    target_prompt_id,
    weights,
    device,
    args,
):
    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    source = {
        'feature': train_data['essay'][weights],
        'normalized_label': train_data['scaled_score'][weights],
        'essay_set': train_data['essay_set'][weights],
    }

    dev = {
        'feature': dev_data['essay'],
        'normalized_label': dev_data['scaled_score'],
        'essay_set': dev_data['essay_set'],
    }

    pseudo = {
        'feature': test_data['essay'],
        'normalized_label': np.array([pseudo_dict[eid] for eid in test_data['essay_id']]),
        'essay_set': test_data['essay_set'],
    }

    target = {
        'feature': test_data['essay'],
        'normalized_label': test_data['scaled_score'],
        'essay_set': test_data['essay_set'],
    }
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    config = AutoConfig.from_pretrained(model_name)
    
    train_loader = create_data_loader(source, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
    if args.dev_size != 0:
        dev_loader = create_data_loader(dev, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
    pseudo_loader = create_data_loader(pseudo, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
    test_loader = create_data_loader(target, tokenizer, max_length=args.max_length, batch_size=args.batch_size)
    
    # Initialize the model
    model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)
    
    # Define loss function, optimizer, and scheduler
    loss_fn = nn.MSELoss(reduction='none').to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=len(train_loader)*args.epochs
    )
    
    # Training loop
    best_dev_qwk = -1
    best_test_qwk = -1
    best_loss = 1000
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
        # Dev
        if args.dev_size != 0:
            dev_history = evaluate_epoch(model, dev_loader, loss_fn, device, 'score')
            dev_loss = dev_history['loss']
            dev_qwk = dev_history['qwk']
        else:
            dev_loss = 0
            dev_qwk = 0
        # Pseudo
        pseudo_history = evaluate_epoch(model, pseudo_loader, loss_fn, device, 'score')
        pseudo_loss = pseudo_history['loss']
        pseudo_qwk = pseudo_history['qwk']
        # Test
        test_history = evaluate_epoch(model, test_loader, loss_fn, device, 'score')

        model_selection_loss = (1 - args.loss_lambda) * dev_loss + args.loss_lambda * pseudo_loss
        model_selection_qwk = (1 - args.loss_lambda) * dev_qwk + args.loss_lambda * pseudo_qwk
    
        if model_selection_qwk > best_dev_qwk:
            best_loss = model_selection_loss
            best_dev_qwk = model_selection_qwk
            best_test_qwk = test_history['qwk']

    return best_test_qwk, best_loss



def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        dev_size=args.dev_size,
        cache_dir='src/.embedding_cache',
        embedding_model=args.embedding_model,
        add_pos=False,
        device=device
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')
    print(f'    Selected Dev data: {dev_data["essay_id"]}')
    print(f'    The score of selected Dev data: {dev_data["original_score"]}')

    # load pseudo label
    df = pl.read_csv('data/pseudo_label.csv').to_dict()
    pseudo_dict = dict(zip(df['essay_id'].to_numpy(), df['y_pred'].to_numpy()))
    estimated_data_value = np.load(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}.npy')
    
    for p_val in np.arange(0.0, 1.0, 0.1):
        ###################################################
        # データの価値が低いものを削除
        ###################################################
        set_seed(args.seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=False,
        )
        qwk_high, dev_loss_high = train_and_evaluate(
            train_data,
            dev_data,
            test_data,
            pseudo_dict,
            target_prompt_id,
            weights,
            device,
            args,
        )
        ###################################################
        # データの価値が高いものを削除
        ###################################################
        set_seed(args.seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=True,
        )
        qwk_low, dev_loss_low = train_and_evaluate(
            train_data,
            dev_data,
            test_data,
            pseudo_dict,
            target_prompt_id,
            weights,
            device,
            args
        )
        
        if args.wandb:
            wandb.log({
                'p': p_val,
                'QWK[High]': qwk_high,
                'QWK[Low]': qwk_low,
                'Dev Loss[High]': dev_loss_high,
                'Dev Loss[Low]': dev_loss_low,
            })
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser('Training Model')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V5')
    parser.add_argument('--run_name', type=str, default='train-BERT')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--max_length', type=int, default=512, choices=[512, 256]) # BERT-base: 512, DeBERTa-v3-large: 256
    parser.add_argument('--batch_size', type=int, default=16, choices=[16, 8]) # BERT-base: 16, DeBERTa-v3-large: 8
    parser.add_argument('--epochs', type=int, default=10, choices=[10, 5]) # BERT-base: 10, DeBERTa-v3-large: 5
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--pred_model', type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--ot', action='store_true')
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-uncased',
        choices=[
            'bert-base-uncased',
            'microsoft/deberta-v3-large'
        ]
    )

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)