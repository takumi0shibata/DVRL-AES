'''
This script trains a PAES model on the target prompt using the estimated data values.
'''

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
import polars as pl
from tqdm import tqdm
import pickle

# my packages
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dvrl_utils import select_non_top_essay_ids_by_percent
from utils.general_utils import set_seed, get_min_max_scores
from dvrl.dataset import EssayDataset
from dvrl.sampling import select_diverse_subset
from models.paes import PAES
from sklearn.metrics import cohen_kappa_score


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        # Move batch to device
        pos_x, linguistic_x, readability_x, labels, essay_set = [b.to(device) for b in batch]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(pos_x, linguistic_x, readability_x)
        
        # Calculate loss
        loss = loss_fn(outputs.squeeze(), labels.squeeze())
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_epoch(model: nn.Module, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_essay_set = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            pos_x, linguistic_x, readability_x, labels, essay_set = [b.to(device) for b in batch]
            
            # Forward pass
            outputs = model(pos_x, linguistic_x, readability_x)
            
            # Calculate loss
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            total_loss += loss.item()
            
            # Store predictions and labels for QWK calculation
            # スカラー値の場合の処理を追加
            squeezed_outputs = outputs.squeeze()
            squeezed_labels = labels.squeeze()
            squeezed_essay_set = essay_set.squeeze()
            
            # スカラー値かテンソルかを判定して適切に処理
            if squeezed_outputs.dim() == 0:  # スカラー値の場合
                all_preds.append(squeezed_outputs.item())
            else:
                all_preds.extend(squeezed_outputs.cpu().numpy())
                
            if squeezed_labels.dim() == 0:  # スカラー値の場合
                all_labels.append(squeezed_labels.item())
            else:
                all_labels.extend(squeezed_labels.cpu().numpy())
            
            if squeezed_essay_set.dim() == 0:
                all_essay_set.append(squeezed_essay_set.item())
            else:
                all_essay_set.extend(squeezed_essay_set.cpu().numpy())
    
    # Calculate QWK
    qwks = []
    lwks = []
    for i in range(1, 9):
        minscore, maxscore = get_min_max_scores()[i]['score']
        indices = np.where(np.array(all_essay_set) == i)[0]
        if len(indices) > 0:
            rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(all_labels)[indices])
            rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(all_preds)[indices])
            qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
            lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')
            qwks.append(qwk)
            lwks.append(lwk)
    
    print(np.array(qwks))
    return {
        'loss': total_loss / len(data_loader),
        'lwk': lwks,
        'qwk': qwk
    }


def train_and_evaluate(
    source_data,
    target_data,
    pseudo_dict,
    embedding_dict,
    top_essay_ids,
    dev_mask,
    args,
    device,
) -> tuple[float, float, float]:
    
    # Build value_mask over source_data
    source_mask = np.array([
        essay_id in top_essay_ids for essay_id in source_data['essay_id']
    ])
    print(f'{np.sum(source_mask)} / {len(source_data["essay_id"])}')

    # Initialize the model
    source = {
        'pos_x': source_data['pos_x'][source_mask],
        'linguistic_x': source_data['feature'][source_mask],
        'readability_x': source_data['readability'][source_mask],
        'normalized_label': source_data['scaled_score'][source_mask],
        'essay_set': source_data['essay_set'][source_mask],
    }

    dev = {
        'pos_x': target_data['pos_x'][dev_mask],
        'linguistic_x': target_data['feature'][dev_mask],
        'readability_x': target_data['readability'][dev_mask],
        'normalized_label': target_data['scaled_score'][dev_mask],
        'essay_set': target_data['essay_set'][dev_mask],
    }

    pseudo = {
        'pos_x': target_data['pos_x'][~dev_mask],
        'linguistic_x': target_data['feature'][~dev_mask],
        'readability_x': target_data['readability'][~dev_mask],
        'normalized_label': np.array([pseudo_dict[eid] for eid in target_data['essay_id'][~dev_mask]]),
        'essay_set': target_data['essay_set'][~dev_mask],
    }

    target = {
        'pos_x': target_data['pos_x'][~dev_mask],
        'linguistic_x': target_data['feature'][~dev_mask],
        'readability_x': target_data['readability'][~dev_mask],
        'normalized_label': target_data['scaled_score'][~dev_mask],
        'essay_set': target_data['essay_set'][~dev_mask],
    }

    source_dataset = TensorDataset(
        torch.tensor(source['pos_x'], dtype=torch.int32),
        torch.tensor(source['linguistic_x'], dtype=torch.float32),
        torch.tensor(source['readability_x'], dtype=torch.float32),
        torch.tensor(source['normalized_label'], dtype=torch.float32),
        torch.tensor(source['essay_set'], dtype=torch.int32),
    ) 
    if args.loss_lambda != 1.0:
        dev_dataset = TensorDataset(
            torch.tensor(dev['pos_x'], dtype=torch.int32),
            torch.tensor(dev['linguistic_x'], dtype=torch.float32),
            torch.tensor(dev['readability_x'], dtype=torch.float32),
            torch.tensor(dev['normalized_label'], dtype=torch.float32),
            torch.tensor(dev['essay_set'], dtype=torch.int32),
        )
    pseudo_dataset = TensorDataset(
        torch.tensor(pseudo['pos_x'], dtype=torch.int32),
        torch.tensor(pseudo['linguistic_x'], dtype=torch.float32),
        torch.tensor(pseudo['readability_x'], dtype=torch.float32),
        torch.tensor(pseudo['normalized_label'], dtype=torch.float32),
        torch.tensor(pseudo['essay_set'], dtype=torch.int32),
    )
    target_dataset = TensorDataset(
        torch.tensor(target['pos_x'], dtype=torch.int32),
        torch.tensor(target['linguistic_x'], dtype=torch.float32),
        torch.tensor(target['readability_x'], dtype=torch.float32),
        torch.tensor(target['normalized_label'], dtype=torch.float32),
        torch.tensor(target['essay_set'], dtype=torch.int32),
    )

    train_loader = DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True)
    if args.loss_lambda != 1.0:
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=True)
    pseudo_loader = DataLoader(dataset=pseudo_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    model = PAES(source_data['max_sentnum'], source_data['max_sentlen'], source_data['pos_vocab']).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='mean').to(device)

    # Training loop
    best_dev_qwk = -1
    best_test_qwk = -1
    best_loss = 1000
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        if args.loss_lambda != 1.0:
            # Dev
            dev_history = evaluate_epoch(model, dev_loader, loss_fn, device)
            dev_loss = dev_history['loss']
            dev_qwk = dev_history['qwk']
            # Pseudo
            pseudo_history = evaluate_epoch(model, pseudo_loader, loss_fn, device)
            pseudo_loss = pseudo_history['loss']
            pseudo_qwk = pseudo_history['qwk']
            # sum
            dev_loss = (1 - args.loss_lambda) * dev_loss + args.loss_lambda * pseudo_loss
            dev_qwk = (1 - args.loss_lambda) * dev_qwk + args.loss_lambda * pseudo_qwk
        else:
            # Pseudo
            pseudo_history = evaluate_epoch(model, pseudo_loader, loss_fn, device)
            dev_loss = pseudo_history['loss']
            dev_qwk = pseudo_history['qwk']
        # Test
        test_history = evaluate_epoch(model, test_loader, loss_fn, device)
    
        if dev_qwk > best_dev_qwk:
            best_loss = dev_loss
            best_dev_qwk = dev_qwk
            best_test_qwk = test_history['qwk']

    return best_test_qwk, best_loss, best_dev_qwk

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
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load embedding
    with open('./outputs/embedding/deberta-v3-large.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)

    # load pseudo label
    pseudo_df = pl.read_csv(f'./outputs/pseudo_labels/PAES_pred_{target_prompt_id}_seed{args.seed}.csv')
    pseudo_dict = dict(zip(pseudo_df['essay_id'].to_numpy(), pseudo_df['pred'].to_numpy()))
    
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    source_data, target_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        add_pos=True,
    )
    print(f'    Number of source samples: {len(source_data["essay_id"])}')
    print(f'    Number of target samples: {len(target_data["essay_id"])}')

    # Load Estimated Data Value
    # data_value_df = pl.read_csv(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}.csv')
    data_value_df = pl.read_csv(f'outputs/DVRL-V7-20250501/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}.csv')

    # Select dev data
    selected_dev_ids = select_diverse_subset(
        {essay_id: embedding_dict[essay_id] for essay_id in target_data['essay_id'] if essay_id in embedding_dict},
        args.dev_size,
        method=args.sampling,
        seed=args.seed
    )
    dev_mask = np.isin(target_data['essay_id'], selected_dev_ids)
    
    for p_val in np.arange(0.0, 1.0, 0.1):
        ##################################################
        # データの価値が低いものを削除
        ##################################################
        set_seed(args.seed)
        top_source_essay_ids = select_non_top_essay_ids_by_percent(
            data_value_df,
            percentage=p_val,
            ascending=True
        )
        qwk_high, dev_loss_high, dev_qwk_high = train_and_evaluate(
            source_data,
            target_data,
            pseudo_dict,
            embedding_dict,
            top_source_essay_ids,
            dev_mask,
            args,
            device,
        )

        ##################################################
        # データの価値が高いものを削除
        ##################################################
        set_seed(args.seed)
        top_target_essay_ids = select_non_top_essay_ids_by_percent(
            data_value_df,
            percentage=p_val,
            ascending=False
        )
        qwk_low, dev_loss_low, dev_qwk_low = train_and_evaluate(
            source_data,
            target_data,
            pseudo_dict,
            embedding_dict,
            top_target_essay_ids,
            dev_mask,
            args,
            device,
        )

        print(f'p: {p_val:.1f}, QWK[High]: {qwk_high:.3f}, QWK[Low]: {qwk_low:.3f}, Dev Loss[High]: {dev_loss_high:.10f}, Dev Loss[Low]: {dev_loss_low:.10f}')

        if args.wandb:
            wandb.log({
                'p': p_val,
                'QWK[High]': qwk_high,
                'QWK[Low]': qwk_low,
                'Dev Loss[High]': dev_loss_high,
                'Dev Loss[Low]': dev_loss_low,
                'Dev QWK[High]': dev_qwk_high,
                'Dev QWK[Low]': dev_qwk_low,
            })

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V7')
    parser.add_argument('--run_name', type=str, default='train-PAES')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pred_model', type=str, default='features_model')
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'greedy', 'maxmin', 'kmeans++'])
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)