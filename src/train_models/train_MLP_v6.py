'''
This script trains MLP model.
    - for DeBERTa-v3-large embedding vectors
    - for mannualy designed features
'''

import os
import torch
import numpy as np
import argparse
import torch
import wandb
import polars as pl
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.predictor import MLP
from models.features import FeaturesModel

from utils.dvrl_utils import (
    fit_func,
    pred_func,
    calc_qwk,
    remove_top_p_sample
)


def train_and_evaluate(
    train_data,
    dev_data,
    test_data,
    pseudo_dict,
    target_prompt_id,
    weights,
    batch_size,
    epochs,
    device,
    attribute_name,
    args,
):
    
    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    # Create predictor
    print('Creating predictor model...')
    mlp_data = {}
    mlp_data['y_source'] = train_data['scaled_score']
    mlp_data['y_dev'] = dev_data['scaled_score']
    mlp_data['y_pseudo'] = np.array([pseudo_dict[eid] for eid in test_data['essay_id']])
    mlp_data['y_test'] = test_data['scaled_score']
    if args.pred_model == 'mlp':
        pred_model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        mlp_data['x_source'] = train_data['embedding']
        mlp_data['x_dev'] = dev_data['embedding']
        mlp_data['x_test'] = test_data['embedding']
    elif args.pred_model == 'features_model':
        pred_model = FeaturesModel().to(device)
        train_data['ridley_feature'] = np.concatenate([train_data['feature'], train_data['readability']], axis=1)
        dev_data['ridley_feature'] = np.concatenate([dev_data['feature'], dev_data['readability']], axis=1)
        test_data['ridley_feature'] = np.concatenate([test_data['feature'], test_data['readability']], axis=1)
        mlp_data['x_source'] = train_data['ridley_feature']
        mlp_data['x_dev'] = dev_data['ridley_feature']
        mlp_data['x_test'] = test_data['ridley_feature']

    fit_func(
        pred_model,
        mlp_data['x_source'][weights],
        mlp_data['y_source'][weights],
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    # For test
    y_pred_test = pred_func(
        pred_model,
        mlp_data['x_test'],
        batch_size=batch_size,
        device=device
    )

    # For dev
    if len(mlp_data['x_dev']) != 0:
        y_pred_dev = pred_func(
            pred_model,
            mlp_data['x_dev'],
            batch_size=batch_size,
            device=device
        )
        dev_loss = mean_squared_error(mlp_data['y_dev'], y_pred_dev)
        pseudo_loss = mean_squared_error(mlp_data['y_pseudo'], y_pred_test)
        dev_qwk = calc_qwk(mlp_data['y_dev'], y_pred_dev, target_prompt_id, attribute_name)
        pseudo_qwk = calc_qwk(mlp_data['y_pseudo'], y_pred_test, target_prompt_id, attribute_name)

        dev_loss = (1 - args.loss_lambda) * dev_loss + args.loss_lambda * pseudo_loss
        dev_qwk = (1 - args.loss_lambda) * dev_qwk + args.loss_lambda * pseudo_qwk
    else:
        dev_loss = mean_squared_error(mlp_data['y_pseudo'], y_pred_test)
        dev_qwk = calc_qwk(mlp_data['y_pseudo'], y_pred_test, target_prompt_id, attribute_name)
    
    test_qwk = calc_qwk(mlp_data['y_test'], y_pred_test, target_prompt_id, attribute_name)
    return test_qwk, dev_loss, dev_qwk

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
            name=args.run_name + f'_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}',
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
    df = pl.read_csv('data/pseudo_label_by_features_model.csv').to_dict()
    pseudo_dict = dict(zip(df['essay_id'].to_numpy(), df['y_pred'].to_numpy()))
    estimated_data_value = np.load(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}.npy')
    
    for p_val in np.arange(0.0, 1.0, 0.1):
        ##################################################
        # データの価値が低いものを削除
        ##################################################
        set_seed(args.seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=False,
        )
        qwk_high, dev_loss_high, dev_qwk_high = train_and_evaluate(
            train_data,
            dev_data,
            test_data,
            pseudo_dict,
            target_prompt_id,
            weights,
            args.batch_size,
            args.epochs,
            args.device,
            args.attribute_name,
            args,
        )

        ##################################################
        # データの価値が高いものを削除
        ##################################################
        set_seed(args.seed)
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p_val,
            ascending=True,
        )
        qwk_low, dev_loss_low, dev_qwk_low = train_and_evaluate(
            train_data,
            dev_data,
            test_data,
            pseudo_dict,
            target_prompt_id,
            weights,
            args.batch_size,
            args.epochs,
            args.device,
            args.attribute_name,
            args,
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
    parser.add_argument('--pjname', type=str, default='DVRL-V5')
    parser.add_argument('--run_name', type=str, default='train-MLP')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pred_model', type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)