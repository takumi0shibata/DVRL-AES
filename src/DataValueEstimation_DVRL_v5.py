"""Training on DVRL
PAESを利用して作成した擬似ラベルを報酬計算に利用する手法
"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb
import polars as pl

from dvrl import dvrl_v5
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.predictor import MLP
from models.features import FeaturesModel


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
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}',
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
    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    dvrl_data = {}
    dvrl_data['y_source'] = train_data['scaled_score']
    dvrl_data['y_dev'] = dev_data['scaled_score']
    dvrl_data['y_pseudo'] = np.array([pseudo_dict[eid] for eid in test_data['essay_id']])
    if args.pred_model == 'mlp':
        pred_model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        dvrl_data['x_source'] = train_data['embedding']
        dvrl_data['x_dev'] = dev_data['embedding']
        dvrl_data['x_pseudo'] = test_data['embedding']
    elif args.pred_model == 'features_model':
        pred_model = FeaturesModel().to(device)
        train_data['ridley_feature'] = np.concatenate([train_data['feature'], train_data['readability']], axis=1)
        dev_data['ridley_feature'] = np.concatenate([dev_data['feature'], dev_data['readability']], axis=1)
        test_data['ridley_feature'] = np.concatenate([test_data['feature'], test_data['readability']], axis=1)
        dvrl_data['x_source'] = train_data['ridley_feature']
        dvrl_data['x_dev'] = dev_data['ridley_feature']
        dvrl_data['x_pseudo'] = test_data['ridley_feature']

    # Network parameters
    print('Initialize DVRL framework...')
    dvrl_params = {
        'hidden_dim': 100,
        'comb_dim': 10,
        'iterations': 1000,
        'activation': nn.Tanh(),
        'layer_number': 5,
        'learning_rate': 0.001,
        'batch_size': 10000,
        'inner_iterations': 100,
        'batch_size_predictor': 512,
        'loss_lambda': args.loss_lambda,
        'wandb': args.wandb,
        'ot': args.ot,
    }

    # Initialize DVRL
    dvrl_class = dvrl_v5.Dvrl(
        dvrl_data,
        pred_model,
        dvrl_params,
        device,
        target_prompt_id
    )

    # Train DVRL
    print('Training DVRL...')
    data_value = dvrl_class.train_dvrl(args.metric)

    # # Estimate data value
    # print('Estimating data value...')
    # data_value = dvrl_class.dvrl_valuator(dvrl_data['x_source'], dvrl_data['y_source'])

    print('Saving DVRL...')
    output_dir = f'./outputs/{args.pjname}'
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir + f'/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}.npy', data_value)

    if args.wandb:
        wandb.alert(title=args.pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V5')
    parser.add_argument('--run_name', type=str, default='Valuation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pred_model',type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--ot', action='store_true')
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)