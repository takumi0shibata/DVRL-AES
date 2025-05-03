"""
Training on DVRL
問題&得点ごとにクラスタを分割し、各クラスタに対して、価値を推定する。
"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb
import polars as pl

from dvrl import dvrl_v3
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.prompt_dataset import PromptDataset
from dvrl.predictor import MLP
from models.features import FeaturesModel
from models.paes import PAES
from dvrl.predictor_config import MLPConfig, FeaturesModelConfig, PAESModelConfig


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
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}',
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
        add_pos=True,
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')

    # Load prompt data
    print('Loading prompt data...')
    prompt_dataset = PromptDataset('data/prompts.csv')
    prompt_data = prompt_dataset.load()
    print(f'    Number of clusters: {len(np.unique(prompt_data["cluster"]))}')

    # Assign cluster
    train_cluster = []
    for i in range(len(train_data['essay_id'])):
        essay_set = train_data['essay_set'][i]
        score = train_data['original_score'][i]

        is_match_essay_set = (prompt_data['essay_set'] == essay_set)
        is_match_score = (prompt_data['score'] == score)
        cluster = prompt_data['cluster'][is_match_essay_set & is_match_score]
        train_cluster.append(cluster[0])
    train_data['cluster'] = np.array(train_cluster)
    print(f'    Number of training clusters: {len(np.unique(train_data["cluster"]))}')

    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    if args.pred_model == 'mlp':
        config = MLPConfig()
        pred_model = MLP(input_feature=train_data['embedding'].shape[1]).to(device)
        x_train = train_data['embedding']
        x_dev = dev_data['embedding']
    elif args.pred_model == 'features_model':
        config = FeaturesModelConfig()
        pred_model = FeaturesModel().to(device)
        train_data['ridley_feature'] = np.concatenate([train_data['feature'], train_data['readability']], axis=1)
        dev_data['ridley_feature'] = np.concatenate([dev_data['feature'], dev_data['readability']], axis=1)
        x_train = train_data['ridley_feature']
        x_dev = dev_data['ridley_feature']
    elif args.pred_model == 'paes':
        config = PAESModelConfig()
        pred_model = PAES(train_data['max_sentnum'], train_data['max_sentlen'], train_data['pos_vocab']).to(device)
        x_train = [train_data['pos_x'], train_data['feature'], train_data['readability']]
        x_dev = [dev_data['pos_x'], dev_data['feature'], dev_data['readability']]

    # Network parameters
    print('Initialize DVRL framework...')
    dvrl_params = {
        'hidden_dim': 100,
        'comb_dim': 64,
        'iterations': 500,
        'activation': nn.Tanh(),
        'layer_number': 5,
        'lr': 0.001,
        'wandb': args.wandb,
    }
    
    # Initialize DVRL
    dvrl_class = dvrl_v3.Dvrl(
        train_data,
        x_train,
        train_data['scaled_score'],
        x_dev,
        dev_data['scaled_score'],
        prompt_data,
        pred_model,
        dvrl_params,
        device,
        target_prompt_id,
        config,
    )

    # Train DVRL
    print('Training DVRL...')
    outputs = dvrl_class.train_dvrl(args.metric)

    # Save DVRL
    print('Saving DVRL...')
    output_dir = './outputs/dvrl_v3'
    os.makedirs(output_dir, exist_ok=True)
    pl.DataFrame(outputs).write_csv(os.path.join(output_dir, f'estimated_values_{target_prompt_id}_{args.pred_model}_{args.seed}.csv'))

    if args.wandb:
        wandb.alert(title=args.pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-v3-test')
    parser.add_argument('--run_name', type=str, default='Valuation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--pred_model',
        type=str,
        default='mlp',
        choices=[
            'mlp',
            'features_model',
            'paes',
            'bert',
            'deberta_v3_large'
        ]
    )
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)