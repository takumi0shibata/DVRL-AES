"""
Training on DVRL
同問題&同得点内でクラスタリングを行い、そのクラスタリング結果を使ってDVRLを学習する。
"""

import os
import torch
import numpy as np
import argparse
import torch
import torch.nn as nn
import wandb
import polars as pl

import umap
from sklearn.cluster import KMeans

from dvrl import dvrl_v4
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.predictor import MLP
from models.features import FeaturesModel
from dvrl.predictor_config import MLPConfig, FeaturesModelConfig


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
        add_pos=False,
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')


    # Assign cluster based on embedding similarity
    print('Assigning clusters based on embedding similarity...')
    embeddings = train_data['embedding']

    # UMAPで5次元に次元削減
    reducer = umap.UMAP(n_components=5, random_state=args.seed)
    reduced_embeddings = reducer.fit_transform(embeddings)
    # 埋め込みベクトルを正規化
    norm_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
    # 得点データを結合
    norm_embeddings = np.concatenate((norm_embeddings, train_data['scaled_score'].reshape(-1, 1)), axis=1)
    # クラスタリング
    clustering = KMeans(n_clusters=args.n_clusters, random_state=args.seed).fit(reduced_embeddings)  # n_clusters should be tuned

    train_data['cluster'] = clustering.labels_

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

    # Network parameters
    print('Initialize DVRL framework...')
    dvrl_params = {
        'hidden_dim': 100,
        'comb_dim': 64,
        'iterations': 1000,
        'activation': nn.Tanh(),
        'layer_number': 5,
        'lr': 0.001,
        'wandb': args.wandb,
    }
    
    # Initialize DVRL
    dvrl_class = dvrl_v4.Dvrl(
        train_data,
        x_train,
        train_data['scaled_score'],
        x_dev,
        dev_data['scaled_score'],
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
    output_dir = './outputs/dvrl_v4'
    os.makedirs(output_dir, exist_ok=True)
    pl.DataFrame(outputs).write_csv(os.path.join(output_dir, f'estimated_values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_cluster{args.n_clusters}.csv'))

    if args.wandb:
        wandb.alert(title=args.pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="DVRL")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V4')
    parser.add_argument('--run_name', type=str, default='Valuation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--metric', type=str, default='qwk', choices=['corr', 'mse', 'qwk'])
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_clusters', type=int, default=1000)
    parser.add_argument('--pred_model',type=str, default='mlp', choices=['mlp', 'features_model'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)