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
import pickle
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.predictor import MLP
from models.features import FeaturesModel
from dvrl.sampling import select_diverse_subset

from utils.dvrl_utils import (
    fit_func,
    pred_func,
    calc_qwk,
)


def train_and_evaluate(
    source_data,
    target_data,
    embedding_dict,
    weight_dict,
    dev_mask,
    args,
    device,
):

    # Create predictor
    print('Creating predictor model...')
    mlp_data = {}
    mlp_data['y_source'] = source_data['scaled_score']
    mlp_data['y_dev'] = target_data['scaled_score'][dev_mask]
    mlp_data['y_target'] = target_data['scaled_score'][~dev_mask]
    if args.pred_model == 'mlp':
        pred_model = MLP(input_feature=embedding_dict[1].shape[0]).to(device)
        mlp_data['x_source'] = np.array([embedding_dict[eid] for eid in source_data['essay_id']])
        mlp_data['x_dev'] = np.array([embedding_dict[eid] for eid in target_data['essay_id'][dev_mask]])
        mlp_data['x_target'] = np.array([embedding_dict[eid] for eid in target_data['essay_id'][~dev_mask]])
    elif args.pred_model == 'features_model':
        pred_model = FeaturesModel().to(device)
        mlp_data['x_source'] = source_data['ridley_feature']
        mlp_data['x_dev'] = target_data['ridley_feature'][dev_mask]
        mlp_data['x_target'] = target_data['ridley_feature'][~dev_mask]

    # For source weights
    source_weights = np.array([weight_dict[eid] for eid in source_data['essay_id']])
    if args.weights == 'original':
        pass
    elif args.weights == 'inverse':
        source_weights = 1 - source_weights # reverse the weight
    elif args.weights == 'uniform':
        source_weights = np.ones_like(source_weights) # use uniform weights
    else:
        raise ValueError(f"Invalid weight option: {args.weights}")

    fit_func(
        pred_model,
        mlp_data['x_source'],
        mlp_data['y_source'],
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        sample_weight=source_weights,
    )

    # For dev
    y_pred_dev = pred_func(
        pred_model,
        mlp_data['x_dev'],
        batch_size=args.batch_size,
        device=device
    )
    dev_loss = mean_squared_error(mlp_data['y_dev'], y_pred_dev)
    dev_qwk = calc_qwk(mlp_data['y_dev'], y_pred_dev, args.target_prompt_id, args.attribute_name)

    # For test
    y_pred_target = pred_func(
        pred_model,
        mlp_data['x_target'],
        batch_size=args.batch_size,
        device=device
    )
        
    test_qwk = calc_qwk(mlp_data['y_target'], y_pred_target, args.target_prompt_id, args.attribute_name)
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
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load embedding
    with open('./outputs/embedding/deberta-v3-large.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)

    # # load pseudo label
    # pseudo_df = pl.read_csv('./outputs/pseudo_labels/pseudo_label_by_features_model.csv')
    # pseudo_dict = dict(zip(pseudo_df['essay_id'].to_numpy(), pseudo_df['y_pred'].to_numpy()))
    
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    source_data, target_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        add_pos=False,
    )
    print(f'    Number of source samples: {len(source_data["essay_id"])}')
    print(f'    Number of target samples: {len(target_data["essay_id"])}')

    # Load Estimated Data Value
    data_value_df = pl.read_csv(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}.csv')
    vals = data_value_df['data_value'].to_numpy().astype(float)
    vmin, vmax = vals.min(), vals.max()
    data_value_df = data_value_df.with_columns(
        ((pl.col('data_value') - vmin) / (vmax - vmin)).alias('weight')
    )
    weight_dict = dict(
        zip(data_value_df['essay_id'].to_list(),
            data_value_df['weight'].to_list())
    )
    # 基本統計量
    weights = np.array(data_value_df['weight'].to_list())
    print(f"Count: {len(weights)}")
    print(f"Min   : {weights.min():.4f}")
    print(f"Max   : {weights.max():.4f}")
    print(f"Mean  : {weights.mean():.4f}")
    print(f"Median: {np.median(weights):.4f}\n")

    # ヒストグラムの度数とビン幅を計算
    counts, bin_edges = np.histogram(weights, bins=20)

    # 最も多いビンを基準にバーの長さを調整
    max_count = counts.max()
    scale = 40 / max_count  # 最大度数を40文字のバーに

    print("Histogram (bin_left-bin_right : count [bar])")
    for cnt, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
        bar = '█' * int(cnt * scale)
        print(f"{left:.2f}-{right:.2f} : {cnt:4d} {bar}")

    # Create dev_mask
    selected_dev_ids = select_diverse_subset(
        {essay_id: embedding_dict[essay_id] for essay_id in target_data['essay_id'] if essay_id in embedding_dict},
        args.dev_size,
        method=args.sampling,
        seed=args.seed
    )
    dev_mask = np.isin(target_data['essay_id'], selected_dev_ids)
    
    # Training
    best_test_qwk, best_dev_loss, best_dev_qwk = train_and_evaluate(
        source_data=source_data,
        target_data=target_data,
        embedding_dict=embedding_dict,
        weight_dict=weight_dict,
        dev_mask=dev_mask,
        args=args,
        device=device
    )
    print(f'Best Test QWK: {best_test_qwk:.4f}, Best Dev Loss: {best_dev_loss:.4f}, Best Dev QWK: {best_dev_qwk:.4f}')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V7')
    parser.add_argument('--run_name', type=str, default='train-weighted-MLP')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pred_model', type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'greedy', 'maxmin', 'kmeans++'])
    parser.add_argument('--weights', type=str, default='uniform', choices=['uniform', 'original', 'inverse'])

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)