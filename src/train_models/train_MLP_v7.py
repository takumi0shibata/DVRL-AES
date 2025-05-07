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
    select_non_top_essay_ids_by_percent
)


def train_and_evaluate(
    source_data,
    target_data,
    pseudo_dict,
    embedding_dict,
    top_essay_ids,
    dev_mask,
    args,
    device,
):
    
    # Build value_mask over source_data
    source_mask = np.array([
        essay_id in top_essay_ids for essay_id in source_data['essay_id']
    ])
    print(f'{np.sum(source_mask)} / {len(source_data["essay_id"])}')

    # Create predictor
    print('Creating predictor model...')
    mlp_data = {}
    mlp_data['y_source'] = source_data['scaled_score'][source_mask]
    mlp_data['y_dev'] = target_data['scaled_score'][dev_mask]
    mlp_data['y_target'] = target_data['scaled_score'][~dev_mask]
    mlp_data['y_pseudo'] = np.array([pseudo_dict[eid] for eid in target_data['essay_id'][~dev_mask]])
    if args.pred_model == 'mlp':
        pred_model = MLP(input_feature=embedding_dict[1].shape[0]).to(device)
        mlp_data['x_source'] = np.array([embedding_dict[eid] for eid in source_data['essay_id'][source_mask]])
        mlp_data['x_dev'] = np.array([embedding_dict[eid] for eid in target_data['essay_id'][dev_mask]])
        mlp_data['x_target'] = np.array([embedding_dict[eid] for eid in target_data['essay_id'][~dev_mask]])
    elif args.pred_model == 'features_model':
        pred_model = FeaturesModel().to(device)
        mlp_data['x_source'] = source_data['ridley_feature'][source_mask]
        mlp_data['x_dev'] = target_data['ridley_feature'][dev_mask]
        mlp_data['x_target'] = target_data['ridley_feature'][~dev_mask]

    fit_func(
        pred_model,
        mlp_data['x_source'],
        mlp_data['y_source'],
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )

    # For test
    y_pred_target = pred_func(
        pred_model,
        mlp_data['x_target'],
        batch_size=args.batch_size,
        device=device
    )

    # For dev
    if args.loss_lambda != 1.0:
        y_pred_dev = pred_func(
            pred_model,
            mlp_data['x_dev'],
            batch_size=args.batch_size,
            device=device
        )
        dev_loss = mean_squared_error(mlp_data['y_dev'], y_pred_dev)
        pseudo_loss = mean_squared_error(mlp_data['y_pseudo'], y_pred_target)
        dev_qwk = calc_qwk(mlp_data['y_dev'], y_pred_dev, args.target_prompt_id, args.attribute_name)
        pseudo_qwk = calc_qwk(mlp_data['y_pseudo'], y_pred_target, args.target_prompt_id, args.attribute_name)

        dev_loss = (1 - args.loss_lambda) * dev_loss + args.loss_lambda * pseudo_loss
        dev_qwk = (1 - args.loss_lambda) * dev_qwk + args.loss_lambda * pseudo_qwk
    else:
        dev_loss = mean_squared_error(mlp_data['y_pseudo'], y_pred_target)
        dev_qwk = calc_qwk(mlp_data['y_pseudo'], y_pred_target, args.target_prompt_id, args.attribute_name)
    
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

    # load pseudo label
    pseudo_df = pl.read_csv(f'./outputs/pseudo_labels/PAES_pred_{target_prompt_id}_seed{args.seed}.csv')
    pseudo_dict = dict(zip(pseudo_df['essay_id'].to_numpy(), pseudo_df['pred'].to_numpy()))
    
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
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'greedy', 'maxmin', 'kmeans++'])
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)