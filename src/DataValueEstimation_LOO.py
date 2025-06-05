"""Training on DVRL
任意の自動採点モデルにより作成した擬似ラベルを報酬計算に利用する手法
"""

import os
import torch
import numpy as np
import argparse
import torch
import wandb
import pickle

from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from dvrl.sampling import select_diverse_subset
from data_valuation.core.loo_valuator import LOOValuator


def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load embedding
    with open('./outputs/embedding/deberta-v3-large.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)
    
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    source_data, target_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        add_pos=False,
    )
    print(f'    Number of source samples: {len(source_data["essay_id"])}')
    print(f'    Number of target samples: {len(target_data["essay_id"])}')

    # Select dev data
    selected_dev_ids = select_diverse_subset(
        {essay_id: embedding_dict[essay_id] for essay_id in target_data['essay_id'] if essay_id in embedding_dict},
        args.dev_size,
        method=args.sampling,
        seed=args.seed
    )
    dev_mask = np.isin(target_data['essay_id'], selected_dev_ids)

    ###################################################
    # Step2. Training DVRL
    ###################################################
    # Create predictor
    print('Creating predictor model...')
    dvrl_data = {}
    dvrl_data['y_source'] = source_data['scaled_score']
    dvrl_data['y_dev'] = target_data['scaled_score'][dev_mask]
    if args.pred_model == 'mlp':
        dvrl_data['x_source'] = np.array([embedding_dict[eid] for eid in source_data['essay_id']])
        dvrl_data['x_dev'] = np.array([embedding_dict[eid] for eid in target_data['essay_id'][dev_mask]])
    elif args.pred_model == 'features_model':
        dvrl_data['x_source'] = source_data['ridley_feature']
        dvrl_data['x_dev'] = target_data['ridley_feature'][dev_mask]
    

    # Network parameters
    print('Initialize loo framework...')
    # LOO Valuation
    valuator = LOOValuator(
        prompt_id=target_prompt_id,
        device=device,
        seed=args.seed,
        wandb_logging=args.wandb,
        wandb_project=args.pjname,
        wandb_name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}',
    )
    estimated_values = valuator.estimate_values(
        x_train=dvrl_data['x_source'],
        y_train= dvrl_data['y_source'],
        x_val= dvrl_data['x_dev'],
        y_val= dvrl_data['y_dev'],
        sample_ids=source_data['essay_id'],
    )

    print('Saving DVRL...')
    output_dir = f'./outputs/{args.pjname}'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}.csv'
    valuator.save_values(estimated_values, os.path.join(output_dir, filename))

    if args.wandb:
        wandb.alert(title=args.pjname, text='Training finished!')
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="LOO")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='LOO-V7')
    parser.add_argument('--run_name', type=str, default='Valuation')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pred_model',type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'greedy', 'maxmin', 'kmeans++'])
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)