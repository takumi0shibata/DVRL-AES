'''
This script trains a PAES model on the target prompt using the estimated data values.
推定されたデータ価値を[0, 1]にリスケールして損失関数を重みつき計算する
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
from utils.general_utils import set_seed, get_min_max_scores
from dvrl.dataset import EssayDataset
from dvrl.sampling import select_diverse_subset
from models.paes import PAES
from sklearn.metrics import cohen_kappa_score


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        pos_x, linguistic_x, readability_x, labels, essay_set, weights = [
            b.to(device) for b in batch
        ]
        optimizer.zero_grad()
        outputs = model(pos_x, linguistic_x, readability_x).squeeze()
        # 個別損失を計算
        losses = loss_fn(outputs, labels.squeeze())  # shape=(batch,)
        # 重みを掛けて平均
        loss = (losses * weights.squeeze()).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> dict:
    """
    Evaluate the model over one epoch on un-weighted data.

    Returns:
      - 'loss': mean squared error per sample
      - 'lwk' : list of linear-weighted kappa scores per essay set
      - 'qwk' : average quadratic-weighted kappa across all essay sets
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    all_preds  = []
    all_labels = []
    all_sets   = []

    with torch.no_grad():
        for pos_x, linguistic_x, readability_x, labels, essay_set in data_loader:
            # device に移動
            pos_x         = pos_x.to(device)
            linguistic_x  = linguistic_x.to(device)
            readability_x = readability_x.to(device)
            labels        = labels.to(device)
            essay_set     = essay_set.to(device)

            # 順伝播して flatten
            outputs = model(pos_x, linguistic_x, readability_x)
            preds   = outputs.view(-1)          # → shape=(batch,)
            targets = labels.view(-1)           # → shape=(batch,)
            sets    = essay_set.view(-1)        # → shape=(batch,)

            # per-sample MSE
            per_sample = loss_fn(preds, targets)
            total_loss += per_sample.sum().item()
            total_count += per_sample.numel()

            # QWK/LWK 用に蓄積（常にリストを返す）
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
            all_sets.extend(sets.cpu().tolist())

    # サンプルあたり平均 MSE
    avg_loss = total_loss / total_count if total_count > 0 else 0.0

    # QWK/LWK の計算
    qwk_list = []
    lwk_list = []
    for set_id in sorted(set(all_sets)):
        minscore, maxscore = get_min_max_scores()[set_id]['score']
        idx = [i for i, s in enumerate(all_sets) if s == set_id]
        if not idx:
            continue

        true_scores = np.round(
            minscore + (maxscore - minscore) * np.array(all_labels)[idx]
        ).astype(int)
        pred_scores = np.round(
            minscore + (maxscore - minscore) * np.array(all_preds)[idx]
        ).astype(int)

        label_range = list(range(minscore, maxscore + 1))
        qwk_list.append(
            cohen_kappa_score(true_scores, pred_scores,
                              labels=label_range, weights='quadratic')
        )
        lwk_list.append(
            cohen_kappa_score(true_scores, pred_scores,
                              labels=label_range, weights='linear')
        )

    avg_qwk = float(np.mean(qwk_list)) if qwk_list else 0.0

    return {
        'loss': avg_loss,
        'lwk' : lwk_list,
        'qwk' : avg_qwk
    }


def train_and_evaluate(
    source_data,
    target_data,
    # pseudo_dict,
    weight_dict,
    dev_mask,
    args,
    device,
) -> tuple[float, float, float]:

    # Initialize the model
    source = {
        'pos_x': source_data['pos_x'],
        'linguistic_x': source_data['feature'],
        'readability_x': source_data['readability'],
        'normalized_label': source_data['scaled_score'],
        'essay_set': source_data['essay_set'],
    }

    dev = {
        'pos_x': target_data['pos_x'][dev_mask],
        'linguistic_x': target_data['feature'][dev_mask],
        'readability_x': target_data['readability'][dev_mask],
        'normalized_label': target_data['scaled_score'][dev_mask],
        'essay_set': target_data['essay_set'][dev_mask],
    }

    # pseudo = {
    #     'pos_x': target_data['pos_x'][~dev_mask],
    #     'linguistic_x': target_data['feature'][~dev_mask],
    #     'readability_x': target_data['readability'][~dev_mask],
    #     'normalized_label': np.array([pseudo_dict[eid] for eid in target_data['essay_id'][~dev_mask]]),
    #     'essay_set': target_data['essay_set'][~dev_mask],
    # }

    target = {
        'pos_x': target_data['pos_x'][~dev_mask],
        'linguistic_x': target_data['feature'][~dev_mask],
        'readability_x': target_data['readability'][~dev_mask],
        'normalized_label': target_data['scaled_score'][~dev_mask],
        'essay_set': target_data['essay_set'][~dev_mask],
    }

    # Create datasets
    source_weights = np.array([weight_dict[eid] for eid in source_data['essay_id']])
    if args.weights == 'original':
        pass
    elif args.weights == 'inverse':
        source_weights = 1 - source_weights # reverse the weight
    elif args.weights == 'uniform':
        source_weights = np.ones_like(source_weights) # use uniform weights
    else:
        raise ValueError(f"Invalid weight option: {args.weights}")
    source_dataset = TensorDataset(
        torch.tensor(source['pos_x'], dtype=torch.int32),
        torch.tensor(source['linguistic_x'], dtype=torch.float32),
        torch.tensor(source['readability_x'], dtype=torch.float32),
        torch.tensor(source['normalized_label'], dtype=torch.float32),
        torch.tensor(source['essay_set'], dtype=torch.int32),
        torch.tensor(source_weights, dtype=torch.float32),
    ) 

    dev_dataset = TensorDataset(
        torch.tensor(dev['pos_x'], dtype=torch.int32),
        torch.tensor(dev['linguistic_x'], dtype=torch.float32),
        torch.tensor(dev['readability_x'], dtype=torch.float32),
        torch.tensor(dev['normalized_label'], dtype=torch.float32),
        torch.tensor(dev['essay_set'], dtype=torch.int32),
    )
    # pseudo_dataset = TensorDataset(
    #     torch.tensor(pseudo['pos_x'], dtype=torch.int32),
    #     torch.tensor(pseudo['linguistic_x'], dtype=torch.float32),
    #     torch.tensor(pseudo['readability_x'], dtype=torch.float32),
    #     torch.tensor(pseudo['normalized_label'], dtype=torch.float32),
    #     torch.tensor(pseudo['essay_set'], dtype=torch.int32),
    # )
    target_dataset = TensorDataset(
        torch.tensor(target['pos_x'], dtype=torch.int32),
        torch.tensor(target['linguistic_x'], dtype=torch.float32),
        torch.tensor(target['readability_x'], dtype=torch.float32),
        torch.tensor(target['normalized_label'], dtype=torch.float32),
        torch.tensor(target['essay_set'], dtype=torch.int32),
    )

    train_loader = DataLoader(dataset=source_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=True)
    # pseudo_loader = DataLoader(dataset=pseudo_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=target_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    model = PAES(source_data['max_sentnum'], source_data['max_sentlen'], source_data['pos_vocab']).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='none').to(device)

    # Training loop
    best_dev_qwk = -1
    best_test_qwk = -1
    best_dev_loss = 1000
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        # Dev
        dev_history = evaluate_epoch(model, dev_loader, loss_fn, device)
        dev_loss, dev_qwk = dev_history['loss'], dev_history['qwk']
        # Test
        test_history = evaluate_epoch(model, test_loader, loss_fn, device)
    
        if dev_qwk > best_dev_qwk:
            best_dev_loss = dev_loss
            best_dev_qwk = dev_qwk
            best_test_qwk = test_history['qwk']

        print(f"    Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev QWK: {dev_qwk:.4f}, Test QWK: {test_history['qwk']:.4f}, Best Test QWK: {best_test_qwk:.4f}")
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'dev_loss': dev_loss,
                'dev_qwk': dev_qwk,
                'test_qwk': test_history['qwk'],
                'best_test_qwk': best_test_qwk
            })

    return best_test_qwk, best_dev_loss, best_dev_qwk

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
        add_pos=True,
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
        # pseudo_dict=pseudo_dict,
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
    parser.add_argument('--run_name', type=str, default='train-weighted-PAES')
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
    parser.add_argument('--weights', type=str, default='uniform', choices=['uniform', 'original', 'inverse'])
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)