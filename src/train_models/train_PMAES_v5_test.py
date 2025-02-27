import os
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
import wandb
import torch
import torch.nn as nn
import polars as pl
from tqdm import tqdm

# my packages
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.PMAES import EssayEncoder, Scorer, PromptMappingCL
from utils.pmaes_utils import PMAESDataSet, GetAllEssayRepresentations, TestSingleOverallScoring, TrainSingleOverallScoring
from utils.dvrl_utils import remove_top_p_sample
from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset


def seed_all(seed_value):
    """
    Setting the random seed across the pipeline
    """
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value)
    np.random.seed(seed_value) # cpu vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def train_and_evaluate(
    train_data,
    dev_data,
    test_data,
    pseudo_dict,
    target_prompt_id,
    weights,
    device,
    args,
) -> tuple[float, float, float]:

    weights = (torch.tensor(weights, dtype=torch.float) == 1)

    # Initialize the model
    source = {
        'prompt_id': train_data['essay_set'][weights], 
        'essay': train_data['pos_x'][weights],
        'linguistic': train_data['feature'][weights],
        'readability': train_data['readability'][weights],
        'score': train_data['scaled_score'][weights],
    }

    dev = {
        'prompt_id': dev_data['essay_set'],
        'essay': dev_data['pos_x'],
        'linguistic': dev_data['feature'],
        'readability': dev_data['readability'],
        'score': dev_data['scaled_score'],
    }

    pseudo = {
        'prompt_id': test_data['essay_set'],
        'essay': test_data['pos_x'],
        'linguistic': test_data['feature'],
        'readability': test_data['readability'],
        'score': np.array([pseudo_dict[eid] for eid in test_data['essay_id']]),
    }

    target = {
        'prompt_id': test_data['essay_set'],
        'essay': test_data['pos_x'],
        'linguistic': test_data['feature'],
        'readability': test_data['readability'],
        'score': test_data['scaled_score'],
    }

    tr_s_num, tr_t_num = len(source['prompt_id']), len(target['prompt_id'])
    batch_num = args.batch_num
    s_batch_size = int(tr_s_num / batch_num)
    t_batch_size = int(tr_t_num / batch_num)
    s_batch_num = int(tr_s_num / s_batch_size)
    t_batch_num = int(tr_t_num / t_batch_size)
    while t_batch_num > s_batch_num:
        s_batch_size -= 1
        s_batch_num = int(tr_s_num / s_batch_size)

    print('s_batch_size: {}'.format(s_batch_size))
    print('t_batch_size: {}'.format(t_batch_size))
    tr_s_loader = DataLoader(PMAESDataSet(**source), batch_size=s_batch_size, shuffle=True)
    if args.dev_size > 0:
        va_s_loader = DataLoader(PMAESDataSet(**dev), batch_size=s_batch_size, shuffle=True)
    pd_t_loader = DataLoader(PMAESDataSet(**pseudo), batch_size=s_batch_size, shuffle=True)
    te_t_loader = DataLoader(PMAESDataSet(**target), batch_size=t_batch_size, shuffle=True)

    essay_encoder = EssayEncoder(
        args,
        max_num=train_data['max_sentnum'],
        max_len=train_data['max_sentlen'],
        embed_dim=args.embedding_dim,
        pos_vocab=train_data['pos_vocab']
    ).to(args.device)
    scorer = Scorer(args).to(args.device)
    pm_cl = PromptMappingCL(args, tr_s_num, tr_t_num).to(args.device)
    optims = torch.optim.Adam([{'params': essay_encoder.parameters()}, {'params': scorer.parameters()}, {'params': pm_cl.parameters()}], lr=args.learning_rate)

    # Training loop
    tr_log = {
        'Epoch_best_dev_qwk': [0, 0, 0],
        'Best_dev_qwk': [0, 0],
    }
    epochs = 50
    for e_index in range(1, epochs+1):
        TrainSingleOverallScoring(args,
                                  essay_encoder, scorer, pm_cl, optims,
                                  tr_s_loader, va_s_loader, te_t_loader,
                                  args.target_prompt_id, e_index,
                                  tr_log, args.attribute_name)

    return None, None

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
        add_pos=True,
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
    # estimated_data_value = np.load(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}.npy')
    estimated_data_value = np.load(f'/Users/takumishibata/Documents/project/DVRL-AES/outputs/dvrl_v5/values_1_mlp_seed12_dev30_lambda1.0_otFalse.npy')


    for p in np.arange(0.0, 1.0, 0.1):
        ################################################
        # データの価値が低いものを削除
        ################################################
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p,
            ascending=False,
        )
        set_seed(args.seed)
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

        ################################################
        # データの価値が高いものを削除
        ################################################
        weights = remove_top_p_sample(
            estimated_data_value,
            top_p=p,
            ascending=True,
        )
        set_seed(args.seed)
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
                'p': p,
                'QWK[High]': qwk_high,
                'QWK[Low]': qwk_low,
                'Dev Loss[High]': dev_loss_high,
                'Dev Loss[Low]': dev_loss_low,
            })
    
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PAES_attributes models")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V5')
    parser.add_argument('--run_name', type=str, default='train-PMAES')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--source2target', type=str, default='many2one', help='Setting of source-target pair')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--filter_num', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='filter length in 1st conv layer')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=50, help='Num of hidden units in recurrent layer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for layers')
    parser.add_argument('--batch_num', type=int, default=35, help='Number of batches')
    parser.add_argument('--max_sentlen', type=int, default=50, help='Max sentence length')
    parser.add_argument('--max_sentnum', type=int, default=100, help='Max sentence number')
    parser.add_argument('--pred_model', type=str, default='features_model')
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--ot', action='store_true')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))
    main(args)
            
        