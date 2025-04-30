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
from utils.pmaes_utils import PMAESDataSet, GetAllEssayRepresentations, TestSingleOverallScoring
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


def train_epoch(args, essay_encoder, scorer, pm_cl, optimizer, tr_s_loader, te_t_loader, target_prompt_id, epoch):
    print('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(target_prompt_id, epoch))
    if epoch == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            essay_encoder.train(True)
            scorer.train(True)
            optimizer.zero_grad()
            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = essay_encoder(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = scorer(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre.squeeze(), s_aes_label.squeeze())
            aes_loss.backward(retain_graph=True)
            optimizer.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, essay_encoder, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, essay_encoder, te_t_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            essay_encoder.train(True)
            scorer.train(True)
            pm_cl.train(True)
            optimizer.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = essay_encoder(s_pos_ids.to(args.device))
            t_essay_fea_1 = essay_encoder(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * pm_cl(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optimizer.step()
            # End First Step

            s_essay_fea_2 = essay_encoder(s_pos_ids.to(args.device))
            t_essay_fea_2 = essay_encoder(t_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = scorer(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2.squeeze(), s_aes_label.squeeze())

            cl_loss_2 = 0.5 * pm_cl(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optimizer.step()

            s_essay_fea_3 = essay_encoder(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = scorer(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3.squeeze(), s_aes_label.squeeze())
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optimizer.step()

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
    ###############
    # テストデータの分布を変えたい場合，普通に実験する場合はコメントアウト
    test_data_id_if_dev_200 = np.load(f'data/ex_data/test_data_id_if_dev_200_{target_prompt_id}.npy')
    mask = np.isin(test_data['essay_id'], test_data_id_if_dev_200)
    ###############
    target = {
        'prompt_id': test_data['essay_set'][mask],
        'essay': test_data['pos_x'][mask],
        'linguistic': test_data['feature'][mask],
        'readability': test_data['readability'][mask],
        'score': test_data['scaled_score'][mask],
    }
    print(f"len final target: {len(test_data['essay_set'][mask])}")

    target_for_cl = {
        'prompt_id': np.concatenate([test_data['essay_set'], dev_data['essay_set']], axis=0),
        'essay': np.concatenate([test_data['pos_x'], dev_data['pos_x']], axis=0),
        'linguistic': np.concatenate([test_data['feature'], dev_data['feature']], axis=0),
        'readability': np.concatenate([test_data['readability'], dev_data['readability']], axis=0),
        'score': np.concatenate([test_data['scaled_score'], dev_data['scaled_score']], axis=0),
    }

    tr_s_num, tr_t_num = len(source['prompt_id']), len(target_for_cl['prompt_id'])
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
    te_t_loader_for_cl = DataLoader(PMAESDataSet(**target_for_cl), batch_size=t_batch_size, shuffle=True)
    
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
    best_dev_qwk = -1
    best_test_qwk = -1
    best_loss = 1000
    for e_index in range(1, args.num_epochs+1):
        # Train
        train_epoch(args, essay_encoder, scorer, pm_cl, optims, tr_s_loader, te_t_loader_for_cl, target_prompt_id, e_index)
        # Dev
        if args.dev_size > 0:   
            dev_qwk, dev_loss = TestSingleOverallScoring(args, essay_encoder, scorer, va_s_loader, 'test', args.attribute_name)
        else:
            dev_qwk = 0
            dev_loss = 0
        # Pseudo
        pseudo_qwk, pseudo_loss = TestSingleOverallScoring(args, essay_encoder, scorer, pd_t_loader, 'test', args.attribute_name)
        # Test
        test_qwk, test_loss = TestSingleOverallScoring(args, essay_encoder, scorer, te_t_loader, 'test', args.attribute_name)
        print('Dev QWK: {:.4f} Dev Loss: {:.4f} Pseudo QWK: {:.4f} Pseudo Loss: {:.4f} Test QWK: {:.4f} Test Loss: {:.4f}'.format(dev_qwk, dev_loss, pseudo_qwk, pseudo_loss, test_qwk, test_loss))


        model_selection_loss = (1 - args.loss_lambda) * dev_loss + args.loss_lambda * pseudo_loss
        model_selection_qwk = (1 - args.loss_lambda) * dev_qwk + args.loss_lambda * pseudo_qwk
    
        if model_selection_qwk > best_dev_qwk:
            best_loss = model_selection_loss
            best_dev_qwk = model_selection_qwk
            best_test_qwk = test_qwk

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
        add_pos=True,
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
    # estimated_data_value = np.load(f'/Users/takumishibata/Documents/project/DVRL-AES/outputs/dvrl_v5/values_1_mlp_seed12_dev30_lambda1.0_otFalse.npy')


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
        qwk_high, dev_loss_high, dev_qwk_high = train_and_evaluate(
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
        qwk_low, dev_loss_low, dev_qwk_low = train_and_evaluate(
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
                'Dev QWK[High]': dev_qwk_high,
                'Dev QWK[Low]': dev_qwk_low,
            })
    
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PAES_attributes models")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V6')
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

    args = parser.parse_args()
    print(dict(args._get_kwargs()))
    main(args)