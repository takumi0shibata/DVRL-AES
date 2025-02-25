import os
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
import wandb
import torch

from models.PMAES import EssayEncoder, Scorer, PromptMappingCL
from utils.pmaes_utils import PMAESDataSet, TrainSingleOverallScoring
from utils.dvrl_utils import remove_top_p_sample
from utils.load_data import load_data_PMAES


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
    args,
    p: float,
    estimated_data_value: np.ndarray,
    ascending: bool,
    data: dict,
    epochs: int,
) -> tuple[float, float, float]:
    weights = remove_top_p_sample(
        estimated_data_value,
        top_p=p,
        ascending=ascending
    )
    weights = (np.array(weights) == 1)
    train_prompt_ids_tmp = data['source_prompts'][weights]
    train_essay_pos_tmp = data['x_source'][weights]
    train_linguistic_tmp = data['x_source_linguistic_features'][weights]
    train_readability_tmp = data['x_source_readability'][weights]
    train_score_tmp = data['y_source'][weights]

    tr_s_num, tr_t_num = len(train_essay_pos_tmp), len(data['x_target'])
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
    tr_s_loader = DataLoader(
        PMAESDataSet(
            train_prompt_ids_tmp,
            train_essay_pos_tmp,
            train_linguistic_tmp,
            train_readability_tmp,
            train_score_tmp
        ),
        batch_size=s_batch_size
    )
    va_s_loader = DataLoader(
        PMAESDataSet(
            data['dev_prompts'],
            data['x_dev'],
            data['x_dev_linguistic_features'],
            data['x_dev_readability'],
            data['y_dev']
        ),
        batch_size=s_batch_size
    )
    te_t_loader = DataLoader(
        PMAESDataSet(
            data['target_prompts'],
            data['x_target'],
            data['x_target_linguistic_features'],
            data['x_target_readability'],
            data['y_target']
        ),
        batch_size=t_batch_size
    )

    essay_encoder = EssayEncoder(
        args,
        max_num=data['max_sentnum'],
        max_len=data['max_sentlen'],
        embed_dim=args.embedding_dim,
        pos_vocab=data['pos_vocab']
    ).to(args.device)
    scorer = Scorer(args).to(args.device)
    pm_cl = PromptMappingCL(args, tr_s_num, tr_t_num).to(args.device)
    optims = torch.optim.Adam([{'params': essay_encoder.parameters()}, {'params': scorer.parameters()}, {'params': pm_cl.parameters()}], lr=args.learning_rate)
    tr_log = {
        'Epoch_best_dev_qwk': [0, 0, 0],
        'Best_dev_qwk': [0, 0],
    }

    best_loss = 1000
    best_dev_qwk = 0
    for e_index in range(1, epochs+1):

        va_loss, te_loss, va_qwk, te_qwk, best_qwk = TrainSingleOverallScoring(args,
                                                                                essay_encoder, scorer, pm_cl, optims,
                                                                                tr_s_loader, va_s_loader, te_t_loader,
                                                                                args.target_id, e_index,
                                                                                tr_log, args.attribute_name
                                                                                )
        if va_loss < best_loss:
            best_loss = va_loss
            best_dev_qwk = va_qwk

    return best_dev_qwk, best_qwk, best_loss

def main(args):
    target_id = args.target_id
    seed = args.seed
    seed_all(seed)
    estimated_data_value = np.load(f'outputs/Estimated_Data_Values/{args.valuation_method}/estimated_data_value{target_id}.npy')

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name+str(target_id),
            config=args
        )

    data = load_data_PMAES(
        f'data/cross_prompt_attributes/{target_id}/',
        args.attribute_name,
        args.embedding_model,
        args.device,
        devsize=args.dev_size
    )

    for p in np.arange(0.1, 1.0, 0.1):
        ################################################
        # データの価値が低いものを削除
        ################################################
        seed_all(seed)
        best_dev_qwk_high, best_qwk_high, best_loss_high = train_and_evaluate(
            args,
            p,
            estimated_data_value,
            False,
            data,
            args.num_epochs,
        )

        ################################################
        # データの価値が高いものを削除
        ################################################
        seed_all(seed)
        best_dev_qwk_low, best_qwk_low, best_loss_low = train_and_evaluate(
            args,
            p,
            estimated_data_value,
            True,
            data,
            args.num_epochs,
        )
    
        if args.wandb:
            wandb.log({
                'p': p,
                'best_dev_qwk_high': best_dev_qwk_high,
                'best_qwk_high': best_qwk_high,
                'best_dev_loss_high': best_loss_high,
                'best_dev_qwk_low': best_dev_qwk_low,
                'best_qwk_low': best_qwk_low,
                'best_dev_loss_low': best_loss_low
            })

    if args.wandb:
        wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PAES_attributes models")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL', choices=['DVRL', 'LOO', 'DataShapley'])
    parser.add_argument('--run_name', type=str, default='train-PMAES')
    parser.add_argument('--target_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument(
        '--valuation_method',
        default='DVRL-pos',
        choices=[
            'DVRL-pos',
            'LOO-pos',
            'DataShapley-pos',
        ],
    )
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

    args = parser.parse_args()
    print(dict(args._get_kwargs()))
    main(args)