from torch.utils.data import Dataset
import torch
from six import string_types
import logging
import numpy as np
from sklearn.metrics import confusion_matrix


class PMAESDataSet(Dataset):
    def __init__(self, prompt_id, essay, linguistic, readability, score):
        super(PMAESDataSet, self).__init__()
        self.prompt_id = prompt_id
        self.essay = essay
        self.linguistic = linguistic
        self.readability = readability
        self.score = score

    def __len__(self):
        return len(self.score)

    def __getitem__(self, item):
        return {
            'prompt': self.prompt_id[item],
            'pos_ids': torch.tensor(self.essay[item], dtype=torch.long),
            'ling': torch.tensor(self.linguistic[item], dtype=torch.float),
            'read': torch.tensor(self.readability[item], dtype=torch.float),
            'score': torch.tensor(self.score[item], dtype=torch.float),
        }
    

def kappa(y_true , y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        logger.error("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k


import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import sys


def get_logger(name, level=logging.INFO, handler=sys.stdout, formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

random.seed(111)
torch.manual_seed(111)
logger = get_logger(name='Train...')


def mask_mse_loss_fn(label, predict):
    loss_fn = nn.MSELoss()
    mask_value = -1
    mask = torch.not_equal(label, mask_value)
    mse = loss_fn(label * mask, predict * mask)
    return mse


def mask_qwk(labels, predict, prompts, target_prompt):
    for index, prompt in enumerate(prompts):
        if prompt == target_prompt:
            labels[index] = 0
            predict[index] = 0
    return kappa(labels, predict, weights='quadratic') * 100


def GetAllEssayRepresentations(args, Gmodel, loader):
    prompt_essay_embed = torch.tensor([]).to('cpu')
    for item in loader:
        prompt, pos_ids, ling, read = item['prompt'], item['pos_ids'], item['ling'], item['read']
        with torch.no_grad():
            essay_embed = Gmodel(pos_ids.to(args.device)).to('cpu')
        prompt_essay_embed = torch.cat([prompt_essay_embed, essay_embed], dim=0)
    return prompt_essay_embed


def get_prompt_essay_embed_with_feature(args, Gmodel, loader):
    prompt_essay_embed = torch.tensor([]).to('cpu')
    for item in loader:
        prompt, pos_ids, ling, read = item['prompt'], item['pos_ids'], item['ling'], item['read']
        with torch.no_grad():
            essay_embed = Gmodel(pos_ids.to(args.device)).to('cpu')
            essay_embed = torch.cat([essay_embed, read, ling], dim=1)
        prompt_essay_embed = torch.cat([prompt_essay_embed, essay_embed], dim=0)
    return prompt_essay_embed


def get_min_max_scores():
    return {
        1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6)},
        8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12)}}

def TransferScoreForSingleTrait(label, predict, prompt, mode, trait):
    label = label.squeeze(-1).detach().numpy()
    predict = predict.squeeze(-1).detach().numpy()
    prompt = prompt.detach().numpy()
    transfer_label = []
    transfer_predict = []
    max_min_scores = get_min_max_scores()
    for i in range(len(prompt)):
        if mode == 'valid':
            s_r = [0, 10]
        else:
            s_r = max_min_scores[prompt[i]][trait]
        transfer_label += [round(label[i] * (s_r[1]-s_r[0]) + s_r[0])]
        transfer_predict += [round(predict[i] * (s_r[1]-s_r[0]) + s_r[0])]

    return transfer_label, transfer_predict

def TestSingleOverallScoring(args, essay_encoder, scorer, loader, mode, attribute_name):
    assert mode in ['valid', 'test']
    essay_encoder.eval()
    scorer.eval()

    aes_label_all = np.array([], dtype=int)
    aes_pre_all = np.array([], dtype=int)
    total_loss = 0.
    with torch.no_grad():
        for item in loader:
            prompt, essay, linguistic, readability, score = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
            essay_fea = essay_encoder(essay.to(args.device))
            fea_cat = torch.cat([essay_fea, linguistic.to(args.device), readability.to(args.device)], dim=1)
            aes_pre = scorer(fea_cat)
            aes_pre = aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(aes_pre.squeeze(), score.squeeze())

            total_loss += aes_loss
            score, aes_pre = TransferScoreForSingleTrait(score, aes_pre, prompt, mode, attribute_name)
            aes_label_all = np.concatenate([aes_label_all, score])
            aes_pre_all = np.concatenate([aes_pre_all, aes_pre])
    qwk = kappa(aes_label_all, aes_pre_all, weights='quadratic') * 100
    loss = total_loss / len(loader)
    return qwk, loss


def TestSingleOverallScoringForMultiTarget(args, essay_encoder, Smodel, loader, mode, attribute_name):
    assert mode in ['valid', 'test']
    essay_encoder.eval()
    Smodel.eval()
    total_loss = 0.

    if mode == 'valid':
        aes_label_all = np.array([], dtype=int)
        aes_pre_all = np.array([], dtype=int)
        with torch.no_grad():
            for item in loader:
                prompt, essay, linguistic, readability, score = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
                essay_fea = essay_encoder(essay.to(args.device))
                fea_cat = torch.cat([essay_fea, linguistic.to(args.device), readability.to(args.device)], dim=1)
                aes_pre = Smodel(fea_cat)
                aes_pre = aes_pre.to('cpu')
                aes_loss = nn.MSELoss()(aes_pre.squeeze(), score.squeeze())

                total_loss += aes_loss
                score, aes_pre = TransferScoreForSingleTrait(score, aes_pre, prompt, mode, attribute_name)
                aes_label_all = np.concatenate([aes_label_all, score])
                aes_pre_all = np.concatenate([aes_pre_all, aes_pre])
        qwk = kappa(aes_label_all, aes_pre_all, weights='quadratic') * 100
        total_loss = total_loss / len(loader)
    else:
        aes_label_all = {}
        aes_pre_all = {}
        with torch.no_grad():
            for item in loader:
                prompt, essay, linguistic, readability, score = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
                essay_fea = essay_encoder(essay.to(args.device))
                fea_cat = torch.cat([essay_fea, linguistic.to(args.device), readability.to(args.device)], dim=1)
                aes_pre = Smodel(fea_cat)
                aes_pre = aes_pre.to('cpu')
                score, aes_pre = TransferScoreForSingleTrait(score, aes_pre, prompt, mode, attribute_name)
                for index in range(len(prompt)):
                    p = prompt.numpy()[index]
                    aes_label_all[p] = aes_label_all.get(p, []) + [score[index]]
                    aes_pre_all[p] = aes_pre_all.get(p, []) + [aes_pre[index]]
        qwk = {}
        for key in aes_label_all.keys():
            score = aes_label_all[key]
            aes_pre = aes_pre_all[key]
            prompt_qwk = round(kappa(score, aes_pre, weights='quadratic') * 100, 2)
            qwk[key] = prompt_qwk
    return qwk, total_loss


def TransferScoreForSingleTrait_dev100(label, predict, prompt, mode, trait):
    label = label.squeeze(-1).detach().numpy()
    predict = predict.squeeze(-1).detach().numpy()
    prompt = prompt.detach().numpy()
    transfer_label = []
    transfer_predict = []
    max_min_scores = get_min_max_scores()
    for i in range(len(prompt)):
        if mode == 'valid':
            s_r = [0, 100]
        else:
            s_r = max_min_scores[prompt[i]][trait]
        transfer_label += [round(label[i] * (s_r[1]-s_r[0]) + s_r[0])]
        transfer_predict += [round(predict[i] * (s_r[1]-s_r[0]) + s_r[0])]

    return transfer_label, transfer_predict

def TestForSingleTrait_dev100(args, Gmodel, Smodel, loader, mode, attribute_name):
    assert mode in ['valid', 'test']
    Gmodel.eval()
    Smodel.eval()

    aes_label_all = np.array([], dtype=int)
    aes_pre_all = np.array([], dtype=int)
    total_loss = 0.
    with torch.no_grad():
        for item in loader:
            prompt, essay_ids, ling, read, aes_label = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
            essay_fea = Gmodel(essay_ids.to(args.device))
            fea_cat = torch.cat([essay_fea, ling.to(args.device), read.to(args.device)], dim=1)
            aes_pre = Smodel(fea_cat)
            aes_pre = aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(aes_pre, aes_label)

            total_loss += aes_loss
            aes_label, aes_pre = TransferScoreForSingleTrait_dev100(aes_label, aes_pre, prompt, mode, attribute_name)
            aes_label_all = np.concatenate([aes_label_all, aes_label])
            aes_pre_all = np.concatenate([aes_pre_all, aes_pre])
    qwk = kappa(aes_label_all, aes_pre_all, weights='quadratic') * 100
    loss = total_loss / len(loader)
    return qwk, loss


def get_score_vector_positions():
    return {
        'score': 0,
        'content': 1,
        'organization': 2,
        'word_choice': 3,
        'sentence_fluency': 4,
        'conventions': 5,
        'prompt_adherence': 6,
        'language': 7,
        'narrativity': 8,
        # 'style': 9,
        # 'voice': 10
    }

def TransferScoreForMultiTrait(all_label, all_predict, label_list, predict_list, prompt_list, mode):
    label_list = label_list.detach().numpy()
    predict_list = predict_list.detach().numpy()
    prompt_list = prompt_list.detach().numpy()

    min_max_scores = get_min_max_scores()
    score_vector_positions = get_score_vector_positions()
    for index in range(len(prompt_list)):
        prompt = prompt_list[index]
        traits_name = min_max_scores[prompt].keys()
        label_line = label_list[index]
        predict_line = predict_list[index]
        for trait in traits_name:
            trait_position = score_vector_positions[trait]
            if label_line[trait_position] != -1:
                if mode == 'valid':
                    min_score, max_score = [0, 10]
                else:
                    min_score, max_score = min_max_scores[prompt][trait]

                trait_label = int(label_line[trait_position] * (max_score - min_score) + min_score)
                trait_predict = int(predict_line[trait_position] * (max_score - min_score) + min_score)
                all_label[trait].append(trait_label)
                all_predict[trait].append(trait_predict)

    return all_label, all_predict

def TestForMultiTrait(args, Gmodel, Smodel, loader, mode, trait_interactive_type):
    assert mode in ['valid', 'test']
    Gmodel.eval()
    Smodel.eval()

    aes_label_all = {
        'score': [],
        'content': [],
        'organization': [],
        'word_choice': [],
        'sentence_fluency': [],
        'conventions': [],
        'prompt_adherence': [],
        'language': [],
        'narrativity': []
    }
    aes_pre_all = {
        'score': [],
        'content': [],
        'organization': [],
        'word_choice': [],
        'sentence_fluency': [],
        'conventions': [],
        'prompt_adherence': [],
        'language': [],
        'narrativity': []
    }
    with torch.no_grad():
        for item in loader:
            prompt, essay_ids, ling, read, aes_label = item['prompt'], item['pos_ids'], item['ling'], item['read'], item['score']
            essay_fea = Gmodel(essay_ids.to(args.device))
            # fea_cat = torch.cat([essay_fea, ling.to(args.device), read.to(args.device)], dim=1)
            if trait_interactive_type in ['attention', 'none']:
                aes_pre = Smodel(essay_fea, ling.to(args.device), read.to(args.device))
            else:
                aes_pre, _ = Smodel(essay_fea, ling.to(args.device), read.to(args.device))
            aes_pre = aes_pre.to('cpu')
            aes_label_all, aes_pre_all = TransferScoreForMultiTrait(aes_label_all, aes_pre_all, aes_label, aes_pre, prompt, mode)

    qwk_result = {}
    average_qwk = 0
    for trait in aes_label_all.keys():
        label = aes_label_all[trait]
        predict = aes_pre_all[trait]
        if len(label) == 0:
            continue
        trait_qwk = kappa(label, predict, weights='quadratic') * 100
        qwk_result[trait] = trait_qwk
        average_qwk += trait_qwk

    average_qwk = np.mean(list(qwk_result.values()))
    qwk_result['Avg'] = average_qwk
    return qwk_result


def TrainSingleOverallScoring(args,
                              essay_encoder, scorer, pm_cl, optimizer,
                              tr_s_loader, va_s_loader, te_t_loader,
                              target_prompt_id, epoch,
                              tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(target_prompt_id, epoch))
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
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
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
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * pm_cl(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optimizer.step()

            s_essay_fea_3 = essay_encoder(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = scorer(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optimizer.step()
    if args.source2target == 'many2one' or args.source2target == 'one2one':
        va_qwk, va_loss = TestSingleOverallScoring(args, essay_encoder, scorer, va_s_loader, 'valid', attribute_name)
        te_qwk, te_loss = TestSingleOverallScoring(args, essay_encoder, scorer, te_t_loader, 'test', attribute_name)
        if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
            tr_log['Epoch_best_dev_qwk'][0] = va_qwk
            tr_log['Epoch_best_dev_qwk'][1] = te_qwk
            tr_log['Epoch_best_dev_qwk'][2] = epoch
        epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}'
        epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1])
        logger.info(epoch_msg)

        return va_loss, te_loss, va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1]

    else:  #  args.source2target == 'many2many' or args.source2target == 'one2many'
        va_qwk, va_loss = TestSingleOverallScoringForMultiTarget(args, essay_encoder, scorer, va_s_loader, 'valid', attribute_name)
        te_qwk, te_loss = TestSingleOverallScoringForMultiTarget(args, essay_encoder, scorer, te_t_loader, 'test', attribute_name)
        if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
            tr_log['Epoch_best_dev_qwk'][0] = va_qwk
            tr_log['Epoch_best_dev_qwk'][1] = str(te_qwk)
            tr_log['Epoch_best_dev_qwk'][2] = epoch

        epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{}, [BEST DEV IN TEST] QWK:{}'
        epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1])
        logger.info(epoch_msg)



def TrainSingleOverallScoringForMultiTarget(args,
                                            Gmodel, Smodel, FCmodel, optims,
                                            tr_s_loader, va_s_loader, te_t_loader,
                                            t_index, e_index,
                                            tr_log, attribute_name):

    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()
            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = str(te_qwk)
        tr_log['Epoch_best_dev_qwk'][2] = e_index

    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{}, [BEST DEV IN TEST] QWK:{}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1])
    logger.info(epoch_msg)


def TrainForSingleTraitDoublePCLDirectly(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
    t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
    for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
        Gmodel.train(True)
        Smodel.train(True)
        FCmodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
        t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

        # Start First Step
        s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
        t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
        cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
        first_loss = cl_loss_1
        first_loss.backward(retain_graph=True)
        optims.step()
        # End First Step

        s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
        t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
        s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre_2 = Smodel(s_fea_cat_2)
        s_aes_pre_2 = s_aes_pre_2.to('cpu')
        aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

        cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
        second_loss = aes_loss_2 + cl_loss_2
        second_loss.backward(retain_graph=True)
        optims.step()

        s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
        s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre_3 = Smodel(s_fea_cat_3)
        s_aes_pre_3 = s_aes_pre_3.to('cpu')
        aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
        third_loss = aes_loss_3
        third_loss.backward(retain_graph=True)
        optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)

    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)


def TrainForSingleTraitDoublePCL_dev100(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()
            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestForSingleTrait_dev100(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestForSingleTrait_dev100(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)

    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)


def TrainForSingleTraitSourcePCL(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)
    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)

def TrainForSingleTraitTargetPCL(args,
                                 Gmodel, Smodel, FCmodel, optims,
                                 tr_s_loader, va_s_loader, te_t_loader,
                                 t_index, e_index,
                                 tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if e_index == 1:
        for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']

            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre = Smodel(s_fea_cat)
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
            aes_loss.backward(retain_graph=True)
            optims.step()
    else:
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        for item_index, (s_item, t_item) in tqdm(enumerate(zip(tr_s_loader, te_t_loader), start=1), desc='Training......'):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(t_essay_fea_1, s_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))

            s_fea_cat_2 = torch.cat([s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_2 = Smodel(s_fea_cat_2)
            s_aes_pre_2 = s_aes_pre_2.to('cpu')
            aes_loss_2 = nn.MSELoss()(s_aes_pre_2, s_aes_label)

            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_2 = 0.5 * FCmodel(t_essay_fea_2, s_essay_embed.to(args.device))
            second_loss = aes_loss_2 + cl_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            s_fea_cat_3 = torch.cat([s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device)], dim=1)
            s_aes_pre_3 = Smodel(s_fea_cat_3)
            s_aes_pre_3 = s_aes_pre_3.to('cpu')
            aes_loss_3 = nn.MSELoss()(s_aes_pre_3, s_aes_label)
            third_loss = aes_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)


def TrainForSingleTraitNoPCL(args,
                             Gmodel, Smodel, optims,
                             tr_s_loader, va_s_loader, te_t_loader,
                             t_index, e_index,
                             tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
        Gmodel.train(True)
        Smodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                           s_item['read'], s_item['score']

        s_essay_fea = Gmodel(s_pos_ids.to(args.device))
        s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre = Smodel(s_fea_cat)
        s_aes_pre = s_aes_pre.to('cpu')
        aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
        aes_loss.backward(retain_graph=True)
        optims.step()

    va_qwk, va_loss = TestSingleOverallScoring(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoring(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = te_qwk
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = te_qwk
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{:.2f}, [BEST DEV IN TEST] QWK:{:.2f}, [LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)
    epoch_final_msg = '[TARGET] P:{} [EPOCH] E:{} [BATCH BEST DEV IN TEST] QWK:{:.2f} [BATCH LOWEST DEV IN TEST] QWK:{:.2f}'
    epoch_final_msg = epoch_final_msg.format(t_index, e_index, tr_log['Best_dev_qwk'][1], tr_log['Lowest_dev_loss'][1])
    logger.info(epoch_final_msg)


def TrainForSingleTraitNoPCLMultiTest(args,
                                      Gmodel, Smodel, optims,
                                      tr_s_loader, va_s_loader, te_t_loader,
                                      t_index, e_index,
                                      tr_log, attribute_name):
    logger.info('Train other epoch: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    for item_index, s_item in tqdm(enumerate(tr_s_loader, start=1), desc='Training......'):
        Gmodel.train(True)
        Smodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                           s_item['read'], s_item['score']

        s_essay_fea = Gmodel(s_pos_ids.to(args.device))
        s_fea_cat = torch.cat([s_essay_fea, s_ling.to(args.device), s_read.to(args.device)], dim=1)
        s_aes_pre = Smodel(s_fea_cat)
        s_aes_pre = s_aes_pre.to('cpu')
        aes_loss = nn.MSELoss()(s_aes_pre, s_aes_label)
        aes_loss.backward(retain_graph=True)
        optims.step()

    va_qwk, va_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, va_s_loader, 'valid', attribute_name)
    te_qwk, te_loss = TestSingleOverallScoringForMultiTarget(args, Gmodel, Smodel, te_t_loader, 'test', attribute_name)
    if va_qwk > tr_log['Epoch_best_dev_qwk'][0]:
        tr_log['Epoch_best_dev_qwk'][0] = va_qwk
        tr_log['Epoch_best_dev_qwk'][1] = str(te_qwk)
        tr_log['Epoch_best_dev_qwk'][2] = e_index
        tr_log['BestModel']['EpochBestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['EpochBestSmodel'] = Smodel.state_dict()

    if va_loss < tr_log['Epoch_lowest_dev_loss'][0]:
        tr_log['Epoch_lowest_dev_loss'][0] = va_loss
        tr_log['Epoch_lowest_dev_loss'][1] = str(te_qwk)
        tr_log['Epoch_lowest_dev_loss'][2] = e_index
    epoch_msg = '[EPOCH DEV] QWK:{:.2f}, [EPOCH TEST] QWK:{}, [BEST DEV IN TEST] QWK:{}, [LOWEST DEV IN TEST] QWK:{}'
    epoch_msg = epoch_msg.format(va_qwk, te_qwk, tr_log['Epoch_best_dev_qwk'][1], tr_log['Epoch_lowest_dev_loss'][1])
    logger.info(epoch_msg)


def TrainForMultiTraitWithCL(args,
                             Gmodel, Smodel, FCmodel, optims,
                             tr_s_loader, va_s_loader, te_t_loader,
                             t_index, e_index,
                             tr_log, trait_interactive_type, if_directly):
    logger.info('TrainForMultiTrait: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    if not if_directly and e_index == 1:
        print('分段训练， 现在是初始化模型参数。。。。。。')
        for item_index, (s_item, t_item) in enumerate(zip(tr_s_loader, te_t_loader), start=1):
            Gmodel.train(True)
            Smodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                               s_item['read'], s_item['score']
            s_essay_fea = Gmodel(s_pos_ids.to(args.device))
            if trait_interactive_type in ['attention', 'none']:
                s_aes_pre = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre = s_aes_pre.to('cpu')
                aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
                loss = aes_loss
            else:
                s_aes_pre, trait_loss = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre = s_aes_pre.to('cpu')
                aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
                loss = aes_loss + 0.5 * trait_loss
            loss.backward(retain_graph=True)
            optims.step()
    else:
        if if_directly:
            print('不分段训练，直接开始对比学习。。。。。。')
        else:
            print('分段训练，现在开始对比学习。。。。。。')
        s_essay_embed = GetAllEssayRepresentations(args, Gmodel, tr_s_loader)
        t_essay_embed = GetAllEssayRepresentations(args, Gmodel, te_t_loader)
        for item_index, (s_item, t_item) in enumerate(zip(tr_s_loader, te_t_loader), start=1):
            Gmodel.train(True)
            Smodel.train(True)
            FCmodel.train(True)
            optims.zero_grad()

            s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], s_item['read'], s_item['score']
            t_prompt, t_pos_ids, t_ling, t_read, t_aes_label = t_item['prompt'], t_item['pos_ids'], t_item['ling'], t_item['read'], t_item['score']

            # Start First Step
            s_essay_fea_1 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_1 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_1 = 0.5 * FCmodel(s_essay_fea_1, t_essay_fea_1, s_essay_embed.to(args.device), t_essay_embed.to(args.device))
            first_loss = cl_loss_1
            first_loss.backward(retain_graph=True)
            optims.step()
            # End First Step

            s_essay_fea_2 = Gmodel(s_pos_ids.to(args.device))
            t_essay_fea_2 = Gmodel(t_pos_ids.to(args.device))
            cl_loss_2 = 0.5 * FCmodel(s_essay_fea_2, t_essay_fea_2, s_essay_embed.to(args.device), t_essay_embed.to(args.device))

            if trait_interactive_type in ['attention', 'none']:
                s_aes_pre_2 = Smodel(s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_2 = s_aes_pre_2.to('cpu')
                aes_loss_2 = mask_mse_loss_fn(s_aes_label, s_aes_pre_2)
                second_loss = aes_loss_2 + cl_loss_2
            else:
                s_aes_pre_2, trait_loss_2 = Smodel(s_essay_fea_2, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_2 = s_aes_pre_2.to('cpu')
                aes_loss_2 = mask_mse_loss_fn(s_aes_label, s_aes_pre_2)
                second_loss = aes_loss_2 + cl_loss_2 + 0.5 * trait_loss_2
            second_loss.backward(retain_graph=True)
            optims.step()

            s_essay_fea_3 = Gmodel(s_pos_ids.to(args.device))
            if trait_interactive_type in ['attention', 'none']:
                s_aes_pre_3 = Smodel(s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_3 = s_aes_pre_3.to('cpu')
                aes_loss_3 = mask_mse_loss_fn(s_aes_label, s_aes_pre_3)
                third_loss = aes_loss_3
            else:
                s_aes_pre_3, trait_loss_3 = Smodel(s_essay_fea_3, s_ling.to(args.device), s_read.to(args.device))
                s_aes_pre_3 = s_aes_pre_3.to('cpu')
                aes_loss_3 = mask_mse_loss_fn(s_aes_label, s_aes_pre_3)
                third_loss = aes_loss_3 + 0.5 * trait_loss_3
            third_loss.backward(retain_graph=True)
            optims.step()

        va_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, va_s_loader, 'valid', trait_interactive_type)
        te_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, te_t_loader, 'test', trait_interactive_type)
        if va_qwk_set['Avg'] > tr_log['Best_dev_qwk_mean']:
            tr_log['Best_dev_qwk_mean'] = va_qwk_set['Avg']
            tr_log['Best_test_qwk_mean'] = te_qwk_set['Avg']
            tr_log['Best_dev_qwk_set'] = va_qwk_set
            tr_log['Best_test_qwk_set'] = te_qwk_set
            tr_log['Best_epoch'] = e_index
            tr_log['BestModel']['BestGmodel'] = Gmodel.state_dict()
            tr_log['BestModel']['BestSmodel'] = Smodel.state_dict()

        epoch_msg = '[CURRENT TARGET] P: {}  [CURRENT EPOCH] E: {}'.format(t_index, e_index)
        logger.info(epoch_msg)
        for trait in va_qwk_set.keys():
            logger.info('[DEV] {} QWK: {:.2f}'.format(trait, va_qwk_set[trait]))
        logger.info('-' * 20)
        for trait in te_qwk_set.keys():
            logger.info('[TEST] {} QWK: {:.2f}'.format(trait, te_qwk_set[trait]))
        logger.info('-' * 20)
        logger.info('Best Epoch: {}'.format(tr_log['Best_epoch']))
        for trait in tr_log['Best_test_qwk_set'].keys():
            logger.info('[BEST DEV IN TEST] {} QWK: {:.2f}'.format(trait, tr_log['Best_test_qwk_set'][trait]))
        logger.info('-' * 50)


def TrainForMultiTraitWOCL(args,
                             Gmodel, Smodel, optims,
                             tr_s_loader, va_s_loader, te_t_loader,
                             t_index, e_index,
                             tr_log, trait_interactive_type):
    logger.info('TrainForMultiTrait: [TARGET] P:{} [EPOCH] E:{}...'.format(t_index, e_index))
    for item_index, (s_item, t_item) in enumerate(zip(tr_s_loader, te_t_loader), start=1):
        Gmodel.train(True)
        Smodel.train(True)
        optims.zero_grad()

        s_prompt, s_pos_ids, s_ling, s_read, s_aes_label = s_item['prompt'], s_item['pos_ids'], s_item['ling'], \
                                                           s_item['read'], s_item['score']
        s_essay_fea = Gmodel(s_pos_ids.to(args.device))
        if 'attention' in trait_interactive_type or 'none' in trait_interactive_type:
            s_aes_pre = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
            loss = aes_loss
        else:
            s_aes_pre, trait_loss = Smodel(s_essay_fea, s_ling.to(args.device), s_read.to(args.device))
            s_aes_pre = s_aes_pre.to('cpu')
            aes_loss = mask_mse_loss_fn(s_aes_label, s_aes_pre)
            loss = aes_loss + 0.5 * trait_loss
        loss.backward(retain_graph=True)
        optims.step()

    va_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, va_s_loader, 'valid', trait_interactive_type)
    te_qwk_set = TestForMultiTrait(args, Gmodel, Smodel, te_t_loader, 'test', trait_interactive_type)
    if va_qwk_set['Avg'] > tr_log['Best_dev_qwk_mean']:
        tr_log['Best_dev_qwk_mean'] = va_qwk_set['Avg']
        tr_log['Best_test_qwk_mean'] = te_qwk_set['Avg']
        tr_log['Best_dev_qwk_set'] = va_qwk_set
        tr_log['Best_test_qwk_set'] = te_qwk_set
        tr_log['Best_epoch'] = e_index
        tr_log['BestModel']['BestGmodel'] = Gmodel.state_dict()
        tr_log['BestModel']['BestSmodel'] = Smodel.state_dict()

    epoch_msg = '[CURRENT TARGET] P: {}  [CURRENT EPOCH] E: {}'.format(t_index, e_index)
    logger.info(epoch_msg)
    for trait in va_qwk_set.keys():
        logger.info('[DEV] {} QWK: {:.2f}'.format(trait, va_qwk_set[trait]))
    logger.info('-' * 20)
    for trait in te_qwk_set.keys():
        logger.info('[TEST] {} QWK: {:.2f}'.format(trait, te_qwk_set[trait]))
    logger.info('-' * 20)
    logger.info('Best Epoch: {}'.format(tr_log['Best_epoch']))
    for trait in tr_log['Best_test_qwk_set'].keys():
        logger.info('[BEST DEV IN TEST] {} QWK: {:.2f}'.format(trait, tr_log['Best_test_qwk_set'][trait]))
    logger.info('-' * 50)