# coding=utf-8
"""
The core class of DVRL(Data Valuation using Reinforcement Learning).
"""

import copy
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from sklearn import metrics

import utils.helper as helper
from models.value_estimator import DataValueEstimator
from dvrl.dvrl_pretrain import pretrain
from dvrl.dvrl_loss import DvrlLoss
from utils.evaluation import evaluate_model
from utils.my_utils_for_DVRL import fit_func, pred_func, calc_qwk

class Dvrl(object):
    """
    Data Valuation using Reinforcement Learning (DVRL) class.
    """

    def __init__(self, x_train, y_train, x_val, y_val, pred_model, parameters, checkpoint_file_name, device, test_prompt_id):

        self.x_train = x_train
        self.y_train = y_train.reshape(-1, 1)
        self.x_val = x_val
        self.y_val = y_val.reshape(-1, 1)
        self.checkpoint_file_name = checkpoint_file_name
        self.device = device
        self.test_prompt_id = test_prompt_id

        # Network parameters for data value estimator
        self.hidden_dim = parameters['hidden_dim']
        self.comb_dim = parameters['comb_dim']
        self.outter_iterations = parameters['iterations']
        self.act_fn = parameters['activation']
        self.layer_number = parameters['layer_number']
        self.inner_iterations = parameters['inner_iterations']
        self.batch_size = int(np.min([parameters['batch_size'], self.x_train.shape[0]]))
        self.learning_rate = parameters['learning_rate']
        self.batch_size_predictor = int(np.min([parameters['batch_size_predictor'], self.x_val.shape[0]]))

        # Basic parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.data_dim = self.x_train.shape[1]
        self.label_dim = self.y_train.shape[1]

        self.pred_model = pred_model
        self.final_model = pred_model

        # save initial model
        print('Saving the initial model...')
        os.makedirs('tmp', exist_ok=True)
        torch.save(self.pred_model.state_dict(), 'tmp/init_model.pth')

        # train baseline model
        self.ori_model = copy.deepcopy(self.pred_model)
        self.ori_model.load_state_dict(torch.load('tmp/init_model.pth'))
        print('Training the original model...')
        history = fit_func(self.ori_model, self.x_train, self.y_train, self.batch_size_predictor, self.inner_iterations, self.device)

        self.val_model = copy.deepcopy(self.pred_model)
        self.val_model.load_state_dict(torch.load('tmp/init_model.pth'))
        print('Training the validation model...')
        fit_func(self.val_model, self.x_val, self.y_val, self.batch_size_predictor, self.inner_iterations, self.device)


    def train_dvrl(self, metric='mse'):
        """
        Train value estimator
        :return:
        :rtype:
        """
        # selection network
        self.value_estimator = DataValueEstimator(self.data_dim+self.label_dim, self.hidden_dim, self.comb_dim, self.layer_number, self.act_fn)
        self.value_estimator = self.value_estimator.to(self.device)
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold).to(self.device)
        dvrl_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)
        # scheduler = lr_scheduler.ExponentialLR(dvrl_optimizer, gamma=0.999)

        # baseline performance
        y_valid_hat = pred_func(self.ori_model, self.x_val, self.batch_size_predictor, self.device)
        if metric == 'mse':
            valid_perf = metrics.mean_squared_error(self.y_val, y_valid_hat)
            print(f'Origin model Performance MSE: {valid_perf: .3f}')
        elif metric == 'qwk':
            valid_perf = calc_qwk(self.y_val, y_valid_hat, self.test_prompt_id, 'score')
            print(f'Origin model Performance QWK: {valid_perf: .3f}')
        else:
            raise ValueError('Metric not supported')

        # Prediction differences
        y_train_valid_pred = pred_func(self.val_model, self.x_train, self.batch_size_predictor, self.device)
        y_pred_diff = np.abs(self.y_train - y_train_valid_pred)

        rewards_history = []
        losses_history = []
        for iter in tqdm(range(self.outter_iterations)):
            self.value_estimator.train()
            dvrl_optimizer.zero_grad()

            # Batch selection
            batch_idx = np.random.permutation(self.x_train.shape[0])[:self.batch_size]

            x_batch = torch.tensor(self.x_train[batch_idx], dtype=torch.float).to(self.device)
            y_batch = torch.tensor(self.y_train[batch_idx], dtype=torch.float).to(self.device)
            y_hat_batch = torch.tensor(y_pred_diff[batch_idx], dtype=torch.float).to(self.device)

            # Generates the selection probability
            est_dv_curr = self.value_estimator(x_batch, y_batch, y_hat_batch).squeeze()

            # Samples the selection probability
            sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy(), est_dv_curr.shape)
            # Exception (When selection probability is 0)
            if np.sum(sel_prob_curr) == 0:
                print('All zero selection probability')
                est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

            new_model = self.pred_model
            new_model.load_state_dict(torch.load('tmp/init_model.pth'))
            history = fit_func(new_model, x_batch, y_batch, self.batch_size_predictor, self.inner_iterations, self.device, sel_prob_curr)
            y_valid_hat = pred_func(new_model, self.x_val, self.batch_size_predictor, self.device)

            # reward computation
            if metric == 'mse':
                dvrl_perf = metrics.mean_squared_error(self.y_val, y_valid_hat)
                reward = valid_perf - dvrl_perf
            elif metric == 'qwk':
                dvrl_perf = calc_qwk(self.y_val, y_valid_hat, self.test_prompt_id, 'score')
                reward = dvrl_perf - valid_perf

            # update the selection network
            reward = torch.tensor([reward]).to(self.device)
            sel_prob_curr = torch.tensor(sel_prob_curr, dtype=torch.float).to(self.device)
            loss = dvrl_criterion(est_dv_curr, sel_prob_curr, reward)
            loss.backward()
            dvrl_optimizer.step()

            print(f'Iteration: {iter+1}, Reward: {reward.item()}, DVRL Loss: {loss.item()}, Prob MAX: {torch.max(est_dv_curr).item():.3f}, Prob MIN: {torch.min(est_dv_curr).item():.3f}, QWK: {dvrl_perf:.3f}')
            rewards_history.append(reward.item())
            losses_history.append(loss.item())


        # Training the final model
        x_train = torch.tensor(self.x_train, dtype=torch.float).to(self.device)
        y_train = torch.tensor(self.y_train, dtype=torch.float).to(self.device)
        y_pred_diff = torch.tensor(y_pred_diff, dtype=torch.float).to(self.device)
        final_data_value = self.value_estimator(x_train, y_train, y_pred_diff).squeeze()
        self.final_model.load_state_dict(torch.load('tmp/init_model.pth'))
        fit_func(self.final_model, self.x_train, self.y_train, self.batch_size_predictor, self.inner_iterations, self.device, final_data_value)

        return rewards_history, losses_history

    def dvrl_valuator(self, x_train, y_train):
        """
        Estimate the given data value.
        :param feature: Data intermediate feature.
        :type feature: torch.Tensor
        :param label: Corresponding labels
        :type label:torch.Tensor
        :return:
        :rtype:
        """
        # first calculate the prection difference
        output = self.val_model(x_train)
        y_pred_diff = torch.abs(y_train - output)

        # predict the value
        data_value = self.value_estimator(x_train, y_train, y_pred_diff)
        
        return data_value.cpu().detach().numpy()
    

    def dvrl_predict(self, x_test):

        test_results = pred_func(self.final_model, x_test, self.batch_size_predictor, self.device)

        return test_results