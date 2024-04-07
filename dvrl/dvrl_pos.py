
"""DVRL class for data valuation using reinforcement learning"""

import copy
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import wandb

from dvrl.dvrl_loss import DvrlLoss
from utils.dvrl_utils import fit_func_for_PAES, pred_func_for_PAES, calc_qwk


class DataValueEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        comb_dim: int,
        layer_number: int,
        act_fn: callable
    ) -> None:
        """
        Args:
          input_dim: The dimensionality of the input features (x_input and y_input combined).
          hidden_dim: The dimensionality of the hidden layers.
          comb_dim: The dimensionality of the combined layer.
          layer_number: Total number of layers in the MLP before combining with y_hat.
          act_fn: Activation function to use.
        """

        super(DataValueEstimator, self).__init__()
        
        self.act_fn = act_fn
        # Initial layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        # Intermediate layers
        self.intermediate_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_number - 3)]
        )
        # Layer before combining with y_hat
        self.pre_comb_layer = nn.Linear(hidden_dim, comb_dim)
        # Layer after combining with y_hat
        self.comb_layer = nn.Linear(comb_dim + 1, comb_dim)
        # Output layer
        self.output_layer = nn.Linear(comb_dim, 1)
        
    def forward(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        y_hat_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x_input: Input features.
          y_input: Target labels.
          y_hat_input: Predicted labels or some representation thereof.
          
        Returns:
          Tensor: The estimated data values.
        """
        inputs = torch.cat((x_input, y_input), dim=1)
        
        # Initial layer
        x = self.act_fn(self.initial_layer(inputs))
        
        # Intermediate layers
        for layer in self.intermediate_layers:
            x = self.act_fn(layer(x))
        
        # Pre-combination layer
        x = self.act_fn(self.pre_comb_layer(x))
        
        # Combining with y_hat_input
        x = torch.cat((x, y_hat_input), dim=1)  # Ensure y_hat_input is properly shaped
        x = self.act_fn(self.comb_layer(x))
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.output_layer(x))
        
        return x



class Dvrl(object):

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        pred_model: nn.Module,
        parameters: dict,
        device: str,
        test_prompt_id: int,
        x_source_embed: np.ndarray,
    ) -> None:
        """
        Args:
            x_train: Training data
            y_train: Training labels
            x_dev: Validation data
            y_dev: Validation labels
            pred_model: Prediction model
            parameters: Parameters for DVRL
            device: Device to run the model
            test_prompt_id: Prompt id for the test
            x_source_embed: embedding vector for data value estimator
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.device = device
        self.test_prompt_id = test_prompt_id
        self.x_source_embed = x_source_embed

        # Network parameters for data value estimator
        self.hidden_dim = parameters['hidden_dim']
        self.comb_dim = parameters['comb_dim']
        self.outter_iterations = parameters['iterations']
        self.act_fn = parameters['activation']
        self.layer_number = parameters['layer_number']
        self.inner_iterations = parameters['inner_iterations']
        self.batch_size = int(np.min([parameters['batch_size'], self.x_train[0].shape[0]]))
        self.learning_rate = parameters['learning_rate']
        self.batch_size_predictor = int(np.max([parameters['batch_size_predictor'], self.x_dev[0].shape[0]]))
        self.moving_average_window = parameters['moving_average_window']
        self.moving_average = parameters['moving_average']

        # Basic parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.std_penalty_weight = parameters['std_penalty_weight']
        self.data_dim = self.x_source_embed.shape[1]
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
        fit_func_for_PAES(self.ori_model, self.x_train, self.y_train, self.batch_size_predictor, self.inner_iterations, self.device)

        self.val_model = copy.deepcopy(self.pred_model)
        self.val_model.load_state_dict(torch.load('tmp/init_model.pth'))
        print('Training the validation model...')
        fit_func_for_PAES(self.val_model, self.x_dev, self.y_dev, self.batch_size_predictor, self.inner_iterations, self.device)


    def train_dvrl(
        self,
        metric: str
    ) -> None:
        """
        Train the DVRL model
        Args:
            metric: Metric to use for the DVRL
                mse or qwk or corr
        """
        # selection network
        self.value_estimator = DataValueEstimator(self.data_dim+self.label_dim, self.hidden_dim, self.comb_dim, self.layer_number, self.act_fn)
        self.value_estimator = self.value_estimator.to(self.device)
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold, self.std_penalty_weight).to(self.device)
        dvrl_optimizer = optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)

        # baseline performance
        valid_perf, _ = pred_func_for_PAES(self.ori_model, self.x_dev, self.y_dev, self.batch_size_predictor, self.device, 'score', metric)
        print(f'Baseline {metric}: {valid_perf:.3f}')

        # Prediction differences
        _, y_train_valid_pred = pred_func_for_PAES(self.val_model, self.x_train, self.y_train, self.batch_size_predictor, self.device, 'score', metric)
        y_pred_diff = np.abs(self.y_train.numpy() - np.array(y_train_valid_pred).reshape(-1, 1))

        if self.moving_average:
            baseline = 0
        else:
            baseline = valid_perf
        
        for iter in tqdm(range(self.outter_iterations)):
            self.value_estimator.train()
            dvrl_optimizer.zero_grad()

            # Batch selection
            batch_idx = np.random.permutation(self.x_train[0].shape[0])[:self.batch_size]

            x_embed_batch = torch.tensor(self.x_source_embed[batch_idx], dtype=torch.float).to(self.device)
            x_source_batch = self.x_train[0][batch_idx].to(self.device)
            x_source_linguistic_batch = self.x_train[1][batch_idx].to(self.device)
            x_source_retrieval_batch = self.x_train[2][batch_idx].to(self.device)
            x_source_prompt_batch = self.x_train[3][batch_idx].to(self.device)
            x_batch = [x_source_batch, x_source_linguistic_batch, x_source_retrieval_batch, x_source_prompt_batch]
            y_batch = self.y_train[batch_idx].to(self.device)
            y_hat_batch = torch.tensor(y_pred_diff[batch_idx], dtype=torch.float).to(self.device)

            # Generates the selection probability
            est_dv_curr = self.value_estimator(x_embed_batch, y_batch, y_hat_batch).squeeze()

            # Samples the selection probability
            sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy(), est_dv_curr.shape)
            # Exception (When selection probability is 0)
            if np.sum(sel_prob_curr) == 0:
                print('All zero selection probability')
                est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

            new_model = self.pred_model
            new_model.load_state_dict(torch.load('tmp/init_model.pth'))
            fit_func_for_PAES(new_model, x_batch, y_batch, self.batch_size_predictor, self.inner_iterations, self.device, sel_prob_curr)
            dvrl_perf, _ = pred_func_for_PAES(new_model, self.x_dev, self.y_dev, self.batch_size_predictor, self.device, 'score', metric)

            # reward computation
            if metric == 'mse':
                reward = baseline - dvrl_perf
            elif metric == 'qwk':
                reward = dvrl_perf - baseline
            elif metric == 'corr':
                reward = dvrl_perf - baseline

            # update the selection network
            reward = torch.tensor([reward]).to(self.device)
            sel_prob_curr = torch.tensor(sel_prob_curr, dtype=torch.float).to(self.device)
            loss = dvrl_criterion(est_dv_curr, sel_prob_curr, reward)
            loss.backward()
            dvrl_optimizer.step()

            # update the baseline
            if self.moving_average:
                baseline = ((self.moving_average_window - 1) / self.moving_average_window) * baseline + (dvrl_perf / self.moving_average_window)

            if metric == 'mse':
                print(f'Iteration: {iter+1}, Reward: {reward.item():.3f}, DVRL Loss: {loss.item():.3f}, Prob MAX: {torch.max(est_dv_curr).item():.3f}, Prob MIN: {torch.min(est_dv_curr).item():.3f}, MSE: {dvrl_perf:.3f}')
            elif metric == 'qwk':
                print(f'Iteration: {iter+1}, Reward: {reward.item():.3f}, DVRL Loss: {loss.item():.3f}, Prob MAX: {torch.max(est_dv_curr).item():.3f}, Prob MIN: {torch.min(est_dv_curr).item():.3f}, QWK: {dvrl_perf:.3f}')
            elif metric == 'corr':
                print(f'Iteration: {iter+1}, Reward: {reward.item():.3f}, DVRL Loss: {loss.item():.3f}, Prob MAX: {torch.max(est_dv_curr).item():.3f}, Prob MIN: {torch.min(est_dv_curr).item():.3f}, Corr: {dvrl_perf:.3f}')

            wandb.log({
                'Reward': reward.item(),
                'DVRL Loss': loss.item(),
                'Prob MAX': torch.max(est_dv_curr).item(),
                'Prob MIN': torch.min(est_dv_curr).item(),
                metric: dvrl_perf
                })


        # Training the final model
        x_train = torch.tensor(self.x_source_embed, dtype=torch.float).to(self.device)
        y_pred_diff = torch.tensor(y_pred_diff, dtype=torch.float).to(self.device)
        final_data_value = self.value_estimator(x_train, self.y_train.to(self.device), y_pred_diff).squeeze()
        self.final_model.load_state_dict(torch.load('tmp/init_model.pth'))
        fit_func_for_PAES(self.final_model, self.x_train, self.y_train, self.batch_size_predictor, self.inner_iterations, self.device, final_data_value)

        return final_data_value.cpu().detach().numpy()
    

    def dvrl_predict(self, x_test: np.ndarray, y_test: np.array) -> np.ndarray:
        """
        Predict the given data using the DVRL model
        Args:
            x_test: Test data
        Returns:
            test_results: Predicted results
        """

        test_results, _ = pred_func_for_PAES(self.final_model, x_test, y_test, self.batch_size_predictor, self.device, 'score', 'qwk')

        return test_results