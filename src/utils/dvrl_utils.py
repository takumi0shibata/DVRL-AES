"""Utility functions for DVRL model."""

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from utils.general_utils import get_min_max_scores
import matplotlib.pyplot as plt
import polars as pl

def fit_func(
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        epochs: int,
        device: torch.device,
        sample_weight: np.ndarray = None
) -> list:
    """
    Fit the model with the given data.
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to run the model
        sample_weight: Sample weight for each data
    Returns:
        list: Loss history
    """
    
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float)

    x_train = x_train.clone().detach().to(device)
    y_train = y_train.clone().detach().to(device)

    if sample_weight is not None:
        if not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.tensor(sample_weight, dtype=torch.float)
        sample_weight = sample_weight.clone().detach().to(device)
        dataset = TensorDataset(x_train, y_train, sample_weight)
        loss_fn = nn.MSELoss(reduction='none')
    else:
        dataset = TensorDataset(x_train, y_train)
        loss_fn = nn.MSELoss(reduction='mean')

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if sample_weight is not None:
                x_batch, y_batch, w_batch = batch
                y_pred = model(x_batch)
                loss = (loss_fn(y_pred.squeeze(), y_batch.squeeze()) * w_batch).mean()
            else:
                x_batch, y_batch = batch
                y_pred = model(x_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch.squeeze())
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        history.append(epoch_loss / len(train_loader))

    return history

def pred_func(
        model: nn.Module,
        x_test: np.ndarray,
        batch_size: int,
        device: torch.device
    ) -> np.ndarray:
    """
    Predict with the given model.
    Args:
        model: Model to predict
        x_test: Test data
        batch_size: Batch size
        device: Device to run the model
    Returns:
        np.ndarray: Predicted results
    """
    model = model.to(device)
    model.eval()

    x_test = torch.tensor(x_test, dtype=torch.float)
    test_data = TensorDataset(x_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = x_batch[0].to(device)
            y_pred = model(x_batch)
            preds.extend(y_pred.cpu().tolist())
    return np.array(preds)

def calc_qwk(y_true: list, y_pred: list, prompt_id: int, attribute: str, weights='quadratic') -> float:
    """
    Calculate the quadratic weighted kappa.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        prompt_id: Prompt ID
        attribute: Attribute name
    Returns:
        float: Quadratic weighted kappa
    """

    minscore, maxscore = get_min_max_scores()[prompt_id][attribute]

    y_true = np.round((maxscore - minscore) * np.array(y_true) + minscore).flatten()
    y_pred = np.round((maxscore - minscore) * np.array(y_pred) + minscore).flatten()
    
    return cohen_kappa_score(y_true, y_pred, weights=weights, labels=[i for i in range(minscore, maxscore+1)])

def remove_top_p_sample(data_value: np.ndarray, top_p: float, ascending: bool =True):
    """
    Get sample weight for the given data value.
    Args:
        data_value: Data value
        top_p: Top percentage to be selected
        ascending: If True, select the lowest data value
    Returns:
        np.ndarray: Sample weight
    """
    if ascending:
        sorted_data_value = data_value.flatten().argsort()
    else:
        sorted_data_value = data_value.flatten().argsort()[::-1]
    num_elements = int(len(sorted_data_value) * top_p)
    sorted_data_value = sorted_data_value[:num_elements]
    weights = np.ones_like(data_value.flatten())
    for i in sorted_data_value:
        weights[i] = 0
    return weights

def select_non_top_essay_ids_by_percent(data_value_df, percentage=0.1, ascending=False):
    """
    Select a set of essay_ids NOT in the top or bottom percentage based on data_value.

    Parameters:
        data_value_df (pl.DataFrame): DataFrame with 'essay_id' and 'data_value' columns.
        percentage (float): Fraction (0 < percentage <= 1) of essays to exclude.
        ascending (bool): If True, exclude lowest values; if False, exclude highest.

    Returns:
        set: Set of essay_ids NOT in the top/bottom specified percentage.
    """
    assert 0 <= percentage <= 1, "Percentage must be between 0 and 1."

    # Sort by data_value
    sorted_df = data_value_df.sort(by='data_value', descending=not ascending)

    # Determine number of essays to exclude
    top_n = int(len(sorted_df) * percentage)
    excluded_essay_ids = set(sorted_df[:top_n]['essay_id'].to_numpy())

    # All essay_ids in the dataframe
    all_essay_ids = set(data_value_df['essay_id'].to_numpy())

    # Get the remaining (non-top) essay_ids
    non_top_essay_ids = all_essay_ids - excluded_essay_ids

    return non_top_essay_ids

def random_remove_sample(data_value: np.ndarray, remove_p: float):
    """
    Get sample weight for the given data value.
    Args:
        data_value: Data value
        remove_p: Percentage to be removed
    Returns:
        np.ndarray: Sample weight
    """
    weights = np.ones_like(data_value.flatten())
    num_elements = int(len(weights) * remove_p)
    remove_idx = np.random.choice(len(weights), num_elements, replace=False)
    weights[remove_idx] = 0
    return weights

def discover_corrupted_sample(dve_out, noise_idx, noise_rate, output_path, plot=True):
  """Reports True Positive Rate (TPR) of corrupted label discovery.

  Args:
    dve_out: data values
    noise_idx: noise index
    noise_rate: the ratio of noisy samples
    plot: print plot or not

  Returns:
    output_perf: True positive rate (TPR) of corrupted label discovery
                 (per 5 percentiles)
  """

  # Sorts samples by data values
  num_bins = 20  # Per 100/20 percentile
  sort_idx = np.argsort(dve_out)

  # Output initialization
  output_perf = np.zeros([num_bins,])

  # For each percentile
  for itt in range(num_bins):
    # from low to high data values
    output_perf[itt] = len(np.intersect1d(sort_idx[:int((itt+1)* \
                              len(dve_out)/num_bins)], noise_idx)) \
                              / len(noise_idx)

  # Plot corrupted label discovery graphs
  if plot:

    # Defines x-axis
    num_x = int(num_bins/2 + 1)
    x = [a*(1.0/num_bins) for a in range(num_x)]

    # Corrupted label discovery results (dvrl, optimal, random)
    y_dvrl = np.concatenate((np.zeros(1), output_perf[:(num_x-1)]))
    y_opt = [min([a*((1.0/num_bins)/noise_rate), 1]) for a in range(num_x)]
    y_random = x

    plt.figure(figsize=(6, 7.5))
    plt.plot(x, y_dvrl, 'o-')
    plt.plot(x, y_opt, '--')
    plt.plot(x, y_random, ':')
    plt.xlabel('Fraction of data Inspected', size=16)
    plt.ylabel('Fraction of discovered corrupted samples', size=16)
    plt.legend(['DVRL', 'Optimal', 'Random'], prop={'size': 16})
    plt.title('Corrupted Sample Discovery', size=16)
    plt.savefig(output_path + 'corrupted_sample_discovery.png')

  # Returns True Positive Rate of corrupted label discovery
  return output_perf


def find_sample_with_max_distance_sum(selected_sample_indices, all_samples):
    max_distance_sum = -1
    selected_sample_index = None
    for index in range(len(all_samples)):
        if index not in selected_sample_indices:
            # Calculate the sum of distances from the current sample to all selected samples
            distance_sum_to_selected = sum([np.linalg.norm(all_samples[index] - all_samples[selected_index]) for selected_index in selected_sample_indices])
            if distance_sum_to_selected > max_distance_sum:
                max_distance_sum = distance_sum_to_selected
                selected_sample_index = index
    return selected_sample_index


def get_dev_sample(
        features: np.ndarray,
        label: np.ndarray, 
        dev_size: float | int
    ) -> tuple:
    """
    Get the dev set samples.
    Args:
        features: Features
        label: Labels
        dev_size: Dev set size
            percentage or number of samples
    Returns:
        tuple:
    """
    # Initialize the list of selected sample indices with the index of the initial sample
    init_sample_idx = np.random.randint(0, len(features), 1)[0]
    selected_sample_indices = [init_sample_idx]
    all_samples = features

    if 0 < dev_size <= 1:
    # Calculate the number of samples to select
        num_samples_to_select = int(len(label) * dev_size)
    else:
        num_samples_to_select = int(dev_size)

    # Repeat the process until we have the desired number of samples
    while len(selected_sample_indices) < num_samples_to_select:
        sample_with_max_distance_sum_index = find_sample_with_max_distance_sum(selected_sample_indices, all_samples)
        selected_sample_indices.append(sample_with_max_distance_sum_index)

    # Convert the list of indices into a numpy array of samples
    selected_samples_array = all_samples[selected_sample_indices]
    selected_labels_array = label[selected_sample_indices]

    # Identify the indices of unselected samples
    unselected_sample_indices = [i for i in range(len(features)) if i not in selected_sample_indices]
    unselected_samples_array = all_samples[unselected_sample_indices]
    unselected_labels_array = label[unselected_sample_indices]

    print(f"Selected {len(selected_sample_indices)} samples.")
    print('Selected sample indices:', selected_sample_indices)

    return selected_samples_array, unselected_samples_array, selected_labels_array, unselected_labels_array, selected_sample_indices, unselected_sample_indices


def fit_func_for_PAES(
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        epochs: int,
        device: torch.device,
        sample_weight: np.ndarray = None
) -> None:
    """
    Fit the model with the given data.
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to run the model
        sample_weight: Sample weight for each data
    """
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    if sample_weight is not None:
        sample_weight = torch.tensor(sample_weight, dtype=torch.float)
        MSE_Loss = nn.MSELoss(reduction='none').to(device)
        train_dataset = TensorDataset(x_train[0], y_train, x_train[1], x_train[2], x_train[3], sample_weight)
    else:
        # Create loss and optimizer
        MSE_Loss = nn.MSELoss(reduction='mean').to(device)
        # Create Datasets
        train_dataset = TensorDataset(x_train[0], y_train, x_train[1], x_train[2], x_train[3])
    # Create Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    from utils.evaluation import train_model
    if sample_weight is not None:
        for _ in range(epochs):
            train_model(model, train_loader, MSE_Loss, optimizer, device, weight=True)
    else:
        for _ in range(epochs):
            train_model(model, train_loader, MSE_Loss, optimizer, device)


def pred_func_for_PAES(
        model: nn.Module,
        x_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int,
        device: torch.device,
        attribute_name: str,
        metric: str
        ) -> list:
    """
    Predict with the given model.
    Args:
        model: Model to predict
        x_test: Test data
        batch_size: Batch size
        device: Device to run the model
        attribute_name: Attribute name
        metric: Metric to use
    Returns:
        list: Predicted results
    """
    model = model.to(device)
    model.eval()

    test_data = TensorDataset(x_test[0], y_test, x_test[1], x_test[2], x_test[3])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    MSE_Loss = nn.MSELoss(reduction='mean').to(device)

    from utils.evaluation import evaluate_model
    test_results = evaluate_model(model, test_loader, MSE_Loss, device, attribute_name)

    return test_results[metric], test_results['y_pred']