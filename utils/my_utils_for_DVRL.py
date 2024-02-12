import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from sklearn.metrics import cohen_kappa_score
from utils.general_utils import get_min_max_scores



def fit_func(model, x_train, y_train, batch_size, epochs, device, sample_weight=None):
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    if sample_weight is not None:
        loss_fn = nn.MSELoss(reduction='none')
        sample_weight = torch.tensor(sample_weight, dtype=torch.float)
        train_data = TensorDataset(x_train, y_train, sample_weight)
    else:
        loss_fn = nn.MSELoss(reduction='mean')
        train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)

    history = []
    for epoch in range(epochs):
        losses = []
        if sample_weight is not None:
            for x_batch, y_batch, w_batch in train_loader:
                optimizer.zero_grad()
                x_batch, y_batch, w_batch = x_batch.to(device), y_batch.to(device), w_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch.squeeze()) * w_batch
                loss = loss.mean()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        else:
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred.squeeze(), y_batch.squeeze())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        history.append(np.mean(losses))
        # if (epoch) % 10 == 0:
        #     print(f'Epoch: {epoch}, Loss:  {loss.item()}')
    
    return history

def pred_func(model, x_test, batch_size, device):
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
    return preds

def calc_qwk(y_true, y_pred, prompt_id, attribute):
    minscore, maxscore = get_min_max_scores()[prompt_id][attribute]

    y_true = (maxscore - minscore) * np.array(y_true) + minscore
    y_pred = np.round((maxscore - minscore) * np.array(y_pred) + minscore).flatten()
    
    return cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[i for i in range(minscore, maxscore+1)])

def get_sample_weight(data_value, top_p, ascending=True):
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