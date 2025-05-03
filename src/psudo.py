import torch
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import polars as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import cohen_kappa_score

from utils.general_utils import set_seed
from dvrl.dataset import EssayDataset
from models.paes import PAES
from utils.general_utils import set_seed, get_min_max_scores

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        # Move batch to device
        pos_x, linguistic_x, readability_x, labels, essay_set = [b.to(device) for b in batch]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(pos_x, linguistic_x, readability_x)
        
        # Calculate loss
        loss = loss_fn(outputs.squeeze(), labels.squeeze())
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_epoch(model: nn.Module, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_essay_set = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            pos_x, linguistic_x, readability_x, labels, essay_set = [b.to(device) for b in batch]
            
            # Forward pass
            outputs = model(pos_x, linguistic_x, readability_x)
            
            # Calculate loss
            loss = loss_fn(outputs.squeeze(), labels.squeeze())
            total_loss += loss.item()
            
            # Store predictions and labels for QWK calculation
            # スカラー値の場合の処理を追加
            squeezed_outputs = outputs.squeeze()
            squeezed_labels = labels.squeeze()
            squeezed_essay_set = essay_set.squeeze()
            
            # スカラー値かテンソルかを判定して適切に処理
            if squeezed_outputs.dim() == 0:  # スカラー値の場合
                all_preds.append(squeezed_outputs.item())
            else:
                all_preds.extend(squeezed_outputs.cpu().numpy())
                
            if squeezed_labels.dim() == 0:  # スカラー値の場合
                all_labels.append(squeezed_labels.item())
            else:
                all_labels.extend(squeezed_labels.cpu().numpy())
            
            if squeezed_essay_set.dim() == 0:
                all_essay_set.append(squeezed_essay_set.item())
            else:
                all_essay_set.extend(squeezed_essay_set.cpu().numpy())
    
    # Calculate QWK
    qwks = []
    lwks = []
    for i in range(1, 9):
        minscore, maxscore = get_min_max_scores()[i]['score']
        indices = np.where(np.array(all_essay_set) == i)[0]
        if len(indices) > 0:
            rescaled_targets = np.round(minscore + (maxscore - minscore) * np.array(all_labels)[indices])
            rescaled_predictions = np.round(minscore + (maxscore - minscore) * np.array(all_preds)[indices])
            qwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='quadratic')
            lwk = cohen_kappa_score(rescaled_targets, rescaled_predictions, labels=[i for i in range(minscore, maxscore+1)], weights='linear')
            qwks.append(qwk)
            lwks.append(lwk)
    
    print(np.array(qwks))
    return {
        'loss': total_loss / len(data_loader),
        'lwk': lwks,
        'qwk': qwk
    }

prompts = []
qwks = []
for prompt in range(1, 9):
    target_prompt_id = prompt
    device = torch.device('cuda')
    set_seed(12)
    
    ###################################################
    # Step1. Load Data
    ###################################################
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    source_data, target_data = dataset.cross_prompt_split(
        target_prompt_set=target_prompt_id,
        add_pos=True,
    )
    print(f'    Number of target samples: {len(target_data["essay_id"])}')
    print(f'    Number of source samples: {len(source_data["essay_id"])}')
    
    from sklearn.model_selection import train_test_split
    
    train_index = np.array(range(len(source_data['essay_id'])))
    
    train_index, val_index = train_test_split(
        train_index,
        test_size=0.2,
        random_state=12,
        shuffle=True
    )

    # Initialize the model
    source = {
        'pos_x': source_data['pos_x'][train_index],
        'linguistic_x': source_data['feature'][train_index],
        'readability_x': source_data['readability'][train_index],
        'normalized_label': source_data['scaled_score'][train_index],
        'essay_set': source_data['essay_set'][train_index],
    }

    dev = {
        'pos_x': source_data['pos_x'][val_index],
        'linguistic_x': source_data['feature'][val_index],
        'readability_x': source_data['readability'][val_index],
        'normalized_label': source_data['scaled_score'][val_index],
        'essay_set': source_data['essay_set'][val_index],
    }

    target = {
        'pos_x': target_data['pos_x'],
        'linguistic_x': target_data['feature'],
        'readability_x': target_data['readability'],
        'normalized_label': target_data['scaled_score'],
        'essay_set': target_data['essay_set'],
    }

    source_dataset = TensorDataset(
        torch.tensor(source['pos_x'], dtype=torch.int32),
        torch.tensor(source['linguistic_x'], dtype=torch.float32),
        torch.tensor(source['readability_x'], dtype=torch.float32),
        torch.tensor(source['normalized_label'], dtype=torch.float32),
        torch.tensor(source['essay_set'], dtype=torch.int32),
    ) 
    dev_dataset = TensorDataset(
        torch.tensor(dev['pos_x'], dtype=torch.int32),
        torch.tensor(dev['linguistic_x'], dtype=torch.float32),
        torch.tensor(dev['readability_x'], dtype=torch.float32),
        torch.tensor(dev['normalized_label'], dtype=torch.float32),
        torch.tensor(dev['essay_set'], dtype=torch.int32),
    )
    target_dataset = TensorDataset(
        torch.tensor(target['pos_x'], dtype=torch.int32),
        torch.tensor(target['linguistic_x'], dtype=torch.float32),
        torch.tensor(target['readability_x'], dtype=torch.float32),
        torch.tensor(target['normalized_label'], dtype=torch.float32),
        torch.tensor(target['essay_set'], dtype=torch.int32),
    )

    train_loader = DataLoader(dataset=source_dataset, batch_size=10, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=target_dataset, batch_size=10, shuffle=True)

    # Initialize the model
    model = PAES(source_data['max_sentnum'], source_data['max_sentlen'], source_data['pos_vocab']).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss(reduction='mean').to(device)

    # Training loop
    best_dev_qwk = -1
    best_test_qwk = -1
    best_loss = 1000
    for epoch in range(50):
        print(f"Epoch {epoch+1}/50")
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        # Dev
        dev_history = evaluate_epoch(model, dev_loader, loss_fn, device)
        dev_loss = dev_history['loss']
        dev_qwk = dev_history['qwk']
        # Test
        test_history = evaluate_epoch(model, test_loader, loss_fn, device)
    
        if dev_qwk > best_dev_qwk:
            best_loss = dev_loss
            best_dev_qwk = dev_qwk
            best_test_qwk = test_history['qwk']
    
        print(f"train_loss: {train_loss:.4f}, dev_loss: {dev_loss:.4f}, dev_qwk: {dev_qwk:.4f}, test_qwk: {test_history['qwk']:.4f}, best_test_qwk: {best_test_qwk:.4f}")
    
    prompts.append(prompt)
    qwks.append(best_test_qwk)
    
pl.DataFrame({'prompt': prompts, 'qwk': qwks}).write_csv('psudo_PAES.csv')
