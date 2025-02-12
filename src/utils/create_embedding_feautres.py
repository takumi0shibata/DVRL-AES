"""Utility functions for creating embedding features by pre-trained language model."""

import pandas as pd
import os
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
from utils.general_utils import get_min_max_scores


def normalize_scores(y, essay_set, attribute_name):
    """
    Normalize scores based on the min and max scores for each unique prompt_id in essay_set.
    Args:
        y: Scores to normalize.
        essay_set: Array of essay_set (prompt_id) for each score.
        attribute_name: The attribute name to filter the min and max scores.
    Returns:
        np.ndarray: Normalized scores.
    """
    min_max_scores = get_min_max_scores()
    normalized_scores = np.zeros_like(y, dtype=float)
    for unique_prompt_id in np.unique(essay_set):
        minscore, maxscore = min_max_scores[unique_prompt_id][attribute_name]
        mask = (essay_set == unique_prompt_id)
        normalized_scores[mask] = (y[mask] - minscore) / (maxscore - minscore)
    return normalized_scores


def create_embedding_features(
        data_path: str,
        attribute_name: str,
        embedding_model_name: str,
        device: torch.device
    ) -> list[dict]:
    """
    Create embedding features for the given data.
    Args:
        data_path: Path to the data.
        attribute_name: Attribute name.
        embedding_model_name: Pre-trained language model name.
        device: Device to run the model.
    Returns:
        tuple: Train, dev, and test features and labels.
    """

    # Load data
    print(f'load data from {data_path}...')
    data = load_data(data_path)

    y_train = np.array(data['train']['label'])
    y_dev = np.array(data['dev']['label'])
    y_test = np.array(data['test']['label'])

    train_essay_prompt = np.array(data['train']['essay_set'])
    dev_essay_prompt = np.array(data['dev']['essay_set'])
    test_essay_prompt = np.array(data['test']['essay_set'])

    train_essay_id = np.array(data['train']['essay_id'])
    dev_essay_id = np.array(data['dev']['essay_id'])
    test_essay_id = np.array(data['test']['essay_id'])

    # Normalize scores
    y_train = normalize_scores(y_train, train_essay_prompt, attribute_name)
    y_dev = normalize_scores(y_dev, dev_essay_prompt, attribute_name)
    y_test = normalize_scores(y_test, test_essay_prompt, attribute_name)

    data['train']['normalized_label'] = y_train
    data['dev']['normalized_label'] = y_dev
    data['test']['normalized_label'] = y_test

    # # Create embedding
    # os.makedirs(data_path + 'cache/', exist_ok=True)
    # pkl_files = [file for file in os.listdir(data_path + 'cache/') if file.endswith('.pkl')]
    # model_name = embedding_model_name
    # if len(pkl_files) == 0:
    #     # Load embedding model
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModel.from_pretrained(model_name).to(device)

    #     train_loader = create_data_loader(data['train'], tokenizer, max_length=512, batch_size=32)
    #     dev_loader = create_data_loader(data['dev'], tokenizer, max_length=512, batch_size=32)
    #     test_loader = create_data_loader(data['test'], tokenizer, max_length=512, batch_size=32)

    #     # Create embedding
    #     print('[Train]')
    #     train_features = run_embedding_model(train_loader, model, device)
    #     print('[Dev]')
    #     dev_features = run_embedding_model(dev_loader, model, device)
    #     print('[Test]')
    #     test_features = run_embedding_model(test_loader, model, device)
        
    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     # Save embedding
    #     with open(data_path + 'cache/train_features.pkl', 'wb') as f:
    #         pickle.dump(train_features, f)
    #     with open(data_path + 'cache/dev_features.pkl', 'wb') as f:
    #         pickle.dump(dev_features, f)
    #     with open(data_path + 'cache/test_features.pkl', 'wb') as f:
    #         pickle.dump(test_features, f)
    # else:
    #     print('Loading embedding from cache...')
    #     train_features = pickle.load(open(data_path + 'cache/train_features.pkl', 'rb'))
    #     dev_features = pickle.load(open(data_path + 'cache/dev_features.pkl', 'rb'))
    #     test_features = pickle.load(open(data_path + 'cache/test_features.pkl', 'rb'))

    train_features = feature_embedding(data['train']['feature'])
    dev_features = feature_embedding(data['dev']['feature'])
    test_features = feature_embedding(data['test']['feature'])

    train_data = {'essay': train_features, 'normalized_label': y_train, 'essay_set': train_essay_prompt, 'essay_id': train_essay_id}
    dev_data = {'essay': dev_features, 'normalized_label': y_dev, 'essay_set': dev_essay_prompt, 'essay_id': dev_essay_id}
    test_data = {'essay': test_features, 'normalized_label': y_test, 'essay_set': test_essay_prompt, 'essay_id': test_essay_id}

    return train_data, dev_data, test_data


def load_data(data_path: str, attribute: str = 'score') -> dict:
    """
    Load data from the given path.
    Args:
        data_path: Path to the data.
    Returns:
        dict: Data.
    """
    data = {}
    for file in ['train', 'dev', 'test']:
        feature = []
        label = []
        essay_id = []
        essay_set = []
        try:
            read_data = pd.read_pickle(data_path + file + '.pkl')
        except:
            read_data = pd.read_pickle(data_path + file + '.pk')
        for i in range(len(read_data)):
            feature.append(read_data[i]['content_text'])
            label.append(int(read_data[i][attribute]))
            essay_id.append(int(read_data[i]['essay_id']))
            essay_set.append(int(read_data[i]['prompt_id']))
        data[file] = {'feature': feature, 'label': label, 'essay_id': essay_id, 'essay_set': essay_set}

    return data

def run_embedding_model(data_loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    """
    Run the embedding model.
    Args:
        data_loader: Data loader.
        model: Embedding model.
        device: Device to run the model.
    Returns:
        np.ndarray: Features.
    """

    model.eval()
    progress_bar = tqdm(data_loader, desc="Create Embedding", unit="batch", ncols=100)
    with torch.no_grad():
        features = []
        for d in progress_bar:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            features.extend(outputs.last_hidden_state[:, 0, :].cpu().tolist())
    return np.array(features)

def feature_embedding(essays: str) -> np.ndarray:
    import re
    # Define feature extraction functions
    def word_count(essay):
        return len(essay.split())

    def sentence_count(essay):
        return len(re.findall(r'\w+[.!?]', essay))

    def avg_word_length(essay):
        words = essay.split()
        return sum(len(word) for word in words) / len(words) if words else 0

    def unique_word_count(essay):
        return len(set(essay.lower().split()))
    
    # Extract features
    features = np.array([[word_count(essay), sentence_count(essay), avg_word_length(essay), unique_word_count(essay)] for essay in essays])

    # Normalize features
    features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)

    return features_normalized


class EssayDataset(Dataset):
    def __init__(self, data: list, tokenizer: AutoTokenizer, max_length: int, weights: np.ndarray = None) -> None:
        """
        Args:
            data: Data.
            tokenizer: Tokenizer.
            max_length: Maximum length of the input.
        """
        self.texts = np.array(data['feature'])
        self.scores = np.array(data['normalized_label'])
        self.prompts = np.array(data['essay_set'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones_like(self.scores)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'score': torch.tensor(self.scores[item], dtype=torch.float),
            'prompt': torch.tensor(self.prompts[item], dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'weights': torch.tensor(self.weights[item], dtype=torch.float)
        }
    

# データローダーの定義
def create_data_loader(data: list, tokenizer: AutoTokenizer, max_length: int, batch_size: int, weights: np.ndarray = None) -> DataLoader:
    """
    Create data loader.
    Args:
        data: Data.
        tokenizer: Tokenizer.
        max_length: Maximum length of the input.
        batch_size: Batch size.
    Returns:
        DataLoader: Data loader.
    """
    ds = EssayDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        weights = weights
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)