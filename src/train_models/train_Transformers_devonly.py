'''
This script is used to train the model using the Transformer-based model.
    - BERT
    - DeBERTa-v3-large
'''
import argparse
import numpy as np
import torch
import json

import polars as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
import polars as pl
import torch
from torch.utils.data import Dataset as TorchDataset # Rename to avoid conflict

# my packages
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dvrl.dataset import EssayDataset
from utils.general_utils import set_seed, get_min_max_scores



def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device('cuda')
    set_seed(12)

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    dataset.preprocess_dataframe()
    train_data, dev_data, test_data = dataset.cross_prompt_split(
        target_prompt_set=target_prompt_id,
        dev_size=args.dev_size,
        cache_dir='src/.embedding_cache',
        embedding_model='microsoft/deberta-v3-large',
        add_pos=False,
        device=device
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')
    print(f'    Selected Dev data: {dev_data["essay_id"]}')
    print(f'    The score of selected Dev data: {dev_data["original_score"]}')

    # --- Configuration ---
    # Specify the pre-trained model name. Can be changed to "bert-base-uncased", "FacebookAI/roberta-base", "microsoft/deberta-v3-large", etc.
    model_name = "bert-base-uncased"
    num_train_epochs = 10 # Number of training epochs (adjust as needed)
    batch_size = 32 # Batch size per device (adjust based on GPU memory)
    max_length = 512 # Max sequence length for tokenizer

    # --- 3. Load Tokenizer and Model ---
    # Load the tokenizer associated with the chosen pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the pre-trained model for sequence classification.
    # Set num_labels=1 for regression tasks.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # --- 4. Tokenize Data ---
    # Tokenize the texts using the loaded tokenizer
    dev_encodings = tokenizer(dev_data['essay'].tolist(), truncation=True, padding="max_length", max_length=max_length)
    test_encodings = tokenizer(test_data['essay'].tolist(), truncation=True, padding="max_length", max_length=max_length)

    # --- 5. Create Custom PyTorch Dataset ---
    class EssayDatasetTmp(TorchDataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            # Retrieve tokenized inputs for the given index
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # Add the corresponding label, converting it to a tensor
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float) # Ensure label is float tensor
            return item

        def __len__(self):
            # Return the total number of samples
            return len(self.labels)

    # Instantiate the custom dataset for training and evaluation sets
    dev_dataset = EssayDatasetTmp(dev_encodings, dev_data['scaled_score'])
    test_dataset = EssayDatasetTmp(test_encodings, test_data['scaled_score'])

    # --- 6. Define Training Arguments ---
    # Configure the training process using TrainingArguments (remains the same)
    training_args = TrainingArguments(
        output_dir='./wandb',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1, # Number of steps for learning rate warmup
        weight_decay=2e-5, # Strength of weight decay regularization
        optim="adamw_torch", # Use the AdamW optimizer
        logging_strategy="steps",  # Log metrics at the end of each epoch
        logging_steps=10, # Log every 10 steps
        eval_strategy="epoch",     # Evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=False, # Load the best model found during training at the end
        metric_for_best_model="eval_qwk", # Use Mean Squared Error to determine the best model
        greater_is_better=True, # Lower MSE is better
        report_to="none", # Disable external reporting integrations like WandB/TensorBoard for simplicity
        fp16=torch.cuda.is_available(), # Use mixed precision training if a GPU is available
    )

    # --- 7. Define Compute Metrics Function ---
    # Define a function to compute metrics during evaluation (MSE, MAE, and QWK for regression)
    def prepare_compute_metrics(minscore, maxscore):
        def compute_metrics(eval_pred: EvalPrediction):
            predictions, labels = eval_pred
            # Predictions might be logits or regression outputs, squeeze them if necessary
            if len(predictions.shape) > 1:
                predictions = predictions.squeeze(-1)

            # Calculate standard regression metrics
            rmse = np.sqrt(mean_squared_error(labels, predictions))
            mae = mean_absolute_error(labels, predictions)

            # Convert predictions and labels to scores based on min/max values
            # This step is necessary for calculating QWK
            predictions = predictions * (maxscore - minscore) + minscore
            labels = labels * (maxscore - minscore) + minscore
            qwk = cohen_kappa_score(np.round(predictions), np.round(labels), weights="quadratic", labels=[i for i in range(minscore, maxscore + 1)])
            lwk = cohen_kappa_score(np.round(predictions), np.round(labels), weights="linear", labels=[i for i in range(minscore, maxscore + 1)])

            # Calculate Correlation Coefficient
            corr = np.corrcoef(predictions, labels)[0, 1]

            return {"rmse": rmse, "mae": mae, "qwk": qwk, "lwk": lwk, "corr": corr}
        return compute_metrics

    # --- 8. Instantiate Trainer ---
    # Initialize the Trainer with the model, arguments, custom datasets, tokenizer, and metrics function
    # Note: The tokenizer is still passed for potential use cases like saving, but not strictly needed for data loading now.
    minscore, maxscore = get_min_max_scores()[target_prompt_id]['score']
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dev_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=prepare_compute_metrics(minscore, maxscore),
    )

    # --- 9. Train the Model ---
    print(f"Starting fine-tuning for {model_name}...")
    trainer.train()
    print("Fine-tuning completed.")

    # --- (Optional) Evaluate the Best Model ---
    print("Evaluating the best model on the validation set...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation results:", eval_results)

    # Save Metrics with json
    print("Saving metrics...")
    with open(f"outputs/devonly_{target_prompt_id}_{args.dev_size}.json", "w") as metrics_file:
        json.dump(eval_results, metrics_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--dev_size', type=int, default=30)
    args = parser.parse_args()
    main(args)