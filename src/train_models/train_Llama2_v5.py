# Description: This script is used to fine-tune the Llama-2-7b model on the ASAP dataset.

# Importing required libraries
import argparse
import warnings
import torch
from datasets import Dataset
import numpy as np
import polars as pl
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    LlamaForSequenceClassification,
    AutoModelForSequenceClassification,
    EvalPrediction,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# my packages
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dvrl.dataset import EssayDataset
from utils.dvrl_utils import remove_top_p_sample, calc_qwk
from utils.general_utils import set_seed


# Custom function to prepare compute metrics
def prepare_compute_metrics(target_prompt_id, attribute_name):
    # 評価データのプロンプトがデータセット内で一意であることに注意（特にCross-promptの状況のとき）
    def compute_metrics(p: EvalPrediction):
        preds = np.squeeze(p.predictions)
        qwk = calc_qwk(p.label_ids, preds, target_prompt_id, attribute_name)
        lwk = calc_qwk(p.label_ids, preds, target_prompt_id, attribute_name, "linear")
        correlation = np.corrcoef(p.label_ids, preds)[0, 1]
        rmse = np.sqrt(mean_squared_error(p.label_ids, preds))
        mae = mean_absolute_error(p.label_ids, preds)

        return {
            "QWK": qwk,
            "LWK": lwk,
            "Correlation": correlation,
            "RMSE": rmse,
            "MAE": mae,
        }
    return compute_metrics


# Main function
warnings.filterwarnings("ignore")
def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    set_seed(args.seed)

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}',
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
        add_pos=False,
        device='cuda'
    )
    print(f'    Number of training samples: {len(train_data["essay_id"])}')
    print(f'    Number of dev samples: {len(dev_data["essay_id"])}')
    print(f'    Number of test samples: {len(test_data["essay_id"])}')
    print(f'    Selected Dev data: {dev_data["essay_id"]}')
    print(f'    The score of selected Dev data: {dev_data["original_score"]}')

    # load pseudo label
    df = pl.read_csv('data/pseudo_label.csv').to_dict()
    pseudo_dict = dict(zip(df['essay_id'].to_numpy(), df['y_pred'].to_numpy()))
    estimated_data_value = np.load(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_ot{args.ot}.npy')

    # Remove top p samples
    weights = remove_top_p_sample(
        estimated_data_value,
        args.top_p,
        args.ascending
    )
    weights = (torch.tensor(weights, dtype=torch.float) == 1)
    
    # Create dataset
    train_data = {
        'essay': train_data['essay'][weights].tolist(),
        'labels': train_data['scaled_score'][weights].tolist()
    }
    dev_data = {
        'essay': dev_data['essay'].tolist(),
        'labels': dev_data['scaled_score'].tolist()
    }
    pseudo = {
        'essay': test_data['essay'].tolist(),
        'labels': np.array([pseudo_dict[eid] for eid in test_data['essay_id']]).tolist(),
    }
    test_data = {
        'essay': test_data['essay'].tolist(),
        'labels': test_data['scaled_score'].tolist()
    }

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)
    pseudo_dataset = Dataset.from_dict(pseudo)
    test_dataset = Dataset.from_dict(test_data)

    ############################################################
    # Load model
    ############################################################
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_eos_token=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    # Lora config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            # "k_proj",
            # "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        task_type = "SEQ_CLS",
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        load_in_4bit=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Tokenize data
    def tokenize_func(example):
        return tokenizer(example['essay'], truncation=True, max_length=args.max_length, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_func, batched=True)
    dev_dataset = dev_dataset.map(tokenize_func, batched=True)
    pseudo_dataset = pseudo_dataset.map(tokenize_func, batched=True)
    test_dataset = test_dataset.map(tokenize_func, batched=True)

    ############################################################
    # Train model
    ############################################################
    if args.wandb:
        report_to = 'wandb'
    else:
        report_to = 'none'

    # Define training arguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type='constant',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        seed=args.seed,
        report_to=report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_QWK",
        label_names=["labels"],
        warmup_ratio=0.1,
        weight_decay=0.001,
        bf16=True,  # bf16混合精度を有効化 (Ampere以降のGPU推奨)
        optim="adamw_8bit", # 8ビットOptimizerを使用
        disable_tqdm=True
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset={
            'dev': dev_dataset,
            'test': test_dataset,
        },
        compute_metrics=prepare_compute_metrics(target_prompt_id, args.attribute_name),
    )

    # Train model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Model')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V5')
    parser.add_argument('--run_name', type=str, default='train-Llama2')
    parser.add_argument('--output_dir', type=str, default='.log/')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--pred_model', type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=1.0)
    parser.add_argument('--ot', action='store_true')
    parser.add_argument('--ascending', action='store_true')
    # Hyperparameters
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--top_p', type=float, default=0.05)
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)


