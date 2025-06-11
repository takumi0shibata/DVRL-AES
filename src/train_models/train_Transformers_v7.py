'''
This script is used to train the model using the Transformer-based model.
    - BERT
    - DeBERTa-v3-large
'''
import numpy as np
import argparse
import torch
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import mean_squared_error
import polars as pl

# my packages
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dvrl.dataset import EssayDataset
from utils.dvrl_utils import select_non_top_essay_ids_by_percent, calc_qwk
from utils.general_utils import set_seed
from dvrl.sampling import select_diverse_subset


def train_and_evaluate(
    source_data,
    target_data,
    pseudo_dict,
    top_essay_ids,
    dev_mask,
    args,
    device,
):
    """
    Fine-tunes a Transformer (BERT/DeBERTa) for regression using Hugging Face Trainer
    with fp16 support. Returns (best_test_qwk, best_dev_loss, best_dev_qwk).
    """
    # Prepare splits
    train_mask = np.array([essay_id in top_essay_ids for essay_id in source_data['essay_id']])
    print(f'{np.sum(train_mask)} / {len(source_data["essay_id"])}')

    train_texts = source_data['essay'][train_mask].tolist()
    train_labels = source_data['scaled_score'][train_mask].tolist()

    dev_texts = target_data['essay'][dev_mask].tolist()
    dev_labels = target_data['scaled_score'][dev_mask].tolist()

    test_texts = target_data['essay'][~dev_mask].tolist()
    test_labels = target_data['scaled_score'][~dev_mask].tolist()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Tokenization helper
    def tokenize(batch):
        enc = tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length
        )
        enc['labels'] = batch['label']
        return enc

    # Build datasets as dicts
    train_dataset = {
        'text': train_texts,
        'label': train_labels
    }
    eval_dataset = {
        'text': dev_texts,
        'label': dev_labels
    }
    test_dataset = {
        'text': test_texts,
        'label': test_labels
    }

    # Tokenize datasets
    from datasets import Dataset
    train_ds = Dataset.from_dict(train_dataset).map(tokenize, batched=True)
    eval_ds = Dataset.from_dict(eval_dataset).map(tokenize, batched=True)
    test_ds = Dataset.from_dict(test_dataset).map(tokenize, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type='regression'
    )

    if args.model_name == 'microsoft/deberta-v3-large':
        # LoRA のハイパラ
        print('Applying LoRA...')
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,   # sequence classification/regression なら SEQ_CLS
            r=2,                          # LoRA ランク（低ランク行列の次元）
            lora_alpha=8,                 # スケーリング係数
            lora_dropout=0.1              # ドロップアウト率
        )
        model = get_peft_model(model, lora_config).to(device)
    else:
        model = model.to(device)

    # Metrics
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze(-1)
        mse = mean_squared_error(labels, preds)
        qwk = calc_qwk(
            np.array(labels),
            np.array(preds),
            args.target_prompt_id,
            args.attribute_name
        )
        return {'mse': mse, 'qwk': qwk}

    # TrainingArguments with fp16
    training_args = TrainingArguments(
        output_dir=f"./logs/{args.run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        learning_rate=args.lr,
        weight_decay=0.001,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=0,
        seed=args.seed,
        disable_tqdm=False,
        report_to='none',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Evaluate on dev (best) and get metrics
    dev_metrics = trainer.evaluate(eval_ds)
    best_dev_loss = dev_metrics['eval_mse']
    best_dev_qwk = dev_metrics['eval_qwk']

    # Predict on test
    test_pred = trainer.predict(test_ds)
    test_preds = test_pred.predictions.squeeze(-1)
    test_qwk = calc_qwk(
        np.array(test_labels),
        np.array(test_preds),
        args.target_prompt_id,
        args.attribute_name
    )

    return test_qwk, best_dev_loss, best_dev_qwk



def main(args):
    ###################################################
    # Step0. Set UP
    ###################################################
    target_prompt_id = args.target_prompt_id
    device = torch.device(args.device)
    set_seed(args.seed)

    if args.wandb:
        wandb.init(
            project=args.pjname,
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load embedding
    with open('./outputs/embedding/deberta-v3-large.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)

    # load pseudo label
    pseudo_df = pl.read_csv('./outputs/pseudo_labels/pseudo_label_by_features_model.csv')
    pseudo_dict = dict(zip(pseudo_df['essay_id'].to_numpy(), pseudo_df['y_pred'].to_numpy()))
    
    # Load essay data
    print('Loading essay data...')
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    source_data, target_data = dataset.cross_prompt_split(
        target_prompt_set=args.target_prompt_id,
        add_pos=False,
    )
    print(f'    Number of source samples: {len(source_data["essay_id"])}')
    print(f'    Number of target samples: {len(target_data["essay_id"])}')

    # Load Estimated Data Value
    data_value_df = pl.read_csv(f'outputs/{args.pjname}/values_{target_prompt_id}_{args.pred_model}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}.csv')

    # Select dev data
    selected_dev_ids = select_diverse_subset(
        {essay_id: embedding_dict[essay_id] for essay_id in target_data['essay_id']},
        args.dev_size,
        method=args.sampling,
        seed=args.seed
    )
    dev_mask = np.isin(target_data['essay_id'], selected_dev_ids)
    
    for p_val in np.arange(0.0, 1.0, 0.1):
        ##################################################
        # データの価値が低いものを削除
        ##################################################
        set_seed(args.seed)
        top_source_essay_ids = select_non_top_essay_ids_by_percent(
            data_value_df,
            percentage=p_val,
            ascending=True
        )
        qwk_high, dev_loss_high, dev_qwk_high = train_and_evaluate(
            source_data,
            target_data,
            pseudo_dict,
            top_source_essay_ids,
            dev_mask,
            args,
            device,
        )

        ##################################################
        # データの価値が高いものを削除
        ##################################################
        set_seed(args.seed)
        top_target_essay_ids = select_non_top_essay_ids_by_percent(
            data_value_df,
            percentage=p_val,
            ascending=False
        )
        qwk_low, dev_loss_low, dev_qwk_low = train_and_evaluate(
            source_data,
            target_data,
            pseudo_dict,
            top_target_essay_ids,
            dev_mask,
            args,
            device,
        )

        print(f'p: {p_val:.1f}, QWK[High]: {qwk_high:.3f}, QWK[Low]: {qwk_low:.3f}, Dev Loss[High]: {dev_loss_high:.10f}, Dev Loss[Low]: {dev_loss_low:.10f}')

        if args.wandb:
            wandb.log({
                'p': p_val,
                'QWK[High]': qwk_high,
                'QWK[Low]': qwk_low,
                'Dev Loss[High]': dev_loss_high,
                'Dev Loss[Low]': dev_loss_low,
                'Dev QWK[High]': dev_qwk_high,
                'Dev QWK[Low]': dev_qwk_low,
            })

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser('Training Model')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V7')
    parser.add_argument('--run_name', type=str, default='train-Transformer')
    parser.add_argument('--target_prompt_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--attribute_name', type=str, default='score')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large')
    parser.add_argument('--dev_size', type=int, default=30)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32, choices=[32, 8]) # BERT-base: 32, DeBERTa-v3-large: 16
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--pred_model', type=str, default='mlp', choices=['mlp', 'features_model', 'hybrid'])
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'greedy', 'maxmin', 'kmeans++'])
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-uncased',
        choices=[
            'bert-base-uncased',
            'microsoft/deberta-v3-large'
        ]
    )

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)