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
from utils.dvrl_utils import calc_qwk
from utils.general_utils import set_seed
from dvrl.sampling import select_diverse_subset

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels   = inputs.pop('labels')
        weights  = inputs.pop('weight', None)
        outputs  = model(**inputs)
        preds    = outputs.logits.squeeze(-1)

        loss_vec = (preds - labels) ** 2
        if weights is not None:
            loss_vec = loss_vec * weights
        loss = loss_vec.mean()

        return (loss, outputs) if return_outputs else loss


def train_and_evaluate(
    source_data,
    target_data,
    embedding_dict,
    weight_dict,
    dev_mask,
    args,
    device,
):
    """
    Fine-tunes a Transformer (BERT/DeBERTa) for regression using Hugging Face Trainer
    with fp16 support. Returns (best_test_qwk, best_dev_loss, best_dev_qwk).
    """
    # For source weights
    source_weights = np.array([weight_dict[eid] for eid in source_data['essay_id']])
    if args.weights == 'original':
        pass
    elif args.weights == 'inverse':
        source_weights = 1 - source_weights # reverse the weight
    elif args.weights == 'uniform':
        source_weights = np.ones_like(source_weights) # use uniform weights
    else:
        raise ValueError(f"Invalid weight option: {args.weights}")
    
    train_texts = source_data['essay'].tolist()
    train_labels = source_data['scaled_score'].tolist()

    dev_texts = target_data['essay'][dev_mask].tolist()
    dev_labels = target_data['scaled_score'][dev_mask].tolist()

    test_texts = target_data['essay'][~dev_mask].tolist()
    test_labels = target_data['scaled_score'][~dev_mask].tolist()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        enc = tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length
        )
        enc['labels'] = batch['label']
        if 'weight' in batch:                 # ←★ ここから
            # Dataset は list か np.ndarray を受け取るのでそのまま渡す
            enc['weight'] = batch['weight']   # ←★ ここまで
        return enc

    # Build datasets as dicts
    train_dataset = {
        'text':   train_texts,
        'label':  train_labels,
        'weight': source_weights.tolist()
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
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = WeightedTrainer(
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
            name=args.run_name + f'_{args.pred_model}_{target_prompt_id}_seed{args.seed}_dev{args.dev_size}_lambda{args.loss_lambda}_{args.sampling}_{args.weights}',
            config=dict(args._get_kwargs())
        )

    ###################################################
    # Step1. Load Data
    ###################################################
    # Load embedding
    with open('./outputs/embedding/deberta-v3-large.pkl', 'rb') as f:
        embedding_dict = pickle.load(f)

    # # load pseudo label
    # pseudo_df = pl.read_csv('./outputs/pseudo_labels/pseudo_label_by_features_model.csv')
    # pseudo_dict = dict(zip(pseudo_df['essay_id'].to_numpy(), pseudo_df['y_pred'].to_numpy()))
    
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
    vals = data_value_df['data_value'].to_numpy().astype(float)
    vmin, vmax = vals.min(), vals.max()
    data_value_df = data_value_df.with_columns(
        ((pl.col('data_value') - vmin) / (vmax - vmin)).alias('weight')
    )
    weight_dict = dict(
        zip(data_value_df['essay_id'].to_list(),
            data_value_df['weight'].to_list())
    )
    # 基本統計量
    weights = np.array(data_value_df['weight'].to_list())
    print(f"Count: {len(weights)}")
    print(f"Min   : {weights.min():.4f}")
    print(f"Max   : {weights.max():.4f}")
    print(f"Mean  : {weights.mean():.4f}")
    print(f"Median: {np.median(weights):.4f}\n")

    # ヒストグラムの度数とビン幅を計算
    counts, bin_edges = np.histogram(weights, bins=20)

    # 最も多いビンを基準にバーの長さを調整
    max_count = counts.max()
    scale = 40 / max_count  # 最大度数を40文字のバーに

    print("Histogram (bin_left-bin_right : count [bar])")
    for cnt, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
        bar = '█' * int(cnt * scale)
        print(f"{left:.2f}-{right:.2f} : {cnt:4d} {bar}")

    # Create dev_mask
    selected_dev_ids = select_diverse_subset(
        {essay_id: embedding_dict[essay_id] for essay_id in target_data['essay_id'] if essay_id in embedding_dict},
        args.dev_size,
        method=args.sampling,
        seed=args.seed
    )
    dev_mask = np.isin(target_data['essay_id'], selected_dev_ids)
    
    # Training
    best_test_qwk, best_dev_loss, best_dev_qwk = train_and_evaluate(
        source_data=source_data,
        target_data=target_data,
        embedding_dict=embedding_dict,
        weight_dict=weight_dict,
        dev_mask=dev_mask,
        args=args,
        device=device
    )
    print(f'Best Test QWK: {best_test_qwk:.4f}, Best Dev Loss: {best_dev_loss:.4f}, Best Dev QWK: {best_dev_qwk:.4f}')
    if args.wandb:
        wandb.log({
            'best_test_qwk': best_test_qwk,
            'best_dev_loss': best_dev_loss,
            'best_dev_qwk': best_dev_qwk
        })

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser('Training Model')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--pjname', type=str, default='DVRL-V8')
    parser.add_argument('--run_name', type=str, default='train-weighted-Transformer')
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
    parser.add_argument('--pred_model', type=str, default='mlp', choices=['mlp', 'features_model'])
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'greedy', 'maxmin', 'kmeans++'])
    parser.add_argument('--weights', type=str, default='uniform', choices=['uniform', 'original', 'inverse'])
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