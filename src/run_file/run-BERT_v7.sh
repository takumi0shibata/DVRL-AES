#!/usr/bin/env bash

device="cuda"
seeds=("12")

pred_model="hybrid"  # モデル名を指定

# ループで各組み合わせを実行
for prompt in {1..8}
do
  for seed in "${seeds[@]}"
  do
    echo "Running with seed: ${seed}, prompt: ${prompt}"
    python3 src/train_models/train_Transformers_v7.py \
        --wandb \
        --pjname "DVRL-V7-20250501" \
        --target_prompt_id "${prompt}" \
        --seed "${seed}" \
        --device "${device}" \
        --pred_model "${pred_model}" \
        --loss_lambda "0.0" \
        --max_length 512 \
        --batch_size 32 \
        --epochs 5 \
        --model_name "bert-base-uncased"
  done
done