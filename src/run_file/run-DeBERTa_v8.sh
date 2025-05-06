#!/usr/bin/env bash

device="cuda"

pred_model="mlp"
seeds=("12" "22" "32")

model_name="microsoft/deberta-v3-large"
loss_lambda="0.0"
weights=("original" "inverse")

# ループで各組み合わせを実行
for seed in "${seeds[@]}"
do
  for prompt in {1..8}
  do
    for weight in "${weights[@]}"
    do
      echo "Running with seed: ${seed}, weight: ${weight}, prompt: ${prompt}"
      python3 src/train_models/train_Transformers_v8.py \
            --wandb \
            --pjname "DVRL-V7-20250501" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --loss_lambda "${loss_lambda}" \
            --sampling "random" \
            --weights "${weight}" \
            --model_name "${model_name}" \
            --max_length 512 \
            --batch_size 8 \
            --epochs 3
    done
  done
done