#!/usr/bin/env bash

device="cuda"

pred_model="mlp"
# seeds=("12" "22" "32")
seeds=("12" "42" "52")

# pred_model="features_model"
# seeds=("12" "32" "52")

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
      python3 src/train_models/train_MLP_v8.py \
            --wandb \
            --pjname "DVRL-V7-20250501" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --loss_lambda "${loss_lambda}" \
            --sampling "random" \
            --weights "${weight}"
    done
  done
done
