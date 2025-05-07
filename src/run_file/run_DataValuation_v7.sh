#!/usr/bin/env bash

# Promptをコンソールから入力を受け付ける
# read -p "Enter the target prompt ID: " prompt

prompt=$1  # コマンドライン引数から取得
device="cuda"

pred_models=("features_model")

lambdas=(0.5 1.0)

for seed in 12 32 52
do
  for pred_model in "${pred_models[@]}"
  do
    for lambda in "${lambdas[@]}"
    do
      echo "Running with pred_model: ${pred_model}, seed: ${seed}, lambda: ${lambda}"

      python3 src/DataValueEstimation_DVRL_v7.py \
            --wandb \
            --pjname "DVRL-V7-20250501" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --sampling "random" \
            --loss_lambda "${lambda}"
    done
  done
done