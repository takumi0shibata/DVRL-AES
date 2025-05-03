#!/usr/bin/env bash

# Promptをコンソールから入力を受け付ける
# read -p "Enter the target prompt ID: " prompt

prompt=$1  # コマンドライン引数から取得
device="cuda"

pred_models=("mlp" "features_model")

for seed in 22 32 42 52
do
  for pred_model in "${pred_models[@]}"
  do
    echo "Running with pred_model: ${pred_model}, seed: ${seed}"

    python3 src/DataValueEstimation_DVRL_v7.py \
          --wandb \
          --pjname "DVRL-V7-20250501" \
          --target_prompt_id "${prompt}" \
          --seed "${seed}" \
          --device "${device}" \
          --pred_model "${pred_model}" \
          --sampling "random"
  done
done