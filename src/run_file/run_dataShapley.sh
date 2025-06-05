#!/usr/bin/env bash

# Promptをコンソールから入力を受け付ける
# read -p "Enter the target prompt ID: " prompt

prompt=$1  # コマンドライン引数から取得
device="cuda"

pred_models=("mlp" "features_model")

lambdas=(0.0)

dev_size_list=(30)

for seed in 12
do
  for pred_model in "${pred_models[@]}"
  do
    for lambda in "${lambdas[@]}"
    do
      for dev_size in "${dev_size_list[@]}"
      do
        echo "Running with pred_model: ${pred_model}, seed: ${seed}, lambda: ${lambda}, dev_size: ${dev_size}"

        python3 src/DataValueEstimation_DataShapley.py \
              --wandb \
              --pjname "DS-V7" \
              --target_prompt_id "${prompt}" \
              --seed "${seed}" \
              --device "${device}" \
              --pred_model "${pred_model}" \
              --sampling "random" \
              --loss_lambda "${lambda}" \
              --dev_size "${dev_size}"
      done
    done
  done
done