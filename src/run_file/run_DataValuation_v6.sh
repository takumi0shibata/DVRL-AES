#!/usr/bin/env bash

source ~/torch/bin/activate
cd ~/notebook/DVRL-AES/

# Promptをコンソールから入力を受け付ける
read -p "Enter the target prompt ID: " prompt

# deviceをcuda:{prompt}に設定
device="cuda:$((prompt))"

# pred_model のリスト
pred_models=("mlp" "features_model")

# loss_lambda のリスト
lambda_list=("0.5" "1.0")

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for lambda in "${lambda_list[@]}"
  do
    echo "Running with pred_model: ${pred_model}, lambda: ${lambda}"

    if [ "${lambda}" = "1.0" ]; then
      python src/DataValueEstimation_DVRL_v6.py \
          --wandb \
          --pjname "DVRL-V5-20250427" \
          --target_prompt_id "${prompt}" \
          --seed 12 \
          --device "${device}" \
          --pred_model "${pred_model}" \
          --loss_lambda "${lambda}" \
          --dev_size 0
    else
      python src/DataValueEstimation_DVRL_v6.py \
          --wandb \
          --pjname "DVRL-V5-20250427" \
          --target_prompt_id "${prompt}" \
          --seed 12 \
          --device "${device}" \
          --pred_model "${pred_model}" \
          --loss_lambda "${lambda}"
    fi
  done
done