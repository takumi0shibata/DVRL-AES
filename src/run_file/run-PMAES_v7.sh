#!/usr/bin/env bash

source ~/torch/bin/activate
cd ~/notebook/DVRL-AES/

# Promptをコンソールから入力を受け付ける
read -p "Enter the target prompt ID: " prompt

# deviceをcuda:{prompt}に設定
device="cuda:$((prompt))"
batch_num=35

# pred_model のリスト
pred_models=("features_model")

# lambda のリスト
# lambda_list=("0.0" "0.5" "1.0")
lambda_list=("0.0")

# seed のリスト
seeds=("12")


# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for lambda in "${lambda_list[@]}"
  do
    for seed in "${seeds[@]}"
    do
      echo "Running with pred_model: ${pred_model}, lambda: ${lambda}, seed: ${seed}"
      python src/train_models/train_PMAES_v7.py \
            --wandb \
            --pjname "DS-V7" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --loss_lambda "${lambda}" \
            --sampling "random" \
            --batch_num "${batch_num}"
    done
  done
done