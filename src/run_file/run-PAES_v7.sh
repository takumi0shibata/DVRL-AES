#!/usr/bin/env bash

# source ~/torch/bin/activate
# cd ~/notebook/DVRL-AES/

# # Promptをコンソールから入力を受け付ける
# read -p "Enter the target prompt ID: " prompt

# # deviceをcuda:{prompt}に設定
# device="cuda:$((prompt))"

prompt=$1  # コマンドライン引数から取得
device="cuda"

# pred_model のリスト
pred_models=("features_model")

# lambda のリスト
# lambda_list=("0.0" "0.5" "1.0")
lambda_list=("0.0")

seeds=("12")

dev_size_list=(30)

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for lambda in "${lambda_list[@]}"
  do
    for seed in "${seeds[@]}"
    do
      for dev_size in "${dev_size_list[@]}"
      do
        echo "Running with pred_model: ${pred_model}, lambda: ${lambda}, seed: ${seed}, dev_size: ${dev_size}"
        python3 src/train_models/train_PAES_v7.py \
              --wandb \
              --pjname "LOO-V7" \
              --target_prompt_id "${prompt}" \
              --seed "${seed}" \
              --device "${device}" \
              --pred_model "${pred_model}" \
              --loss_lambda "${lambda}" \
              --sampling "random" \
              --dev_size "${dev_size}"
      done
    done
  done
done