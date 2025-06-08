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

weights=("original" "inverse")

# seed のリスト
seeds=("22" "42")


# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for seed in "${seeds[@]}"
  do
    for weight in "${weights[@]}"
    do
      echo "Running with pred_model: ${pred_model}, lambda: ${lambda}, seed: ${seed}"
      python src/train_models/train_PMAES_v8.py \
            --wandb \
            --pjname "DVRL-V7-20250501" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --loss_lambda "0.0" \
            --sampling "random" \
            --batch_num "${batch_num}" \
            --weights "${weight}"
    done
  done
done