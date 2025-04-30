#!/usr/bin/env bash

source ~/torch/bin/activate
cd ~/notebook/DVRL-AES/

# Promptをコンソールから入力を受け付ける
read -p "Enter the target prompt ID: " prompt

# deviceをcuda:{prompt-1}に設定
device="cuda:$((prompt - 1))"

# 実行コマンド
python src/DataValueEstimation_DVRL_v3.py \
    --wandb \
    --pjname "DVRL-V3" \
    --target_prompt_id "${prompt}" \
    --seed 12 \
    --device "${device}" \
    --pred_model "paes"