#!/usr/bin/env bash
prompt=$1  # コマンドライン引数から取得
device="cuda"

# pred_model のリスト
pred_models=("mlp" "features_model")

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for seed in 22 32 42 52
  do
    echo "Running with prompt: ${prompt}, pred_model: ${pred_model}, seed: ${seed}"
    python3 src/train_models/train_MLP_v7.py \
          --wandb \
          --pjname "DVRL-V7-20250501" \
          --target_prompt_id "${prompt}" \
          --seed "${seed}" \
          --device "${device}" \
          --pred_model "${pred_model}" \
          --sampling "random"
  done
done
