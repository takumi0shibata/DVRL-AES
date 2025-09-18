#!/usr/bin/env bash
prompt=$1  # コマンドライン引数から取得
device="cuda"

# pred_model のリスト
pred_models=("mlp")
loss_lambda_list=("0.0")

# ループで各組み合わせを実行
for pred_model in "${pred_models[@]}"
do
  for seed in 12
  do
    for loss_lambda in "${loss_lambda_list[@]}"
    do
      echo "Running with prompt: ${prompt}, pred_model: ${pred_model}, seed: ${seed}, loss_lambda: ${loss_lambda}"
      python3 src/train_models/train_MLP_v7.py \
            --wandb \
            --pjname "DVRL-V7-20250501" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --sampling "random" \
            --loss_lambda "${loss_lambda}"
    done
  done
done
