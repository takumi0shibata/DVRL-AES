#!/usr/bin/env bash

device="cuda"

# pred_model のリスト
pred_model="features_model"
loss_lambda="0.0"
seeds=("32" "52")
weights=("uniform" "original" "inverse")

# ループで各組み合わせを実行
for seed in "${seeds[@]}"
do
  for prompt in {1..8}
  do
    for weight in "${weights[@]}"
    do
      echo "Running with seed: ${seed}, weight: ${weight}, prompt: ${prompt}"
      python3 src/train_models/train_PAES_v8_test.py \
            --wandb \
            --pjname "DVRL-V7-20250501" \
            --target_prompt_id "${prompt}" \
            --seed "${seed}" \
            --device "${device}" \
            --pred_model "${pred_model}" \
            --loss_lambda "${loss_lambda}" \
            --sampling "random" \
            --weight "${weight}"
    done
  done
done