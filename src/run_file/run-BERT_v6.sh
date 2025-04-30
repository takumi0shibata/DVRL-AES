#!/usr/bin/env bash

device="cuda"

# pred_model のリスト
pred_models=("mlp" "features_model")

# lambda のリスト
lambda_list=("0.5" "1.0")

# ループで各組み合わせを実行
for prompt in {1..8}
do
  for pred_model in "${pred_models[@]}"
  do
    for lambda in "${lambda_list[@]}"
    do
      echo "Running with prompt: ${prompt}, pred_model: ${pred_model}, lambda: ${lambda}"
      
      if [ "${lambda}" = "1.0" ]; then
          python src/train_models/train_Transformers_v6.py \
              --wandb \
              --pjname "DVRL-V5-20250427" \
              --target_prompt_id "${prompt}" \
              --seed 12 \
              --device "${device}" \
              --pred_model "${pred_model}" \
              --loss_lambda "${lambda}" \
              --dev_size 0 \
              --max_length 512 \
              --batch_size 16 \
              --epochs 10 \
              --model_name "bert-base-uncased"
      else
          python src/train_models/train_Transformers_v6.py \
              --wandb \
              --pjname "DVRL-V5-20250427" \
              --target_prompt_id "${prompt}" \
              --seed 12 \
              --device "${device}" \
              --pred_model "${pred_model}" \
              --loss_lambda "${lambda}" \
              --max_length 512 \
              --batch_size 16 \
              --epochs 10 \
              --model_name "bert-base-uncased"
      fi
    done
  done
done
