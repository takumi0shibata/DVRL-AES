#!/usr/bin/env bash

source ~/torch/bin/activate
cd ~/notebook/DVRL-AES/

dev_size_list=("10" "20" "30" "40" "50" "100")
prompt_list=("1" "2" "3" "4" "5" "6" "7" "8")

for dev_size in "${dev_size_list[@]}"
do
  for prompt in "${prompt_list[@]}"
  do
    echo "Running with dev_size: ${dev_size}, prompt: ${prompt}"
    python src/train_models/train_Transformers_devonly.py \
        --target_prompt_id "${prompt}" \
        --dev_size "${dev_size}"
  done
done