#!/usr/bin/env bash

dev_size_list=("10" "20" "30" "40" "50" "100" "200" "500")

for dev_size in "${dev_size_list[@]}"
do
  for prompt in {1..8}
  do
    echo "Running with dev_size: ${dev_size}, prompt: ${prompt}"
    python3 src/train_models/train_Transformers_devonly.py \
        --target_prompt_id "${prompt}" \
        --dev_size "${dev_size}"
  done
done