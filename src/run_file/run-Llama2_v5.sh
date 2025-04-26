#!/usr/bin/env bash

# Define the project name
pjname="DVRL-V5-20250206"
loss_lambda=1.0
seed=12
pred_model="mlp"

for prompt in {1..8}
do
    for top_p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        python src/train_models/train_Llama2_v5.py \
            --target_prompt_id ${prompt} \
            --wandb \
            --top_p ${top_p} \
            --pjname ${pjname} \
            --loss_lambda ${loss_lambda} \
            --seed ${seed} \
            --pred_model ${pred_model}
    done
done