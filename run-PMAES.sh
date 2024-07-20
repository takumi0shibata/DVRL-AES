#!/usr/bin/env bash
pjname=("DVRL" "LOO" "Data Shapley")
max_batch_size=320
num_epochs=50
run_name="train-PMAES"
valuation_method=("DVRL-pos" "LOO-pos" "DataShapley-pos")
seed=12
device="cuda"

for ((i=0; i<${#pjname[@]}; i++))
do
    for prompt in {1..8}
    do
        python train_PMAES.py \
            --wandb \
            --pjname ${pjname[i]} \
            --run_name ${run_name} \
            --valuation_method ${valuation_method[i]} \
            --target_id ${prompt} \
            --seed ${seed} \
            --max_batch_size ${max_batch_size} \
            --num_epochs ${epochs} \
            --device ${device}
    done
done