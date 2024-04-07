#!/usr/bin/env bash
python BERT-DVRL.py --test_prompt_id 3 --max_length 512 --batch_size 16 --epochs 10 --model_name bert-base-uncased --device cuda:2
for seed in 12 22 32 42 52
do  
    python BERT-DevOnly.py --test_prompt_id 3 --seed ${seed} --max_length 512 --batch_size 16 --epochs 15 --model_name bert-base-uncased --device cuda:2
    python BERT-FullSource.py --test_prompt_id 3 --seed ${seed} --max_length 512 --batch_size 16 --epochs 10 --model_name bert-base-uncased --device cuda:2
done