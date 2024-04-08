#!/usr/bin/env bash
for seed in 12 22 32 42 52
do
	for target_id in 1 2 3 4 5 6 7 8 
	do
		python PMAES-FullSource.py --seed $seed --target_id $target_id --device cuda
	done
done