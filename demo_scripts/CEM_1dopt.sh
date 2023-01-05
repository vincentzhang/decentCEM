#!/bin/bash
# This script will run CEM on the 1d optimization task
# with popsize 200

python utils/cem_opt_1d.py \
            --alg "CEM" \
            --max_iters 100 \
            --popsize 200 \
            --ensemble_size 1 \
            --randomize_init_mean 2 \
            --elites_ratio 0.1 \
            --epsilon 1e-3 \
            --alpha 0.1 \
            --seed 1 \
            --episodes 10
