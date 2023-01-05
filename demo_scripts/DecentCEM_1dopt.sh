#!/bin/bash
# This script will run DecentCEM on the 1d optimization task
#     with a popsize 200, and
#     the best hyperparameters summarized in Table A.2

python utils/cem_opt_1d.py \
            --alg "CEM-E" \
            --max_iters 100 \
            --popsize 200 \
            --ensemble_size 10 \
            --randomize_init_mean 2 \
            --elites_ratio 0.1 \
            --epsilon 1e-3 \
            --alpha 0.1 \
            --seed 1 \
            --episodes 10
