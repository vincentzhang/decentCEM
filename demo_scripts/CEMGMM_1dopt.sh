#!/bin/bash
# This script will run CEM-GMM on the 1d optimization task
#     with a popsize 200, and
#     the best hyperparameters summarized in Table A.2

python utils/cem_opt_1d.py \
            --alg "CEM-GMM" \
            --max_iters 100 \
            --popsize 200 \
            --M 8 \
            --randomize_init_mean 2 \
            --elites_ratio 0.1 \
            --epsilon 1e-3 \
            --alpha 0.1 \
            --seed 1 \
            --episodes 10 \
            --kappa 0.5 \
            --return_mode m
