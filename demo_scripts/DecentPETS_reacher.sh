#!/bin/bash
# The following script will run DecentPETS on gym_reacher 

python mbexp.py \
    -env gym_reacher \
    -logdir log/gym_reacher/DecentPETS \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o exp_cfg.exp_cfg.ninit_rollouts 0 \
    -o exp_cfg.sim_cfg.task_hor 200 \
    -o exp_cfg.sim_cfg.seed_train 1 \
    -o ctrl_cfg.cem_cfg.cem_type DecentCEM \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -o ctrl_cfg.cem_cfg.ensemble_size 5 \
    -o ctrl_cfg.cem_cfg.test_policy_epochs 5 \
    -o ctrl_cfg.cem_cfg.eval_ctrl_type MPC \
    -o ctrl_cfg.cem_cfg.use_prev_sol True \
    -o ctrl_cfg.cem_cfg.debug_optimizer False \
    -o ctrl_cfg.cem_cfg.eval_cem_policy False \
    -ca model-type PE \
    -ca prop-type E \
    -ca opt-type DecentCEM
