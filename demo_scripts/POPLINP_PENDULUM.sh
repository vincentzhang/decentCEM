#!/bin/bash
# The following script will run POPLIN-P on pendulum

python mbexp.py \
    -env gym_pendulum \
    -logdir log/gym_pendulum/POPLIN-P \
    -o exp_cfg.exp_cfg.ntrain_iters 50 \
    -o exp_cfg.exp_cfg.ninit_rollouts 0 \
    -o exp_cfg.sim_cfg.task_hor 200 \
    -o exp_cfg.sim_cfg.seed_train 1 \
    -o ctrl_cfg.cem_cfg.cem_type POPLINP-SEP \
    -o ctrl_cfg.cem_cfg.training_scheme AVG-R \
    -o ctrl_cfg.cem_cfg.policy_network_shape [32] \
    -o ctrl_cfg.opt_cfg.init_var 0.1 \
    -o ctrl_cfg.cem_cfg.test_policy 1 \
    -o ctrl_cfg.cem_cfg.test_policy_epochs 5 \
    -o ctrl_cfg.cem_cfg.eval_ctrl_type MPC \
    -o ctrl_cfg.cem_cfg.debug_optimizer False \
    -ca model-type PE \
    -ca prop-type E \
    -ca opt-type POPLIN-P
