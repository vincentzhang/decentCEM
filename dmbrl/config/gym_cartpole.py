from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
"""
    Module name,
    MODEL_IN, MODEL_OUT,
    import env, env_name
"""


class GymCartpoleConfigModule:
    ENV_NAME = "MBRLGYM_cartpole-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30
    INIT_VAR = 0.25
    MODEL_IN, MODEL_OUT = 5, 4  # obs -> 3, action -> 1
    GP_NINDUCING_POINTS = 300
    SEED_EVAL = 0

    def __init__(self):
        # self.ENV = gym.make(self.ENV_NAME)
        from mbbl.env.gym_env import cartpole
        self.ENV = cartpole.env(env_name='gym_cartpole', rand_seed=1234,
                                misc_info={'reset_type': 'gym'})
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "GBPRandom": {
                "popsize": 2500
            },
            "GBPCEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "DecentCEM": {
                "popsize": 100,
                "num_elites": 10,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-P": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-A": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-A-E": {
                "popsize": 100,
                "num_elites": 10,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-P-E": {
                "popsize": 100,
                "num_elites": 10,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        """ @brief: no cheating of the observation function
        """
        if isinstance(obs, np.ndarray):
            return obs
        else:
            return obs

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return obs + pred
        else:
            return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        """ @brief:

            x, _, theta, _ = data_dict['start_state']
            up_reward = np.cos(theta)
            distance_penalty_reward = -0.01 * (x ** 2)
            return up_reward + distance_penalty_reward
        """
        x = obs[:, 0]
        theta = obs[:, 2]
        if isinstance(obs, np.ndarray):
            return -(np.cos(theta) - 0.01 * (x ** 2))
        else:
            return -(tf.cos(theta) - 0.01 * (x ** 2))

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return np.sum(np.square(acs), axis=1) * 0.0
        else:
            return tf.reduce_sum(tf.square(acs), axis=1) * 0.0

    def nn_constructor(self, model_init_cfg, misc=None):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None),
            misc=misc
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = GymCartpoleConfigModule
