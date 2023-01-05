from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import localtime, strftime

from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent
from dmbrl.misc import logger
import copy
import numpy as np
import tensorflow as tf


class MBExperiment:

    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.
                    .seed_train (int): Seed for training

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        self._params = params
        params.sim_cfg.misc = copy.copy(params)
        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())+"-{}".format(params.sim_cfg.seed_train)
        )

        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                ),
                params=params
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False, params=params))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        #self.policy.set_logdir(self.logdir)
        self.policy.construct_tf_summary(self.logdir)
        self.exp_tf_writer = self.policy.get_tf_writer()

        logger.set_file_handler(path=self.logdir)
        logger.info('Starting the experiments')
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        self.policy_train_freq = 1 # per 2 episodes

        # Set up the evaluation environment
        self.agent._setup_eval_env() # this sets the random seed to the eval env

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []
        test_traj_obs, test_traj_acs, test_traj_rets, test_traj_rets_ste = [], [], [], []
        episode_iter_id = []
        num_steps = []

        # Perform initial rollouts
        samples = []
        needed_num_steps = self.ninit_rollouts * self.task_hor
        finished_num_steps = 0

        while finished_num_steps < needed_num_steps:
            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy
                )
            )
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])
            traj_rets.append(sample[-1]["reward_sum"])
            finished_num_steps += len(samples[-1]["ac"])
            
        if self.ninit_rollouts > 0: # this trains the model, not policy
            num_steps.extend(finished_num_steps)
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )
                

        # Training loop
        for i in range(self.ntrain_iters):

            logger.info("####################################################################")
            logger.info("Starting training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            assert self.nrecord == 0

            needed_num_episodes = max(self.neval, self.nrollouts_per_iter) - self.nrecord
            needed_num_steps = self.task_hor * needed_num_episodes
            finished_num_steps = 0
            
            while finished_num_steps < needed_num_steps:
                # collect enough transitions in one iteration before training
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
                finished_num_steps += len(samples[-1]["ac"])

            # Each sample is a rollout of max length of the environment limit
            # Only log the return of the first neval rollout 
            logger.info("Training return obtained: {}".format(
                [sample["reward_sum"] for sample in samples[:self.neval]])
            )

            # Evaluation if needed
            # We run the evaluation for > needed_num_steps every training steps 
            if self._params.misc.ctrl_cfg.cem_cfg.test_policy > 0:
                test_data = []
                # self.agent._setup_eval_env() # this resets the eval env to the same random seed
                test_epochs = self._params.misc.ctrl_cfg.cem_cfg.test_policy_epochs 
                # the evaluation is on a fixed num of rollouts
                for _ in range(test_epochs):
                    test_data.append(
                        self.agent.sample(self.task_hor, self.policy,
                                          test_policy=True, average=False)
                    )
                test_traj_rets.extend([
                    np.mean([i_test_data["reward_sum"] for i_test_data in test_data])
                ])
                test_traj_rets_ste.extend([
                    np.std([i_test_data["reward_sum"] for i_test_data in test_data]) / np.sqrt(test_epochs)
                ])
                test_traj_obs.extend(
                    [i_test_data["obs"] for i_test_data in test_data]
                )
                test_traj_acs.extend(
                    [i_test_data["ac"] for i_test_data in test_data]
                )
                logger.info("Testing the agent at iteration {}, return in {} runs, mean+-stderr: {} +- {}".format(
                    i+1,
                    test_epochs, 
                    test_traj_rets[-1],
                    test_traj_rets_ste[-1]
                ))

            traj_obs.extend([sample["obs"] for sample in samples])
            traj_acs.extend([sample["ac"] for sample in samples])
            traj_rets.extend([sample["reward_sum"] for sample in samples])
            traj_rews.extend([sample["rewards"] for sample in samples])
            episode_iter_id.extend([i] * len(samples))
            num_steps.append(finished_num_steps) # num_steps is a list of steps in each rollout
            # why aren't we using all the samples??
            # samples = samples[:self.nrollouts_per_iter] # this causes that the model and policy net can be updated at different num of data

            #if i % 25 == 0:
            #    self.policy.dump_logs(self.logdir, iter_dir, i)
            if i % 25 == 0 or (i == self.ntrain_iters - 1):
                # only store data at a certain freq or at the last iteration for analysis
                # note this does not affect the evaluation since the new data are appended so we don't lose anything
                savemat(
                    os.path.join(self.logdir, "logs.mat"),
                    {
                        "observations": traj_obs,   # states during training rollouts
                        "actions": traj_acs,        # actions during training rollouts
                        "returns": traj_rets,       # returns during training rollouts
                        "rewards": traj_rews,       # rewards during training rollouts
                        "test_returns": test_traj_rets, # returns during testing rollouts
                        "test_returns_std_err": test_traj_rets_ste,# std error of the testing returns
                        "test_obs": test_traj_obs,  # states during testing rollouts
                        "test_acs": test_traj_acs,  # actions during testing rollouts
                        'episode_iter_id': episode_iter_id, # episode id of each sample
                        'ninit_rollouts': self.ninit_rollouts, # number of initial rollouts before the main loop, 0 in our experiments
                        'num_steps': num_steps      # number of steps in the rollout
                    }
                )
            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)
            
            # log the train/test return as a tf summary 
            if False:
                self.ret_train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_return", simple_value=samples[0]["reward_sum"])]) 
                self.ret_test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_return", simple_value=test_traj_rets[-1])]) 
                self.exp_tf_writer.add_summary(self.ret_train_summary, i) 
                self.exp_tf_writer.add_summary(self.ret_test_summary, i) 

            if i < self.ntrain_iters - 1: # no training at the last iteration
                train_policy =  i % self.policy_train_freq == 0
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples],
                    train_policy=train_policy,
                    episode=i
                )

                # policy is trained with imaginary data as well, which comes from those extra rollouts
                # model is strictly trained with real data
