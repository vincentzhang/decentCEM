from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from .optimizer import Optimizer
from .policy_network import BC_A_S_policy
from .policy_network import whitening_util
from .. import utils
from dmbrl.misc import logger
import time


class POPLINAESOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer for CEM ensemble.

        In CEM, we use a population based search algorithm (evolutionary search)
        This might be quite local, as it might be overfitting, and hard to find
        policy for a complex structure like humanoid.
        
        We use a policy network to choose the action.

        But even with the policy network, we still only perform local search around a trajectory.
        This optimizer attempts to use an ensemble of workers to explore a larger space.

        1. CEM strategy and Noise strategy:
            @POPLINA-INIT: Adding noise in the action space. Using a policy network as
                the initial proposal, and using CEM to get fine-grained action.
            @POPLINA-REPLAN: Adding noise in the action space. Using a policy network as
                the initial proposal, and using CEM to get fine-grained action.

        2. training_scheme
            @BC-AR: behavior cloning training only with the real data
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25,
                 params=None):
        """Creates an instance of this class.

        It's based on the POPLINA class where it creates and manages multiple copies of POPLINA.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.max_iters, self.popsize, self.num_elites = \
            max_iters, popsize, num_elites
        self._params = params
        self.num_heads = params.cem_cfg.ensemble_size
        assert self._params.cem_cfg.training_scheme in ['BC-AES']
        assert self._params.cem_cfg.cem_type in ['POPLINA-INIT']

        self._cem_type = params.cem_cfg.cem_type
        self.set_sol_dim(sol_dim)
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self._dataset = []

        self._whitening_stats = whitening_util.init_whitening_stats(['state'])

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.init_mean = [0] * self.num_heads
        self.init_var = [0] * self.num_heads 
        self.values = [0] * self.num_heads 
        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver"):
                    for idx in range(self.num_heads):
                        self.init_mean[idx] = \
                            tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])
                        self.init_var[idx] = \
                            tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])
                        self.values[idx] = tf.Variable(np.zeros([self.num_elites]), dtype=tf.float32)

        self.num_opt_iters, self.mean, self.var = [0]*self.num_heads,[0]*self.num_heads, [0]*self.num_heads
        self.tf_compatible, self.cost_function = None, None
        self._tf_writer = None
        self._topk_values = [0]*self.num_heads
        self.best_val =[0]*self.num_heads
        self.avg_cost =[0]*self.num_heads
        self.act_seq_cost =[0]*self.num_heads
        self._total_steps = 0 # this tracks the number of training steps
        self.init_returns = [0] * self.num_heads
        self.init_elites = [0] * self.num_heads
        self.best_sol, self.elites, self.returns = ([0]*self.num_heads for _ in range(3))

        # behavior cloning network
        with self.tf_sess.graph.as_default():
            with tf.variable_scope("bc_policy_network"):
                if self._params.cem_cfg.training_scheme in ['BC-AES']:
                    self._policy_network = BC_A_S_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                else:
                    raise NotImplementedError

        ### construct the CEM workers
        self._best_worker_idx = 0
        self._init_mean = None # used to keep the initial mean of sampling distribution
        self._debug = params.cem_cfg.debug_optimizer
        self._selection_count = np.zeros((self.num_heads))

    def set_tf_writer(self, tf_writer=None):
        if tf_writer is None:
            self._tf_writer = self._policy_network.get_tf_writer()
        else:
            self._tf_writer = tf_writer

    def reset_prev_sol(self, prev_sol):
        self._init_mean = prev_sol
        return prev_sol

    def set_sol_dim(self, sol_dim):
        self.dO, self.dU = self._params.env.observation_space.shape[0], \
            self._params.env.action_space.shape[0]
        self.plan_hor = self._params.opt_cfg.plan_hor
        self.npart = self._params.prop_cfg.npart  # number of dynamics particles

        if self._params.cem_cfg.cem_type in ['POPLINA-INIT']:
            self.sol_dim = sol_dim  # add noise ontop of the policy output
        else:
            raise NotImplementedError

    def clean_dataset(self):
        self._dataset = []

    def train(self, obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer=None, episode=None):
        """ organize the data to train the policy network 
            
            the index of the best worker for the current state is in self._best_worker_idx
            Only calls the training method for the best worker
        """
        # train each head of the policy network
        # construct the "on policy" data
        if imaginary_replay_buffer is None:
            imaginary_replay_buffer = self._dataset
        imaginary_dataset = {}

        # construct the dataset with fake data
        # only when there are data in the imaginary replay buffer
        if imaginary_replay_buffer:
            key_list = ['target_action'+str(idx) for idx in range(self.num_heads)]
            key_list.append('start_state')
            imaginary_dataset = {
                key: [] for key in key_list
                if key in imaginary_replay_buffer[0]
            }
            for timesteps in range(len(imaginary_replay_buffer)):
                for key in imaginary_dataset:
                    imaginary_dataset[key].append(
                        imaginary_replay_buffer[timesteps][key]
                    )
            for key in imaginary_dataset:
                assert len(imaginary_dataset[key]) > 0
                imaginary_dataset[key] = np.concatenate(imaginary_dataset[key])

        # the dataset with real data
        real_dataset = {
            'start_state': np.concatenate([i_traj[:-1] for i_traj in obs_trajs],
                                          axis=0),
            'action': np.concatenate(acs_trajs, axis=0),
        }
        real_dataset['state'] = real_dataset['start_state']

        # get the new running mean
        whitening_util.update_whitening_stats(self._whitening_stats,
                                              real_dataset, 'state')
        real_dataset.update({'whitening_stats': self._whitening_stats})

        # train the behavior cloning policy network
        self._policy_network.train(
            real_dataset, training_info={'imaginary_dataset': imaginary_dataset},
            episode=episode
        )

        # clear dataset
        self.clean_dataset()

    def save(self, ckpt_path=None, step=None): 
        # save the network
        self._policy_network.save_checkpoint(ckpt_path, step)
        self._policy_network.save_net_graph(ckpt_path)
 
    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):
        """ the output is a list of actions """
        with self.tf_sess.graph.as_default():
            with tf.variable_scope("forward"):
                self._proposed_act_seqs_ph = None
                self._proposed_act_seqs_res = []

                # first_base_acs is used to recover the first choices
                # the raw action output of the network 
                self._first_base_acs = self._policy_network.forward_network(sy_cur_obs[None])
                #self._first_base_acs = self.get_best_acs(raw_actions)[:self.dU]

                # set up the initial values
                cur_obs = tf.tile(sy_cur_obs[None],
                                [self.popsize * self.npart, 1])

                # propagate the trajectory using each head
                self._act_seq = [] # holds the acs sequence of each head
                for i_head in range(self.num_heads):
                    proposed_act_seqs = []
                    obs = []
                    for i_t in range(self.plan_hor):
                        proposed_act = self._policy_network.forward_network(cur_obs)[i_head]
                        cur_obs = predict_next_obs(cur_obs, proposed_act)
                        obs.append(cur_obs)
                        proposed_act_seqs.append(proposed_act)

                    self._proposed_act_seqs_res.append(tf.stack(proposed_act_seqs))
            
                    ### a faster way
                    act_seq = self._proposed_act_seqs_res[i_head][:, :1, :] 
                    ## shape: [hor, 1, dU] -> [1, sol_dim]
                    self._act_seq.append(tf.reshape(tf.transpose(act_seq, [1, 0, 2]), [1, -1]))

    def eval_gaussian_cost(self, cost_function, mu, var, idx):
        """ evaluate the cost of a gaussian N(mu, var) by taking the avg cost of samples """
        # Generate samples to estimate the cost of the Gaussian
        samples = tf.truncated_normal([self.num_elites, self.sol_dim],
                                          mu, tf.sqrt(var))

        # We reuse the proposed act seqs from the current policy network
        costs = cost_function(
                samples, cem_type=self._params.cem_cfg.cem_type,
                tf_data_dict={'policy_network': self._policy_network,
                              'proposed_act_seqs': self._proposed_act_seqs_res[idx][:, :self.num_elites, :]}
        )
        # take the average of the costs
        avg_cost = tf.reduce_mean(costs)
        return avg_cost

    def get_act_seq_cost(self, cost_function, idx):
        """ get a model rollout from current obs using the model """
        costs = cost_function(self._act_seq[idx])
        return costs[0]
                    
    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if not tf_compatible or self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible
        
        for i_head in range(self.num_heads):
            self._setup_cem_worker(cost_function, i_head)
        
        # only initialize all vars at the end
        self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

    def _setup_cem_worker(self, cost_function, idx):
        def continue_optimization_eval(t, mean, var, best_val, best_sol,
                                elites, returns, values):
            return tf.logical_and(tf.less(t, self.max_iters),
                                tf.reduce_max(var) > self.epsilon)

        def iteration_eval(t, mean, var, best_val, best_sol, elites, returns, values):
            samples = tf.truncated_normal([self.popsize, self.sol_dim],
                                        mean, tf.sqrt(var))  # the noise

            costs = cost_function(
                samples, cem_type=self._params.cem_cfg.cem_type,
                tf_data_dict={'policy_network': self._policy_network,
                            'proposed_act_seqs': self._proposed_act_seqs_res[idx]}
            )
            values, indices = tf.nn.top_k(-costs, k=self.num_elites,
                                        sorted=True)

            best_val, best_sol = tf.cond(
                tf.less(-values[0], best_val),
                lambda: (-values[0], samples[indices[0]]),
                lambda: (best_val, best_sol)
            )

            elites = tf.gather(samples, indices)
            returns = -tf.gather(costs, indices)
            new_mean = tf.reduce_mean(elites, axis=0)
            new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            return t + 1, mean, var, best_val, best_sol, elites, returns, values 

        def continue_optimization(t, mean, var, best_val, best_sol,
                                elites, returns):
            return tf.logical_and(tf.less(t, self.max_iters),
                                tf.reduce_max(var) > self.epsilon)

        def iteration(t, mean, var, best_val, best_sol, elites, returns):
            samples = tf.truncated_normal([self.popsize, self.sol_dim],
                                        mean, tf.sqrt(var))  # the noise

            costs = cost_function(
                samples, cem_type=self._params.cem_cfg.cem_type,
                tf_data_dict={'policy_network': self._policy_network,
                            'proposed_act_seqs': self._proposed_act_seqs_res[idx]}
            )
            values, indices = tf.nn.top_k(-costs, k=self.num_elites,
                                        sorted=True)

            best_val, best_sol = tf.cond(
                tf.less(-values[0], best_val),
                lambda: (-values[0], samples[indices[0]]),
                lambda: (best_val, best_sol)
            )

            elites = tf.gather(samples, indices)
            returns = -tf.gather(costs, indices)
            new_mean = tf.reduce_mean(elites, axis=0)
            new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            return t + 1, mean, var, best_val, best_sol, elites, returns 

        with self.tf_sess.graph.as_default():
            with tf.variable_scope("poplin_aes_setup"+str(idx)):
                self.init_returns[idx] = tf.Variable(np.zeros([self.num_elites]),
                                                dtype=tf.float32)

                self.init_elites[idx] = tf.tile(self.init_mean[idx][None, :],
                                        [self.num_elites, 1])
                                    
                if not self._params.cem_cfg.eval_cem_policy:
                    self.num_opt_iters[idx], self.mean[idx], self.var[idx], self.best_val[idx], \
                        self.best_sol[idx], self.elites[idx], self.returns[idx], self._topk_values[idx] = \
                        tf.while_loop(cond=continue_optimization_eval, body=iteration_eval,
                                    loop_vars=[0, self.init_mean[idx], self.init_var[idx],
                                                float("inf"), self.init_mean[idx],
                                                self.init_elites[idx], self.init_returns[idx], 
                                                self.values[idx]])
                    self.avg_cost[idx] = -tf.reduce_mean(self._topk_values[idx])
                else:
                    self.num_opt_iters[idx], self.mean[idx], self.var[idx], self.best_val[idx], \
                        self.best_sol[idx], self.elites[idx], self.returns[idx] = \
                        tf.while_loop(cond=continue_optimization, body=iteration,
                                    loop_vars=[0, self.init_mean[idx], self.init_var[idx],
                                                float("inf"), self.init_mean[idx],
                                                self.init_elites[idx], self.init_returns[idx]])
                    self.avg_cost[idx] = self.eval_gaussian_cost(cost_function, self.mean[idx], self.var[idx], idx)

                self.act_seq_cost[idx] = self.get_act_seq_cost(cost_function, idx)

    def reset(self):
        pass
    
    def obtain_solution(self, init_mean, init_var, per, dU, obs=None, store_data=True, return_var=True, return_avg_cost=True, return_baseacs=False):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
            store_data (boolean): Whether to store data. Set to False during inference
        """
        assert self.tf_compatible
        sol, solvar, num_opt_iters, elites, returns, first_base_acs, avg_cost = ([0] * self.num_heads for _ in range(7))
            
        #start = time.time()

        feed_dict = {}
        for idx in range(self.num_heads):
            if idx == self._best_worker_idx:
                feed_dict.update({self.init_mean[idx]: init_mean, self.init_var[idx]: init_var})
            else:
                feed_dict.update({self.init_mean[idx]: self._init_mean, self.init_var[idx]: init_var})

        if return_avg_cost:
            sol, solvar, num_opt_iters, elites, returns, \
                start_state, first_base_acs, \
                avg_cost = self.tf_sess.run(
                    [self.mean, self.var, self.num_opt_iters,
                    self.elites, self.returns, self.sy_cur_obs, self._first_base_acs,
                    self.avg_cost
                    ],
                    feed_dict=feed_dict
                )
        else:
            sol, solvar, num_opt_iters, elites, returns, \
                start_state, first_base_acs = self.tf_sess.run(
                    [self.mean, self.var, self.num_opt_iters,
                    self.elites, self.returns, self.sy_cur_obs, self._first_base_acs
                    ],
                    feed_dict=feed_dict
                )

        # for idx in range(self.num_heads):
        #     if idx == self._best_worker_idx:
        #         feed_dict = {self.init_mean[idx]: init_mean, self.init_var[idx]: init_var} 
        #     else:
        #         feed_dict = {self.init_mean[idx]: self._init_mean, self.init_var[idx]: init_var} 

        #     if return_avg_cost:
        #         sol[idx], solvar[idx], num_opt_iters[idx], elites[idx], returns[idx], \
        #             start_state, first_base_acs[idx], \
        #             avg_cost[idx] = self.tf_sess.run(
        #                 [self.mean[idx], self.var[idx], self.num_opt_iters[idx],
        #                 self.elites[idx], self.returns[idx], self.sy_cur_obs, self._first_base_acs[idx],
        #                 self.avg_cost[idx]
        #                 ],
        #                 feed_dict=feed_dict
        #             )
        #         assert len(avg_cost[idx].shape) == 0
        #     else:
        #         sol[idx], solvar[idx], num_opt_iters[idx], elites[idx], returns[idx], \
        #             start_state, first_base_acs[idx] = self.tf_sess.run(
        #                 [self.mean[idx], self.var[idx], self.num_opt_iters[idx],
        #                 self.elites[idx], self.returns[idx], self.sy_cur_obs, self._first_base_acs[idx]
        #                 ],
        #                 feed_dict=feed_dict
        #             )
        
        #logger.info("time spent in forward passing : {}".format(time.time() - start))
    
        assert start_state.shape[0] == self.dO
        assert first_base_acs[0].shape[0] == 1
        assert first_base_acs[0].shape[1] == self.dU
        assert elites[0].shape[0] == self.num_elites
        assert returns[0].shape[0] == self.num_elites
        
        # sol_action = first_base_acs + sol[:self.dU]  # the real control signal
 
        #### compare the workers to pick the best one
        cost_min = 1e5
        idx_min = 0
        
        if store_data:
            imaginary_data = {'start_state': start_state[None, :] }
        
        for i in range(self.num_heads):
            # Need to take the minimum cost of the sol of the workers
            # print("Cost of the sol from worker {:d} is {:f}".format(i, avg_cost))
            if avg_cost[i] < cost_min: 
                cost_min = avg_cost[i]
                idx_min = i
                sol_action = first_base_acs[i] + sol[i][:self.dU]
            if store_data:
                if self._debug:
                    self._action_worker = utils.make_summary_vec_np('ensemble/action_worker_{}'.format(i), first_base_acs[i][0] + sol[i][:self.dU])
                    self._cost_worker = tf.Summary(value=[tf.Summary.Value(tag='ensemble/avgcost_worker_{}'.format(i), simple_value=avg_cost[i])] ) 
                    self._baseacs_worker = utils.make_summary_vec_np('ensemble/base_action_worker_{}'.format(i), first_base_acs[i][0]) 
                    self._cem_acs_worker = utils.make_summary_vec_np('ensemble/cem_action_worker_{}'.format(i), sol[i][:self.dU]) 
                    self._tf_writer.add_summary(self._action_worker, self._total_steps)
                    self._tf_writer.add_summary(self._baseacs_worker, self._total_steps)
                    self._tf_writer.add_summary(self._cem_acs_worker, self._total_steps)
                    self._tf_writer.add_summary(self._cost_worker, self._total_steps)
                
                imaginary_data.update({ 'target_action'+str(i): first_base_acs[i] + sol[i][:self.dU] })

        # update the index 
        self._best_worker_idx = idx_min 
        self._selection_count[idx_min] += 1 

        # update the tf summary
        if store_data and self._debug:
            self._best_worker = tf.Summary(value=[tf.Summary.Value(tag='ensemble/best_worker_idx', simple_value=self._best_worker_idx)] ) 
            self._tf_writer.add_summary(self._best_worker, self._total_steps)
            self._selection_count_summary = utils.make_summary_vec_np('ensemble/selection_ratio', self._selection_count/sum(self._selection_count))
            self._tf_writer.add_summary(self._selection_count_summary, self._total_steps)

        if not self._params.cem_cfg.use_prev_sol:
            prev_sol = np.zeros(self.sol_dim) # only search around policy net output
        else: 
            prev_sol = self.update_prev_sol(per, dU, sol[idx_min]) # used to initialize the CEM of the next step
        sol_action = first_base_acs[idx_min] + sol[idx_min][:self.dU]  # the real control signal

        if store_data:
            self._dataset.append(imaginary_data)
            self._total_steps += 1 # only for training
        
        return sol_action, prev_sol
    
    def train_policy_network(self):
        return True

    def get_policy_network(self):
        return self._policy_network

    def obtain_test_solution(self, init_mean, init_var, per, dU, obs=None, average=False, ctrl_type='PolicyControl', return_avg_cost=True):
        """ For evaluation. Optimizes the cost function using the provided initial candidate distribution
        
            Arguments:
                init_mean (np.ndarray): The mean of the initial candidate distribution.
                init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if average:
            raise NotImplementedError
        else:
            if ctrl_type == 'PolicyControl':
                prev_sol = np.zeros(self.sol_dim) # only as a placeholder in this mode
                # the actual solution, this is a list of heads
                first_acs_workers = self.tf_sess.run(self._first_base_acs)
                # get the cost of the predicted traj using the policy net
                avg_cost_workers = self.tf_sess.run(self.act_seq_cost) # just avg of 1 traj in this case
                idx_min = np.argmin(avg_cost_workers)
                # take the minimum cost of the sol of the workers
                cost_min = avg_cost_workers[idx_min]
                sol_action = first_acs_workers[idx_min].reshape([-1]) # the actual action to be used
            else: 
                # MPC with CEM
                sol_action, prev_sol = self.obtain_solution(init_mean, init_var, per, dU, obs, store_data=False, return_avg_cost=return_avg_cost)

        return sol_action, prev_sol
