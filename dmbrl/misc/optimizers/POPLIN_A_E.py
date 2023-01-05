from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from .POPLIN_A import POPLINAOptimizer
from .. import utils


class POPLINAEOptimizer(POPLINAOptimizer):
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
            @BC-AE: behavior cloning training only with the real data
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
        # construct the CEM workers
        self.cem_workers = []
        for idx in range(params.cem_cfg.ensemble_size):
            self.cem_workers.append(POPLINAOptimizer(sol_dim, max_iters, popsize, num_elites, tf_session, upper_bound, lower_bound, epsilon, alpha, params, str(idx)))
        self._best_worker_idx = 0
        self._init_mean = None # used to keep the initial mean of sampling distribution
        self.tf_sess = self.cem_workers[0].tf_sess
        self._total_steps = 0
        self._debug = params.cem_cfg.debug_optimizer
        self._selection_count = np.zeros((params.cem_cfg.ensemble_size))

    def reset_prev_sol(self, prev_sol):
        self._init_mean = prev_sol
        return prev_sol

    def train(self, obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer=None, episode=None):
        """ organize the data to train the policy network 
            
            the index of the best worker for the current state is in self._best_worker_idx
            Only calls the training method for the best worker
        """
        #train each policy network
        for worker in self.cem_workers:
            worker.train(obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer, episode)


    def save(self, ckpt_path=None, step=None):
        """ save the policy net of all the workers """
        for worker in self.cem_workers:
            worker.save(ckpt_path, step)
    
    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):
        for worker in self.cem_workers:
            worker.forward_policy_propose(predict_next_obs, sy_cur_obs)
    
    def set_sy_cur_obs(self, sy_cur_obs):
        # NOTE: it is a hack! be careful
        for worker in self.cem_workers:
            worker.set_sy_cur_obs(sy_cur_obs)

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        for worker in self.cem_workers:
            worker.setup(cost_function, tf_compatible, init_vars=False)
        # only initialize all vars at the end
        with self.tf_sess.graph.as_default():
            self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None, store_data=True):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
            store_data (boolean): Whether to store data. Set to False during inference
        """
        cost_min = 1e5
        idx_min = 0
        # in case all workers give high cost
        sol_action = init_mean
        prev_sol = init_mean
        
        for i, worker in enumerate(self.cem_workers):
            if i == self._best_worker_idx:
                if self._debug:
                    sol_mu, prev_sol_worker, avg_cost, base_acs = worker.obtain_solution(init_mean, init_var, per, dU, obs, store_data, return_avg_cost=True, return_baseacs=True)
                else:
                    sol_mu, prev_sol_worker, avg_cost = worker.obtain_solution(init_mean, init_var, per, dU, obs, store_data, return_avg_cost=True, return_baseacs=False)
            else:
                if self._debug:
                    sol_mu, prev_sol_worker, avg_cost, base_acs = worker.obtain_solution(self._init_mean, init_var, per, dU, obs, store_data, return_avg_cost=True, return_baseacs=True)
                else:
                    sol_mu, prev_sol_worker, avg_cost = worker.obtain_solution(self._init_mean, init_var, per, dU, obs, store_data, return_avg_cost=True, return_baseacs=False)
            # Need to take the minimum cost of the sol of the workers
            # print("Cost of the sol from worker {:d} is {:f}".format(i, avg_cost))
            if avg_cost < cost_min: 
                cost_min = avg_cost
                idx_min = i
                sol_action = sol_mu
                prev_sol = prev_sol_worker
            if self._debug:
                self._action_worker = utils.make_summary_vec_np('ensemble/action_worker_{}'.format(i), sol_mu[0])
                self._cost_worker = tf.Summary(value=[tf.Summary.Value(tag='ensemble/avgcost_worker_{}'.format(i), simple_value=avg_cost)] ) 
                self._baseacs_worker = utils.make_summary_vec_np('ensemble/base_action_worker_{}'.format(i), base_acs[0]) 
                self._cem_acs_worker = utils.make_summary_vec_np('ensemble/cem_action_worker_{}'.format(i), sol_mu[0] - base_acs[0]) 
                self._tf_writer.add_summary(self._action_worker, self._total_steps)
                self._tf_writer.add_summary(self._baseacs_worker, self._total_steps)
                self._tf_writer.add_summary(self._cem_acs_worker, self._total_steps)
                self._tf_writer.add_summary(self._cost_worker, self._total_steps)

        # update the index 
        self._best_worker_idx = idx_min 
        self._selection_count[idx_min] += 1 

        # update the tf summary
        if self._debug:
            self._best_worker = tf.Summary(value=[tf.Summary.Value(tag='ensemble/best_worker_idx', simple_value=self._best_worker_idx)] ) 
            self._tf_writer.add_summary(self._best_worker, self._total_steps)
            self._selection_count_summary = utils.make_summary_vec_np('ensemble/selection_ratio', self._selection_count/sum(self._selection_count))
            self._tf_writer.add_summary(self._selection_count_summary, self._total_steps)
        
        self._total_steps += 1

        return sol_action, prev_sol

    def get_policy_network(self):
        """ return a list of policy networks """
        pn = []
        for worker in self.cem_workers:
            pn.append(worker._policy_network)
        return pn

    def set_tf_writer(self, tf_writer=None):
        """ There should be only one tf_writer for the ensemble """
        if tf_writer is None:
            self._tf_writer = self.cem_workers[0]._policy_network.get_tf_writer()
        else:
            self._tf_writer = tf_writer
        for worker in self.cem_workers:
            worker.set_tf_writer(self._tf_writer)

    def obtain_test_solution(self, init_mean, init_var, per, dU, obs=None, average=False, ctrl_type='PolicyControl'):
        """ For evaluation. Optimizes the cost function using the provided initial candidate distribution
        
            Arguments:
                init_mean (np.ndarray): The mean of the initial candidate distribution.
                init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        cost_min = 1e5
        idx_min = 0
        # in case all workers give high cost
        sol_action = init_mean
        prev_sol = init_mean
        
        for i, worker in enumerate(self.cem_workers):
            if i == self._best_worker_idx:
                sol_mu, prev_sol_worker, act_seq_cost = worker.obtain_test_solution(init_mean, init_var, per, dU, obs=obs, average=average, ctrl_type=ctrl_type, return_avg_cost=True)
            else:
                sol_mu, prev_sol_worker, act_seq_cost = worker.obtain_test_solution(self._init_mean, init_var, per, dU, obs=obs, average=average, ctrl_type=ctrl_type, return_avg_cost=True)
            # take the minimum cost of the sol of the workers
            if act_seq_cost < cost_min: 
                cost_min = act_seq_cost
                idx_min = i
                sol_action = sol_mu
                prev_sol = prev_sol_worker
        
        # update the index 
        self._best_worker_idx = idx_min 
        return sol_action, prev_sol
