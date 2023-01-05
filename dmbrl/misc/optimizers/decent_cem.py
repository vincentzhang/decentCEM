from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

# local
from dmbrl.misc import logger
from .cem import CEMOptimizer 
from .. import utils


class DecentCEMOptimizer(CEMOptimizer):
    """A Tensorflow-compatible Decentralized CEM optimizer.
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25,
                 params=None):
        """Creates an instance of this class.

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
        self.cem_workers = []
        for idx in range(params.cem_cfg.ensemble_size):
            self.cem_workers.append(CEMOptimizer(sol_dim, max_iters, popsize, num_elites, tf_session, upper_bound, lower_bound, epsilon, alpha, params, str(idx)))
        
        # self._params = params


        if popsize / params.cem_cfg.ensemble_size < 1:
            raise ValueError("Number of ensemble instances must be at most the population size.")
        
        self._best_worker_idx = 0
        self._init_mean = None # used to keep the initial mean of sampling distribution
        self.tf_sess = self.cem_workers[0].tf_sess
        self._total_steps = 0
        self._debug = params.cem_cfg.debug_optimizer
        self._selection_count = np.zeros((params.cem_cfg.ensemble_size))


    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        for worker in self.cem_workers:
            worker.setup(cost_function, tf_compatible, True)

        # only initialize all vars at the end
        with self.tf_sess.graph.as_default():
            self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

    def reset(self):
        pass

    def save(self, *args, **kwargs): 
        pass

    def reset_prev_sol(self, prev_sol):
        self._init_mean = prev_sol
        return prev_sol

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

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
                sol_mu, prev_sol_worker, avg_cost = worker.obtain_solution(init_mean, init_var, per, dU, obs, return_avg_cost=True)
            else:
                sol_mu, prev_sol_worker, avg_cost = worker.obtain_solution(self._init_mean, init_var, per, dU, obs, return_avg_cost=True)
                
            # Need to take the minimum cost of the sol of the workers
            # print("Cost of the sol from worker {:d} is {:f}".format(i, avg_cost))
            if avg_cost < cost_min: 
                cost_min = avg_cost
                idx_min = i
                sol_action = sol_mu
                prev_sol = prev_sol_worker
 
            if self._debug:
                self._action_worker = utils.make_summary_vec_np('ensemble/action_worker_{}'.format(i), sol_mu[:dU])
                self._cost_worker = tf.Summary(value=[tf.Summary.Value(tag='ensemble/avgcost_worker_{}'.format(i), simple_value=avg_cost)] ) 
                self._writer.add_summary(self._action_worker, self._total_steps)
                self._writer.add_summary(self._cost_worker, self._total_steps)

        # update the index 
        self._best_worker_idx = idx_min 
        self._selection_count[idx_min] += 1 

        # update the tf summary
        if self._debug:
            self._best_worker = tf.Summary(value=[tf.Summary.Value(tag='ensemble/best_worker_idx', simple_value=self._best_worker_idx)] ) 
            self._writer.add_summary(self._best_worker, self._total_steps)
            self._selection_count_summary = utils.make_summary_vec_np('ensemble/selection_ratio', self._selection_count/sum(self._selection_count))
            self._writer.add_summary(self._selection_count_summary, self._total_steps)
        
        self._total_steps += 1

        return sol_action, prev_sol
 

    def obtain_test_solution(self, init_mean, init_var, per, dU, obs=None, average=False, ctrl_type='PolicyControl'):
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
 

    def set_tf_writer(self, logdir):
        self.cem_workers[0].set_tf_writer(logdir)
        self._writer = self.cem_workers[0]._writer
        for worker in self.cem_workers[1:]:
            worker.set_tf_writer(logdir, tf_writer=self._writer) 


    def get_tf_writer(self):
        return self._writer