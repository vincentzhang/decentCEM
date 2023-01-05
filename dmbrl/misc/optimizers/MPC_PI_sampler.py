from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

from .optimizer import Optimizer
from .policy_network import BC_KL_sampler_A_policy
from .policy_network import whitening_util
from dmbrl.misc import logger

STD_MAX = np.exp(1)
STD_MIN = np.exp(-3)
EPS = 1e-6

class MPCPISamplerOptimizer(Optimizer):
    """A Tensorflow-compatible optimizer for MPC policy distillation.

        We use a policy network to generate actions. 
        During training, the policy net generates a number of action proposals. 
            The distribution of the action proposals weighted by the finite-horizon reward is distilled to the policy network by KL loss.
        During testing, the action is the mean output from policy net. 

        training_scheme
        @BC-KL: behavior cloning training using forward KL loss(target, prediction).
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites=None, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, params=None):
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
        """
        super().__init__()
        if num_elites is None: # if not specified, use all the samples to fit the distribution
            num_elites = popsize
        self.max_iters, self.popsize, self.num_elites = \
            max_iters, popsize, num_elites
        self._params = params
        self._cem_type = params.cem_cfg.cem_type
        self.set_sol_dim(sol_dim)
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon = epsilon
        self.tf_sess = tf_session
        self._dataset = []
        self._tau = 1 #0.01 #0.1 # temperature for softmax TODO: pass it in as a param
        self._var = self._params.opt_cfg.init_var # variance of the gaussian policy, fix for now
        # TODO: there's this question of where this var should be set, whether on the optimizer level or policy level.
        logger.info("The policy variance of MPC-PI is set to {}".format(self._var))

        self._whitening_stats = whitening_util.init_whitening_stats(['state'])

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        # if self.tf_sess is not None:
        #     with self.tf_sess.graph.as_default():
        #         with tf.variable_scope("CEMSolver"):
        #             self.init_mean = \
        #                 tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])
        #             self.init_var = \
        #                 tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])

        self.num_opt_iters, self.mean  = None, None 
        self.tf_compatible, self.cost_function = None, None
        self.target_pi = None # target policy

        # behavior cloning network
        with self.tf_sess.graph.as_default():
            with tf.variable_scope("sampler_policy_network"):
                if self._params.cem_cfg.training_scheme in ['BC-KL-AR']:
                    self._policy_network = BC_KL_sampler_A_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'mpc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0],
                        self._var, 
                        [self.lb.reshape([self.plan_hor, -1])[0, :], self.ub.reshape([self.plan_hor, -1])[0, :]],
                        self.num_elites
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                else:
                    raise NotImplementedError

    def set_sol_dim(self, sol_dim):

        self.dO, self.dU = self._params.env.observation_space.shape[0], \
            self._params.env.action_space.shape[0]
        self.plan_hor = self._params.opt_cfg.plan_hor
        self.npart = self._params.prop_cfg.npart  # number of dynamics particles
        self.sol_dim = sol_dim  # add noise ontop of the policy output

    def clean_dataset(self):
        self._dataset = []

    def upload_dataset(self):
        # TODO: in the future, we might need several threads.
        # THE datasets should be a data structure in the MPC.py
        data = self._dataset
        self.clean_dataset()
        return data

    def train(self, obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer=None, episode=None):
        """ For training of the policy net, we would need the following
                1. all the samples where each sample is a sequence of action proposals
                2. sample weight: the cost for each sample sequence (sum of undiscounted reward)
                3. get the target distribution from the weighted samples
                4. get the policy distribution by computing prob from the gaussian policy of policy net
                5. backprop from the distribution loss
            This function is called at the end of every episode, where we need a dataset of all samples generated at each state
            We do not really need to store the true action seperately anymore.
            All we need is for each state, the action samples and their cost, along with the prob based on the policy net output
            The solution of control should really be the first action of the sample that has the highest cost

        """
        # construct the "on policy" data
        if imaginary_replay_buffer is None:
            imaginary_replay_buffer = self._dataset
        imaginary_dataset = {}

        # construct the dataset with imaginary data
        # only when there are data in the imaginary replay buffer
        # TODO: 'num_samples' is invariant to states so don't really need it for each state
        if imaginary_replay_buffer:
            imaginary_dataset = {
                key: [] for key in ['start_state', 'target_action', 'action_samples', 'target_pi', 'num_samples', 'returns', 'weights', 'target_mu', 'target_std', 'target_pdf']
                if key in imaginary_replay_buffer[0]
            }
            for timesteps in range(len(imaginary_replay_buffer)):
                # for each timestep, there's a dictionary
                for key in imaginary_dataset:
                    imaginary_dataset[key].append(
                        imaginary_replay_buffer[timesteps][key]
                    )
            # at this step, imaginary_dataset is a dictionary of list
            for key in imaginary_dataset:
                assert len(imaginary_dataset[key]) > 0
                imaginary_dataset[key] = np.array(imaginary_dataset[key]) # shape [timestep, shape0, shape1, ...]

        # get the new running stats of the states per dimension
        imaginary_dataset['state'] = imaginary_dataset['start_state'] # this is for updating the runnings stats which assumes the key is 'state'
        whitening_util.update_whitening_stats(self._whitening_stats,
                                              imaginary_dataset, 'state')
        imaginary_dataset.update({'whitening_stats': self._whitening_stats})

        # train the behavior cloning policy network
        self._policy_network.train(imaginary_dataset, episode=episode)

        # clear dataset
        self.clean_dataset()

    #def set_logdir(self, logdir):
    #    self._policy_network.set_ckpt_path(logdir)

    def save(self, ckpt_path=None, step=None): 
        # save the network
        self._policy_network.save_checkpoint(ckpt_path, step)
        self._policy_network.save_net_graph(ckpt_path)
    
    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):
        """ generate a base action for the current observation """
        with self.tf_sess.graph.as_default():
            self._proposed_act_seqs_ph = None
            self._proposed_act_seqs_res = None
            #self._proposed_act_seqs_prob = None

            # first_base_acs is used to recover the first choices
            self._first_base_acs = \
                self._policy_network.forward_network(sy_cur_obs[None], test=True)
            
            if self._cem_type == 'Gaussian':
                # set up the initial values
                proposed_act_seqs = []
                #proposed_act_seqs_prob = []
                obs = []
                cur_obs = tf.tile(sy_cur_obs[None],
                                  [self.popsize, 1]) # shape (popsize, dO)

                for t in range(self.plan_hor):
                    # here the policy should be stochastic to explore
                    policy_mean, policy_std = self._policy_network.forward_network(cur_obs[:-1, :]) # shape (popsize-1, dU)
                    policy_action = self._policy_network.forward_network(cur_obs[-1, :], test=True) # shape (1, dU)
                    # want to sample from a truncated norm
                    # TODO: might make the steps 2 - H using truncated_normal sampling directly without tfd
                    #policy_dist = tfd.TruncatedNormal(loc=policy_mean, scale=np.sqrt(self._var, dtype=np.float32), low=self.lb[0], high=self.ub[0])
                    policy_dist = tfd.TruncatedNormal(loc=policy_mean, scale=policy_std, low=self.lb[0], high=self.ub[0])
                    proposed_act = policy_dist.sample() # shape (popsize-1, dU)
                    proposed_act = tf.concat([proposed_act, policy_action], axis=0) 
                    #if t == 0:
                        # only compute the prob for the current time step
                        # for now, keep prob seperately for each action dimension
                        #proposed_act_prob = policy_dist.prob(proposed_act) # get the probability of the action proposals (popsize, dU)
                        #proposed_act_prob = tf.reduce_prod(proposed_act_prob, axis=1, keepdims=False) # get the joint prob of all actions dims (popsize,)

                    # issue in computing the probability
                    cur_obs = predict_next_obs(cur_obs, proposed_act)
                    obs.append(cur_obs)
                    proposed_act_seqs.append(proposed_act) # arr of (popsize, dU)
                    #proposed_act_seqs_prob.append(proposed_act_prob) # arr of (popsize,)

                # _proposed_act_seq_res is popsize sequences of actions
                # shape: plan_hor, popsize, dU
                # should add some noise to the proposed actions
                #cur_act = prosposed_act_seqs[0][0, :] # action for the current obs, shape dU
                
                self._proposed_act_seqs_res = tf.stack(proposed_act_seqs)
                #self._proposed_act_seqs_prob = tf.stack(proposed_act_seqs_prob) # (plan_hor, popsize,)
                self._debug_obs = tf.stack(obs)

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

        # no need for evolving distribution, one-shot sampling is fine
        # step 1, generate action proposals, setup was done in function forward_policy_propose
        act_seq = tf.transpose(self._proposed_act_seqs_res, [1, 0, 2]) # reshape to (popsize, plan_hor, dU)
        act_seq = tf.reshape(act_seq, [-1, self.sol_dim]) # reshape to (popsize, sol_dim)
        #act_seq_prob = tf.reduce_prod(tf.transpose(self._proposed_act_seqs_prob, [1, 0]), axis=1, keepdims=False) # (popsize, )

        # step 2: compute cost
        costs = cost_function(
            act_seq, cem_type=self._cem_type
            #tf_data_dict={'policy_network': self._policy_network}
        ) # costs shape (popsize, 1)

        # step 3: compute the weights based on the cost
        if self.num_elites < self.popsize: 
            # only keep the elites for the following steps
            values, indices = tf.nn.top_k(-costs, k=self.num_elites,
                                          sorted=True)
            samples = tf.gather(act_seq, indices)
            self.returns = -tf.gather(costs, indices)
        else:
            # use all samples
            samples = act_seq # shape (popsize, sol_dim)
            self.returns = - costs # shape (popsize, 1)
        
        #self.best_action_seq = tf.gather(samples, tf.argmax(self.returns)) # (1, sol_dim)
        self.act_samples = samples[:, :self.dU] # action samples for the current time step, shape (popsize, dU)

        # compute the weights for samples
        # returns shape (popsize, 1)
        self.weights = tf.nn.softmax(self.returns/self._tau, axis=0)

        # step 4, MWLE 
        weighted_samples = self.weights[:, None] * self.act_samples # weights (popsize, 1) * act_samples (popsize, dU)
        self.pi_mu = tf.reduce_sum(weighted_samples, axis=0) # (popsize, dU) -> (dU,)
        # add pi_std
        weighted_var = tf.reduce_sum( self.weights[:, None] * tf.square(self.act_samples - self.pi_mu), axis=0)
        self.pi_std = tf.clip_by_value(tf.sqrt( weighted_var ), STD_MIN, STD_MAX)

        #pi_dist = tfd.TruncatedNormal(loc=self.pi_mu, scale=np.sqrt(self._var, dtype=np.float32), low=self.lb[0], high=self.ub[0])
        pi_dist = tfd.TruncatedNormal(loc=self.pi_mu, scale=self.pi_std, low=self.lb[0], high=self.ub[0])
        self.pi = pi_dist.prob(self.act_samples) # compute the pdf on the action samples, shape (popsize, dU)
        #pi = tf.reduce_prod(pi_dist.prob(self.act_samples), axis=1, keepdims=False) # compute the pdf on the action samples, shape (popsize,)
        self.target_pi = self.pi / (EPS + tf.reduce_sum(self.pi, axis=0)) # this normalizes the pdf on each action dim such that it's a pmf over this discrete support
        #self.target_pi = tf.stop_gradient(pi) # this is the target distribution, no gradient
       
        # step 5, compute the prob of the first action based on the current policy net
        # one prob distribution on each action axis
        #pred_pi = proposed_act_prob # the prob of each actions dim (popsize, dU)
        #self.pred_pi = pred_pi / tf.reduce_sum(pred_pi, axis=0)
        
        # TODO: make sure it's needed?
        with self.tf_sess.graph.as_default():
            self.tf_sess.run(tf.variables_initializer(tf.global_variables()))

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): Dummy Argument. The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        assert self.tf_compatible

        # sol: the best action seq (1, sol_dim) from MPC
        # start_state: (dO)
        # action_samples: (popsize, dU)
        # target_pi: (popsize, dU)
        #sol, start_state, action_samples, target_pi, returns, weights, target_mu, target_std, target_pdf = self.tf_sess.run([self.best_action_seq, 
        sol, start_state, action_samples, target_pi, returns, weights, target_mu, target_std, target_pdf = self.tf_sess.run([self.pi_mu, 
                self.sy_cur_obs, self.act_samples, self.target_pi, self.returns, self.weights, self.pi_mu, self.pi_std, self.pi]) 

        assert np.sum(np.isnan(target_pi)) == 0
        #assert sol.shape[0] == self.sol_dim
        assert sol.shape[0] == self.dU
        assert start_state.shape[0] == self.dO
        assert action_samples.shape[0] == self.num_elites
        assert action_samples.shape[1] == self.dU
        assert target_pi.shape[0] == self.num_elites
        assert returns.shape[0] == self.num_elites
        assert weights.shape[0] == self.num_elites
        assert target_mu.shape[0] == self.dU
        assert target_std.shape[0] == self.dU
        assert target_pdf.shape[0] == self.num_elites
        
        prev_sol = self.update_prev_sol(per, dU, sol)
        #sol_action = sol[:self.dU]  # the real control signal
        sol_action = sol

        if self._params.cem_cfg.training_scheme == 'BC-KL-AR':
            # store the data for computing KL divergence between MPC policy and current policy
            # start_state: (dO,)
            # action_samples: (popsize, dU), only need the samples for the current time step
            # target_pi: (popsize, dU)
            # To train the policy net, we use it to predict the mean for the current state, then formulate a distribution 
            # to compute the pdf of the action_samples
            imaginary_data = {
                'num_samples': self.num_elites,
                'start_state': start_state,
                'action_samples': action_samples,
                'target_pi': target_pi,
                "returns": returns[:, None],
                "weights": weights[:, None],
                "target_mu": target_mu,
                "target_std": target_std,
                "target_pdf": target_pdf,
                "target_action": sol_action
            }
            self._dataset.append(imaginary_data)

        return sol_action, prev_sol

    def train_policy_network(self):
        return True

    def get_policy_network(self):
        return self._policy_network

    def obtain_test_solution(self, init_mean, init_var, per, dU, obs=None, average=False, ctrl_type='PolicyControl'):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if average:
            raise NotImplementedError
        else:
            if ctrl_type == 'PolicyControl':
                prev_sol = np.zeros(self.sol_dim) # only as a placeholder in this mode

                # the actual solution
                first_acs = self.tf_sess.run(self._first_base_acs)
                sol_action = first_acs.reshape([-1])  # the actual action to be used
            else: # MPC
                sol_action, prev_sol = self.obtain_solution(init_mean, init_var, per, dU, obs)

        return sol_action, prev_sol
