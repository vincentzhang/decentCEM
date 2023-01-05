from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import io
from tensorflow.python.ops import tensor_array_ops
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from .optimizer import Optimizer
from .policy_network import BC_A_policy
from .policy_network import BC_WD_policy
from .policy_network import BC_WA_policy
from .policy_network import gan_policy
from .policy_network import wgan_policy
from .policy_network import whitening_util


class POPLINAOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer.

        In CEM, we use a population based search algorithm (evolutionary search)
        This might be quite local, as it might be overfitting, and hard to find
        policy for a complex structure like humanoid.

        We use a policy network to choose the action.

        1. CEM strategy and Noise strategy:
            @POPLINA-INIT: Adding noise in the action space. Using a policy network as
                the initial proposal, and using CEM to get fine-grained action.
            @POPLINA-REPLAN: Adding noise in the action space. Using a policy network as
                the initial proposal, and using CEM to get fine-grained action.
            @POPLINP-SEP: Adding noise in the weight space. Using a PW-CEM policy.
                For each output, we have separate noise
            @POPLINP-UNI: Adding noise in the weight space. Using a PW-CEM
                policy for each candaites, with different CEM noise.
            @PACEM: noise in the activation space (this might not be useful)

        2. training_scheme
            @BC-AR: behavior cloning training only with the real data
            @BC-AI: behavior cloning training, train towards the action (added
                by the noise) during planning (imaginary_dataset).
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25,
                 params=None, name_suffix=""):
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
        super().__init__()
        self.max_iters, self.popsize, self.num_elites = \
            max_iters, popsize, num_elites
        self._params = params
        self._cem_type = params.cem_cfg.cem_type
        self.set_sol_dim(sol_dim)
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self._dataset = []

        self._whitening_stats = whitening_util.init_whitening_stats(['state'])

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.name_suffix = name_suffix

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver"+self.name_suffix):
                    self.init_mean = \
                        tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])
                    self.init_var = \
                        tf.placeholder(dtype=tf.float32, shape=[self.sol_dim])
                    self.values = tf.Variable(np.zeros([self.num_elites]), dtype=tf.float32)

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None
        self._tf_writer = None
        self._sample_mean, self._sample_var = None, None
        self._sample_elites, self._sample_returns = None, None
        self._topk_values = None
        self._total_steps = 0 # this tracks the number of training steps

        with self.tf_sess.graph.as_default():
            with tf.variable_scope("debug_info"+self.name_suffix):
                # use tensorarray to store the intermediate value of mean,var
                # size has max_iters+1 due to adding the initial value
                self.sample_mean = tensor_array_ops.TensorArray(dtype=tf.float32, 
                                                    size = self.max_iters+1,
                                                    dynamic_size=False,
                                                    clear_after_read=False,
                                                    element_shape=[self.sol_dim]
                )
                self.sample_var = tensor_array_ops.TensorArray(dtype=tf.float32, 
                                                    size = self.max_iters+1,
                                                    dynamic_size=False,
                                                    clear_after_read=False,
                                                    element_shape=[self.sol_dim]
                )
                self.sample_elites = tensor_array_ops.TensorArray(dtype=tf.float32, 
                                                    size = self.max_iters,
                                                    dynamic_size=False,
                                                    clear_after_read=False,
                                                    element_shape=[self.num_elites]
                )
                self.sample_returns = tensor_array_ops.TensorArray(dtype=tf.float32, 
                                                    size = self.max_iters,
                                                    dynamic_size=False,
                                                    clear_after_read=False,
                                                    element_shape=[self.num_elites]
                )
                
        # behavior cloning network
        with self.tf_sess.graph.as_default():
            with tf.variable_scope("bc_policy_network"+self.name_suffix):
                if self._params.cem_cfg.training_scheme in ['BC-AR', 'BC-AI', 'BC-AE']:
                    self._policy_network = BC_A_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network'+self.name_suffix,
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.training_scheme in ['BC-PI', 'BC-PR', 'BC-PE']:
                    self._policy_network = BC_WD_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network'+self.name_suffix,
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.training_scheme in ['AVG-I', 'AVG-R', 'AVG-RE']:
                    self._policy_network = BC_WA_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network'+self.name_suffix,
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.gan_type == 'GAN' and \
                        self._params.cem_cfg.training_scheme in \
                        ['GAN-R', 'GAN-I']:
                    self._policy_network = gan_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                elif self._params.cem_cfg.gan_type == 'WGAN' and \
                        self._params.cem_cfg.training_scheme in \
                        ['GAN-R', 'GAN-I']:
                    self._policy_network = wgan_policy.policy_network(
                        self._params.cem_cfg, self.tf_sess,
                        'proposal_bc_network',
                        self._params.env.observation_space.shape[0],
                        self._params.env.action_space.shape[0]
                    )
                    self._policy_network.build_network()
                    self._policy_network.build_loss()
                else:
                    raise NotImplementedError
        
    def set_tf_writer(self, tf_writer=None):
        if tf_writer is None:
            self._tf_writer = self._policy_network.get_tf_writer()
        else:
            self._tf_writer = tf_writer

    def set_sol_dim(self, sol_dim):

        self.dO, self.dU = self._params.env.observation_space.shape[0], \
            self._params.env.action_space.shape[0]
        self.plan_hor = self._params.opt_cfg.plan_hor
        self.npart = self._params.prop_cfg.npart  # number of dynamics particles

        if self._params.cem_cfg.cem_type in ['POPLINA-INIT', 'POPLINA-REPLAN']:
            self.sol_dim = sol_dim  # add noise ontop of the policy output
        elif self._params.cem_cfg.cem_type == 'POPLINP-SEP':
            # policy network: [dO=17, 64, 64, dU=6], sol_dim = 5568
            # policy network: [dO=17, 64, dU=6], sol_dim = 1472
            # policy network: [dO=17, dU=6], sol_dim = 112
            policy_shape = self._params.cem_cfg.policy_network_shape
            weight_shape = [self.dO] + policy_shape + [self.dU]
            self.sol_dim = 0
            for i_input in range(len(weight_shape) - 1):
                self.sol_dim += \
                    weight_shape[i_input] * weight_shape[i_input + 1]
                self.sol_dim += weight_shape[i_input + 1]

            self._weight_size = self.sol_dim  # passed to the compile_cost()
            self.sol_dim *= self.plan_hor

        elif self._params.cem_cfg.cem_type == 'POPLINP-UNI':
            policy_shape = self._params.cem_cfg.policy_network_shape
            weight_shape = [self.dO] + policy_shape + [self.dU]
            self.sol_dim = 0
            for i_input in range(len(weight_shape) - 1):
                self.sol_dim += \
                    weight_shape[i_input] * weight_shape[i_input + 1]
                self.sol_dim += weight_shape[i_input + 1]

            self._weight_size = self.sol_dim
        else:
            raise NotImplementedError

    def clean_dataset(self):
        self._dataset = []

    def upload_dataset(self):
        # TODO: in the future, we might need several threads.
        # THE datasets should be a data structure in the MPC.py
        data = self._dataset
        self.clean_dataset()
        return data

    def train(self, obs_trajs, acs_trajs, rews_trajs, imaginary_replay_buffer=None, episode=None):
        # construct the "on policy" data
        if imaginary_replay_buffer is None:
            imaginary_replay_buffer = self._dataset
        imaginary_dataset = {}

        # construct the dataset with fake data
        # only when there are data in the imaginary replay buffer
        if imaginary_replay_buffer:
            imaginary_dataset = {
                key: [] for key in ['start_state', 'action', 'return', 'weight', 'noise']#, 'sample_mu', 'sample_std']
                if key in imaginary_replay_buffer[0]
            }
            for timesteps in range(len(imaginary_replay_buffer)):
                # range the ordering
                '''
                indices = np.argsort(imaginary_replay_buffer[timesteps]['return'])
                # TODO: what about the itermediate steps?
                indices = indices[-self._params.cem_cfg.training_top_k:]
                '''
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
        if imaginary_replay_buffer:
            if 'sol_weight' in imaginary_replay_buffer[0]:
                real_dataset['weight'] = np.array(
                    [imaginary_replay_buffer[i]['sol_weight']
                    for i in range(len(imaginary_replay_buffer))]
                )
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

    #def set_logdir(self, logdir):
    #    self._policy_network.set_ckpt_path(logdir)

    def save(self, ckpt_path=None, step=None): 
        # save the network
        self._policy_network.save_checkpoint(ckpt_path, step)
        self._policy_network.save_net_graph(ckpt_path)
    
    def forward_policy_propose(self, predict_next_obs, sy_cur_obs):

        with self.tf_sess.graph.as_default():
            with tf.variable_scope("forward"+self.name_suffix):
                self._proposed_act_seqs_ph = None
                self._proposed_act_seqs_res = None

                # first_base_acs is used to recover the first choices
                self._first_base_acs = \
                    self._policy_network.forward_network(sy_cur_obs[None])

        if self._params.cem_cfg.cem_type == 'POPLINA-INIT':
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("forward"+self.name_suffix):
                    # set up the initial values
                    proposed_act_seqs = []
                    obs = []
                    cur_obs = tf.tile(sy_cur_obs[None],
                                    [self.popsize * self.npart, 1])

                    for i_t in range(self.plan_hor):

                        proposed_act = self._policy_network.forward_network(cur_obs)
                        cur_obs = predict_next_obs(cur_obs, proposed_act)
                        obs.append(cur_obs)
                        proposed_act_seqs.append(proposed_act)

                    self._proposed_act_seqs_res = tf.stack(proposed_act_seqs)
                    self._debug_obs = tf.stack(obs)
            
                    if self._params.cem_cfg.training_scheme == 'BC-AE':
                        ### a faster way
                        act_seq = self._proposed_act_seqs_res[:, :1, :] 
                        self._act_seq = tf.reshape(tf.transpose(act_seq, [1, 0, 2]), [1, -1])

                        ### a slower way
                        # set up the initial values
                        #act_seq = []
                        #obs = []
                        #cur_obs = sy_cur_obs[None]
                        
                        #for _ in range(self.plan_hor):
                        #    proposed_act = self._policy_network.forward_network(cur_obs)
                        #    cur_obs = predict_next_obs(cur_obs, proposed_act)
                        #    act_seq.append(proposed_act)
                        
                        ## shape: [hor, 1, dU] -> [1, sol_dim]
                        #self._act_seq = tf.reshape(tf.transpose(tf.stack(act_seq), [1, 0, 2]), [1, -1])

        else:
            pass

    def eval_gaussian_cost(self, cost_function, mu, var):
        """ evaluate the cost of a gaussian N(mu, var) by taking the avg cost of samples """
        # Generate samples to estimate the cost of the Gaussian
        samples = tf.truncated_normal([self.num_elites, self.sol_dim],
                                          mu, tf.sqrt(var))

        # We reuse the proposed act seqs from the current policy network
        # TODO: An alternative is to replan based on the policy net + gaussian 
        costs = cost_function(
                samples, cem_type=self._params.cem_cfg.cem_type,
                tf_data_dict={'policy_network': self._policy_network,
                              'proposed_act_seqs': self._proposed_act_seqs_res[:, :self.num_elites, :]}
        )
        # take the average of the costs
        avg_cost = tf.reduce_mean(costs)
        return avg_cost
    
    def get_act_seq_cost(self, cost_function):
        """ get a model rollout from current obs using the model """
        costs = cost_function(self._act_seq)
        return costs[0]

    def get_act_seq_cost_orig(self, cost_function):
        """ get a model rollout from current obs using the model """
        # [hor, popsize, dU] -> [hor, 1, dU] 
        proposed_act_seqs = self._proposed_act_seqs_res[:, 0:1, :]
        # [1, hor*dU]
        proposed_act_seqs = tf.reshape( tf.transpose(proposed_act_seqs, [1, 0, 2]), [1, -1])
        # [1]
        costs = cost_function(proposed_act_seqs)
        return costs[0]

    def setup(self, cost_function, tf_compatible, init_vars=True):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.
            init_vars(bool): True if the vars should be initialized now, False for later.

        Returns: None
        """
        if not tf_compatible or self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible
        # self._summary = None # 

        if self._params.cem_cfg.training_scheme == 'BC-AE' and (not self._params.cem_cfg.eval_cem_policy):
            def continue_optimization(t, mean, var, best_val, best_sol,
                                    elites, returns, sample_mean, sample_var, sample_elites, sample_returns, values):
                return tf.logical_and(tf.less(t, self.max_iters),
                                    tf.reduce_max(var) > self.epsilon)

            def iteration(t, mean, var, best_val, best_sol, elites, returns, sample_mean, sample_var, sample_elites, sample_returns, values):
                samples = tf.truncated_normal([self.popsize, self.sol_dim],
                                            mean, tf.sqrt(var))  # the noise

                costs = cost_function(
                    samples, cem_type=self._params.cem_cfg.cem_type,
                    tf_data_dict={'policy_network': self._policy_network,
                                'proposed_act_seqs': self._proposed_act_seqs_res}
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

                sample_mean = sample_mean.write(t+1, mean)
                sample_var = sample_var.write(t+1, var)

                sample_elites = sample_elites.write(t, elites[:, 0]) # (max_iters, num_elites, sol_dim)
                sample_returns = sample_returns.write(t, returns) # (max_iters, num_elites)

                return t + 1, mean, var, best_val, best_sol, elites, returns, sample_mean, sample_var, sample_elites, sample_returns, values

        else:
            def continue_optimization(t, mean, var, best_val, best_sol,
                                    elites, returns, sample_mean, sample_var, sample_elites, sample_returns):
                return tf.logical_and(tf.less(t, self.max_iters),
                                    tf.reduce_max(var) > self.epsilon)

            def iteration(t, mean, var, best_val, best_sol, elites, returns, sample_mean, sample_var, sample_elites, sample_returns):
                samples = tf.truncated_normal([self.popsize, self.sol_dim],
                                            mean, tf.sqrt(var))  # the noise

                costs = cost_function(
                    samples, cem_type=self._params.cem_cfg.cem_type,
                    tf_data_dict={'policy_network': self._policy_network,
                                'proposed_act_seqs': self._proposed_act_seqs_res}
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

                sample_mean = sample_mean.write(t+1, mean)
                sample_var = sample_var.write(t+1, var)

                sample_elites = sample_elites.write(t, elites[:, 0]) # (max_iters, num_elites, sol_dim)
                sample_returns = sample_returns.write(t, returns) # (max_iters, num_elites)

                return t + 1, mean, var, best_val, best_sol, elites, returns, sample_mean, sample_var, sample_elites, sample_returns

        with self.tf_sess.graph.as_default():
            with tf.variable_scope("poplina_setup"+self.name_suffix):
                self.init_returns = tf.Variable(np.zeros([self.num_elites]),
                                                dtype=tf.float32)

                self.init_elites = tf.tile(self.init_mean[None, :],
                                        [self.num_elites, 1])
                                    
                self.sample_mean = self.sample_mean.write(0, self.init_mean)
                self.sample_var = self.sample_var.write(0, self.init_var)

                if self._params.cem_cfg.training_scheme == 'BC-AE' and (not self._params.cem_cfg.eval_cem_policy):
                    self.num_opt_iters, self.mean, self.var, self.best_val, \
                        self.best_sol, self.elites, self.returns, self._sample_mean, self._sample_var, \
                        self._sample_elites, self._sample_returns, self._topk_values = \
                        tf.while_loop(cond=continue_optimization, body=iteration,
                                    loop_vars=[0, self.init_mean, self.init_var,
                                                float("inf"), self.init_mean,
                                                self.init_elites, self.init_returns, 
                                                self.sample_mean, self.sample_var,
                                                self.sample_elites, self.sample_returns, self.values])
                else:
                    self.num_opt_iters, self.mean, self.var, self.best_val, \
                        self.best_sol, self.elites, self.returns, self._sample_mean, self._sample_var, \
                        self._sample_elites, self._sample_returns = \
                        tf.while_loop(cond=continue_optimization, body=iteration,
                                    loop_vars=[0, self.init_mean, self.init_var,
                                                float("inf"), self.init_mean,
                                                self.init_elites, self.init_returns, 
                                                self.sample_mean, self.sample_var,
                                                self.sample_elites, self.sample_returns])

                self._sample_mean = self._sample_mean.stack() # shape (max_iters+1, sol_dim)
                self._sample_var = self._sample_var.stack()
                self._sample_elites = self._sample_elites.stack()# shape (max_iters, num_elites, sol_dim)
                self._sample_returns = self._sample_returns.stack() # shape (max_iters, num_elites, 1)

                #self._summary = tf.summary.merge(self._get_summary_list())

                if init_vars:
                    self.tf_sess.run(tf.variables_initializer(tf.global_variables()))
                else:
                    # additional setup if no need to initialize variables
                    if self._params.cem_cfg.training_scheme == 'BC-AE' and (not self._params.cem_cfg.eval_cem_policy):
                        self.avg_cost = -tf.reduce_mean(self._topk_values) # this hopefully reduces one sampling/eval for the avg_cost
                    else:
                        self.avg_cost = self.eval_gaussian_cost(cost_function, self.mean, self.var)
                    #self.avg_cost = -tf.reduce_mean(self._topk_values) # this hopefully reduces one sampling/eval for the avg_cost
                    self.act_seq_cost = self.get_act_seq_cost(cost_function)
    
    def _get_summary_list(self):
        """ make a list of tf summary ops """
        summary_op_sample  = []
        for i in range(self.max_iters+1):
            summary_op_sample.append(tf.summary.scalar('sample_mean_iter{}'.format(i), self._sample_mean[i, 0]) )
            summary_op_sample.append(tf.summary.scalar('sample_std_iter{}'.format(i), tf.sqrt(self._sample_var)[i, 0]) )
        # add image summary to plot the samples and mean on the same plot
        return summary_op_sample

    def reset(self):
        pass

    def make_img_ops(self, mean, std, elites, returns, base_act, name, family='samples_cem'):
        ops = []
        ops.append(self._get_img_op(mean, std, elites, returns, base_act, name, family))
        merged_ops = tf.summary.merge(ops)
        return merged_ops

    def _get_img_op(self, mean, std, elites, returns, base_act, name, family='samples_cem'):
        # mean shape: 6,
        image_buf = self._get_img_buf(mean, std, elites, returns, base_act)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(image_buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add img summary
        img_op = tf.summary.image(name, image, max_outputs=1, family=family)
        return img_op

    def _get_img_buf(self, x1, x2, x3, x4, x5):
        """ A helper function to create a pyplot plot and save to buffer
            x1: mean
            x2: std
            x3: elites
            x4: returns
            x5: base_act
        """
        fig = plt.figure()

        # set up colormap with discrete colorbar
        cmap = plt.cm.tab10  # define the colormap

        # ... draw plot here ...
        ax = fig.add_subplot(2,1,1)
        for i, (x,y) in enumerate(zip(x3, x4)): # x = elites, 1
            ax.scatter(x+x5, y, alpha=0.3, color=cmap(i+1), label='samples@iter{}'.format(i+1)) # offset by 1 to match the iter for mu
        for i, mu in enumerate(x1):
            ax.axvline(x=mu+x5, lw=2, alpha=0.7, color=cmap(i), label='noise mu@iter{}'.format(i))
        ax.axvline(x=x5, color='c', linestyle='dashed', lw=2, alpha=0.8, label='Base_Action')
        ax.set_xlabel('Action')
        ax.set_ylabel('Returns')
        legend_elements = [Line2D([0], [0], color='k', marker='o', alpha=0.5, label='Elites'), 
                            Line2D([0], [0], color='k', alpha=0.5, label='Mean'),
                            Line2D([0], [0], color='c', linestyle='dashed', alpha=0.8, lw=2, label='Base_Action')
                            ]
        ax.legend(handles=legend_elements)

        ax2 = fig.add_subplot(2,1,2)
        ax2.errorbar(range(len(x1)), x1, yerr = x2, fmt ='o-', alpha=0.7, capsize=3, ecolor='hotpink') 
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Noise')
        #ax2.legend()
        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.BoundaryNorm([0,1,2,3,4,5,6], 6)), 
                ax=ax, ticks=[0,1,2,3,4,5], alpha=0.3, label='Iter')

        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()

        buf.seek(0)
        return buf     

    def _get_scatter_buf(self, x, y):
        """ A helper function to create a pyplot plot and save to buffer """
        plt.figure()
        
        # ... draw plot here ...
        plt.scatter(x,y)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()

        buf.seek(0)
        return buf     

    def obtain_solution(self, init_mean, init_var, per, dU, obs=None, store_data=True, return_var=False, return_avg_cost=False, return_baseacs=False):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
            store_data (boolean): Whether to store data. Set to False during inference
            return_var (boolean): whether to return variance as the third returned item
            return_avg_cost (boolean): whether to return the avg cost of the CEM policy
            return_baseacs (boolean): whether to return the first base action
        """
        assert self.tf_compatible
        if return_avg_cost:
            sol, solvar, num_opt_iters, elites, returns, \
                start_state, first_base_acs, sample_mean, sample_var, \
                    sample_elites, sample_returns, avg_cost = self.tf_sess.run(
                    [self.mean, self.var, self.num_opt_iters,
                    self.elites, self.returns, self.sy_cur_obs, self._first_base_acs,
                    self._sample_mean, self._sample_var, 
                    self._sample_elites, self._sample_returns, self.avg_cost
                    ],
                    feed_dict={self.init_mean: init_mean, self.init_var: init_var}
                )
            assert len(avg_cost.shape) == 0
        else:
            sol, solvar, num_opt_iters, elites, returns, \
                start_state, first_base_acs, sample_mean, sample_var, \
                    sample_elites, sample_returns = self.tf_sess.run(
                    [self.mean, self.var, self.num_opt_iters,
                    self.elites, self.returns, self.sy_cur_obs, self._first_base_acs,
                    self._sample_mean, self._sample_var, 
                    self._sample_elites, self._sample_returns
                    ],
                    feed_dict={self.init_mean: init_mean, self.init_var: init_var}
                )
        
        if store_data and self._params.cem_cfg.debug_optimizer:
            # here, we might as well add another run for tf summary directly from training episode
            # self._tf_writer.add_summary(summary, self._total_steps)
            if self._total_steps % 1000 == 0:
                img_summary_ops = self.make_img_ops(sample_mean[:, 0], np.sqrt(sample_var[:, 0]), sample_elites, sample_returns, first_base_acs[0, 0], name='samples'+self.name_suffix)
                img_summary = self.tf_sess.run(img_summary_ops)
                self._tf_writer.add_summary(img_summary, self._total_steps)

        if store_data:
            self._total_steps += 1

        '''
        propose_act_seq, propose_obs = self.tf_sess.run(
            [self._proposed_act_seqs_res, self._debug_obs],
            feed_dict={self.init_mean: init_mean, self.init_var: init_var}
        )
        '''

        assert start_state.shape[0] == self.dO
        assert first_base_acs.shape[0] == 1
        assert first_base_acs.shape[1] == self.dU
        assert elites.shape[0] == self.num_elites
        assert returns.shape[0] == self.num_elites
        assert sample_mean.shape[0] == self.max_iters+1
        assert sample_mean.shape[1] == self.sol_dim
        assert sample_var.shape[0] == self.max_iters+1
        assert sample_var.shape[1] == self.sol_dim
            
        if self._params.cem_cfg.training_scheme == 'BC-AE' and (not self._params.cem_cfg.use_prev_sol):
            prev_sol = np.zeros(self.sol_dim) # only search around policy net output
        else: 
            prev_sol = self.update_prev_sol(per, dU, sol) # used to initialize the CEM of the next step
        sol_action = first_base_acs + sol[:self.dU]  # the real control signal

        if store_data:
            if self._params.cem_cfg.training_scheme == 'BC-AI':
                # only store imaginary data when the training scheme says so
                imaginary_data = {
                    # TODO: use data that are not good? check lb size (all -1?)
                    'start_state': np.tile(start_state[None, :], [self.num_elites, 1]),
                    'action': np.maximum(
                        np.minimum(first_base_acs + elites[:, :self.dU],
                                self.ub[0]), self.lb[0]
                    ),
                    'return': returns.reshape([-1, 1])
                }
            elif self._params.cem_cfg.training_scheme == 'BC-AR':
                # store for tf summary
                imaginary_data = {
                        'noise':  sol[None, :self.dU],
                        #'sample_mu': sample_mean,
                        #'sample_std': np.sqrt(sample_var)
                    }
            else: # training_scheme == 'BC-AE'
                    # if it's ensemble mode, store some data for later
                    # store the best action mean
                    imaginary_data = { 'action':  sol_action , 'start_state': start_state[None, :] }
            self._dataset.append(imaginary_data)

        if return_var:
            return sol_action, prev_sol, solvar
        elif return_avg_cost and not return_baseacs:
            return sol_action, prev_sol, avg_cost
        elif return_avg_cost and return_baseacs:
            return sol_action, prev_sol, avg_cost, first_base_acs
        else:
            return sol_action, prev_sol

    def train_policy_network(self):
        return True

    def get_policy_network(self):
        return self._policy_network

    def obtain_test_solution(self, init_mean, init_var, per, dU, obs=None, average=False, ctrl_type='PolicyControl', return_avg_cost=False):
        """ For evaluation. Optimizes the cost function using the provided initial candidate distribution

            Arguments:
                init_mean (np.ndarray): The mean of the initial candidate distribution.
                init_var (np.ndarray): The variance of the initial candidate distribution.
                return_avg_cost (boolean): whether to return the avg cost of the CEM policy
                ctrl_type (str): 'PolicyControl' if only using policy network for the policy; 'MPC' if performing CEM-based MPC
        """
        if average:
            raise NotImplementedError
        else:
            if ctrl_type == 'PolicyControl':
                #sol = np.zeros(self.sol_dim)
                #prev_sol = self.update_prev_sol(per, dU, sol)
                prev_sol = np.zeros(self.sol_dim) # only as a placeholder in this mode
                # the actual solution
                first_acs = self.tf_sess.run(self._first_base_acs)
                sol_action = first_acs.reshape([-1])  # the actual action to be used

                if return_avg_cost:
                    # need to adapt it for the ensemble mode
                    # get the cost of the predicted traj using the policy net
                    avg_cost = self.tf_sess.run(self.act_seq_cost) # just avg of 1 traj in this case

            else: 
                # MPC with CEM
                if return_avg_cost:
                    sol_action, prev_sol, avg_cost = self.obtain_solution(init_mean, init_var, per, dU, obs, store_data=False, return_avg_cost=return_avg_cost)
                    if not self._params.cem_cfg.use_prev_sol:
                        prev_sol = np.zeros(self.sol_dim) # only search around policy net output
                else:
                    sol_action, prev_sol = self.obtain_solution(init_mean, init_var, per, dU, obs, store_data=False)

        if return_avg_cost:
            return sol_action, prev_sol, avg_cost
        else:
            return sol_action, prev_sol
