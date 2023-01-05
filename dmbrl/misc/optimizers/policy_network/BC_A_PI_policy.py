# -----------------------------------------------------------------------------
#   @author:
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import base_policy
from . import tf_networks
from dmbrl.misc import logger
from ... import utils

LOG_STD_MAX = 1
LOG_STD_MIN = -3
LOG_VAR_MIN = 2 * LOG_STD_MIN
LOG_VAR_MAX = 2 * LOG_STD_MAX


class policy_network(base_policy.base_policy_network):
    ''' @brief:
            In this object class, we define the network structure, the restore
            function and save function.

        @self.args.training_scheme
            @BC-AR: (action space) behavior cloning with the real data
            @BC-AI: (action space) behavior cloning using imaginary dataset.
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size, action_limit):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        assert self.args.training_scheme in ['BC-AR', 'BC-AI']
        assert self.args.cem_type in ['POPLINA-INIT']
        self._lb, self._ub = action_limit # each has a shape of (dU)

    def build_network(self):
        """ @brief: Note that build_network is only needed for the training
        """
        network_shape = [self._observation_size] + \
            self.args.policy_network_shape + [self._action_size]
        num_layer = len(network_shape) - 1
        act_type = ['tanh'] * (num_layer - 1) + [None]
        norm_type = [None] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std

        self._MLP = tf_networks.MLP(
            dims=network_shape, scope=self._name_scope, train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        self._MLP_var = tf_networks.MLP(
            dims=network_shape, scope=self._name_scope+'_var', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        # fetch all the trainable variables
        self._set_var_list()

        self._MLP_structure_saved = False # flag if the MLP nns file has been created yet

    def build_loss(self):

        self._build_ph()
        self._tensor, self._update_operator = {}, {}

        # construct the input to the forward network, we normalize the state
        # input, and concatenate with the action
        self._tensor['normalized_start_state'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['net_input'] = self._tensor['normalized_start_state']

        # the output policy of the network
        self._tensor['action'] = self._MLP(self._tensor['net_input'])
        self._tensor['action'] = tf.clip_by_value(self._tensor['action'], self._lb, self._ub)
        self._tensor['var'] = tf.clip_by_value(self._MLP_var(self._tensor['net_input']), LOG_VAR_MIN, LOG_VAR_MAX)
        self._tensor['var'] = tf.exp(self._tensor['var'])
        
        self._input_ph['target_action'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='target_action'
        )
        self._input_ph['target_var'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='target_var'
        )
        self._input_ph['noise'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='noise'
        )

        self._update_operator['var_loss'] = tf.reduce_mean(
            tf.square(self._input_ph['target_var'] -
                      self._tensor['var'])
        ) 
        self._update_operator['action_loss'] = tf.reduce_mean(
            tf.square(self._input_ph['target_action'] -
                      self._tensor['action'])
        )
        self._update_operator['loss'] = self._update_operator['var_loss'] + self._update_operator['action_loss']

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
        ).minimize(self._update_operator['loss'])

        logger.info("policy training learning rate: {}".format(
            self.args.policy_lr)
        )

    def forward_network(self, observation):
        """ this overwrites the forward network method in the base class 
        
            Returns: 
                mu (tensor): mean of the gaussian policy
                std (tensor): std of the gaussian policy
        """
        normalized_start_state = (
            observation - self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']

        # the output policy of the network
        action = self._MLP(normalized_start_state)
        action = tf.clip_by_value(action, self._lb, self._ub)
        
        return action
    
    def forward_network_var(self, observation):
        normalized_start_state = (
            observation - self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        
        log_var = self._MLP_var(normalized_start_state)
        log_var = tf.clip_by_value(log_var, LOG_VAR_MIN, LOG_VAR_MAX)    
        var = tf.exp(log_var)

        return var

    def construct_tf_summary(self, logdir, get_display_obs_fn=None):
        ops = []
        ops.append(tf.summary.scalar('loss', self._update_operator['loss']))
        ops.append(tf.summary.scalar('var_loss', self._update_operator['var_loss']))
        ops.append(tf.summary.scalar('action_loss', self._update_operator['action_loss']))
        state_mean_op = utils.make_summary_vec('state_mean', self._whitening_operator['state_mean'])
        state_std_op = utils.make_summary_vec('state_std', self._whitening_operator['state_std'])
        for i in state_mean_op:
            ops.append(i)
        for i in state_std_op:
            ops.append(i)
        # added samples and distribution of CEM
        # shape (B, num_samples, dU)
        # these are computed during planning rather than policy network, so these should be placeholders instead of tensor
        #tf.summary.histogram('action_elites', self._input_ph['action_elites'][0, :, :])
        #tf.summary.histogram('action_samples', self._input_ph['action_samples'][0, :, :])
        #utils.make_summary_vec('action_mean', self._input_ph['sample_mu'][0, 0, :]) #TODO add more iters
        #utils.make_summary_vec('action_std', self._input_ph['sample_std'][0, 0, :])
        action_op = utils.make_summary_vec('action', self._tensor['action'][0, :])
        target_a_op = utils.make_summary_vec('target_action', self._input_ph['target_action'][0, :])
        var_op = utils.make_summary_vec('var', self._tensor['var'][0, :])
        target_var_op = utils.make_summary_vec('target_var', self._input_ph['target_var'][0, :])
        for i in action_op:
            ops.append(i)
        for i in target_a_op:
            ops.append(i)

        if get_display_obs_fn is not None:
            logger.info("applied get_display_obs_fn")
            # turn state_mean to env-specific var for better visualization
            env_dict = get_display_obs_fn(self._whitening_operator['state_mean'])
            for key in env_dict.keys():
                ops.append(tf.summary.scalar(key, env_dict[key][0]))
        #self._merged = tf.summary.merge_all()
        self._merged = tf.summary.merge(ops)
        subdir = 'tensorlog'
        self._writer = tf.summary.FileWriter('{}/{}'.format(logdir, subdir), graph=self._session.graph)

    def add_tf_summary(self):
        # TODO
        return True
    
    def train(self, data_dict, training_info={}, episode=None):

        # Step 1: update the running mean???
        self._set_whitening_var(data_dict['whitening_stats'])

        # Step 2: data processing
        if self.args.training_scheme == 'BC-AR':
            data_dict['target_action'] = data_dict['action']  # for training
            imaginary_dataset = training_info['imaginary_dataset']
            for key in ['noise', 'target_var']: #, 'sample_mu', 'sample_std']
                data_dict[key] = imaginary_dataset[key]
        elif self.args.training_scheme == 'BC-AI':
            # add imaginary data to the dataset
            imaginary_dataset = training_info['imaginary_dataset']
            for key in ['start_state', 'action']:
                data_dict[key] = \
                    np.concatenate([data_dict[key], imaginary_dataset[key]])
            data_dict['target_action'] = data_dict['action']  # for training
        else:
            raise NotImplementedError

        # the keys here for loading the placeholders
        #self.optimize_weights(data_dict, ['start_state', 'target_action', 'noise', 'sample_mu', 'sample_std'], episode)
        self.optimize_weights(data_dict, ['start_state', 'target_action', 'noise', 'target_var'], episode)
