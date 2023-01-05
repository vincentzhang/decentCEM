# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import base_policy
from . import tf_networks
from . import tf_utils
from . import whitening_util
from dmbrl.misc import logger
from ... import utils


class policy_network(base_policy.base_policy_network):
    ''' @brief:
            In this object class, we define the network structure, the restore
            function and save function.

        @self.args.training_scheme
            @BC-AR: (action space) behavior cloning with the real data
            @BC-AI: (action space) behavior cloning using imaginary dataset.

            @AVG-R: (weight space) behavior cloning by setting the weight to
                the average of the weights selected during sampling
            @BC-PR: (weight space) behavior cloning by distilling the policy
                produced by the weights during sampling
            @AVG-I: (weight space) AVG-R but with imaginary dataset
            @BC-PI: (weight space) BC-PR but with imaginary dataset
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        assert self.args.cem_type in ['POPLINP-SEP', 'POPLINP-UNI']
        assert self.args.training_scheme in ['BC-PR', 'BC-PI', 'BC-PE']

    def build_network(self):
        """ @brief: Note that build_network is only needed for the training
        """
        network_shape = [self._observation_size] + \
            self.args.policy_network_shape + [self._action_size]
        num_layer = len(network_shape) - 1
        act_type = ['tanh'] * (num_layer - 1) + [None]
        norm_type = [None] * (num_layer - 1) + [None]
        init_data = []
        # TODO: be careful when it comes to batchnorm
        assert norm_type[0] is not 'batchnorm' and \
            norm_type[0] is not 'batch_norm'

        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std

        self._MLP = tf_networks.W_MLP(
            dims=network_shape, scope=self._name_scope, train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )
        self._target_MLP = tf_networks.W_MLP(
            dims=network_shape, scope='target_'+self._name_scope, train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        # fetch all the trainable variables
        self._set_var_list()
        
        self._MLP_structure_saved = False # flag if the MLP nns file has been created yet

    def build_loss(self):
        """ @brief: the MLP is used to generate samples,
            while the target_MLP is used during the training. target_MLP is
            always older than the MLP, and we feed the dataset into target_MLP
            to train MLP.

            After each update, we synchronize target_MLP by copying weights from
            MLP.
        """

        self._build_ph()
        self._tensor, self._update_operator = {}, {}
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'target_state', self._observation_size
        )

        # the weight input_ph is always set to 0.0
        self._input_ph['weight'] = tf.placeholder(
            shape=[None, self._MLP.get_weight_size()],
            dtype=tf.float32, name='weight_noise'
        )
        # the actual weight generated from the planning
        self._input_ph['target_weight'] = tf.placeholder(
            shape=[None, self._MLP.get_weight_size()], dtype=tf.float32,
            name='target_weight_noise'
        )
        self._tensor['net_input'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['target_net_input'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['target_state_mean']
        ) / self._whitening_operator['target_state_std']

        # the output policy of the network
        self._tensor['action'] = self._MLP(
            self._tensor['net_input'], self._input_ph['weight']
        )
        self._tensor['target_action'] = self._target_MLP(
            self._tensor['target_net_input'],
            self._input_ph['target_weight']
        )

        # the distillation loss
        self._update_operator['loss'] = tf.reduce_mean(
            tf.square(self._tensor['target_action'] -
                      self._tensor['action'])
        )
        self._target_MLP_var_list = self._target_MLP.get_variable_list()
        self._MLP_var_list = self._MLP.get_variable_list()

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
        ).minimize(self._update_operator['loss'],
                   var_list=self._MLP_var_list)
        logger.info("policy training learning rate: {}".format(
            self.args.policy_lr)
        )

        # synchronize the weights
        self._get_weight = tf_utils.get_network_weights(
            self._session, self._MLP_var_list, self._name_scope
        )
        self._set_weight = tf_utils.set_network_weights(
            self._session, self._target_MLP_var_list, 'target_' + self._name_scope
        )
        
        if self.args.training_scheme not in ['BC-PE']:
            self._session.run(tf.variables_initializer(tf.global_variables()))
            self.sync_weight()

    def sync_weight(self):
        # synchronize the two networks if needed
        self._set_weight(self._get_weight())     # set the target MLP

    def construct_tf_summary(self, logdir, get_display_obs_fn=None, writer=None):
        """ set up tf summary operators """
        if self.args.training_scheme in ['BC-PR', 'BC-PI', 'BC-PE']:
            self._construct_tf_summary_worker(logdir, get_display_obs_fn, writer)

    def _construct_tf_summary_worker(self, logdir, get_display_obs_fn=None, writer=None):
        ops = []
        state_mean_op = utils.make_summary_vec(self._name_scope+'/state_mean', self._whitening_operator['state_mean'])
        state_std_op = utils.make_summary_vec(self._name_scope+'/state_std', self._whitening_operator['state_std'])
        for i in state_mean_op:
            ops.append(i)
        for i in state_std_op:
            ops.append(i)

        action_op = utils.make_summary_vec('action', self._tensor['action'][0, :])
        target_a_op = utils.make_summary_vec('target_action', self._tensor['target_action'][0, :])
        for i in action_op:
            ops.append(i)
        for i in target_a_op:
            ops.append(i)
        if get_display_obs_fn is not None:
            logger.info("applied get_display_obs_fn")
            # turn state_mean to env-specific var for better visualization
            env_dict = get_display_obs_fn(self._whitening_operator['state_mean'])
            for key in env_dict.keys():
                ops.append(tf.summary.scalar(self._name_scope+'/'+key, env_dict[key][0]))
        # what else needs to be shown?
        self._merged = tf.summary.merge(ops)
        if writer is None:
            subdir = 'tensorlog'
            self._writer = tf.summary.FileWriter('{}/{}'.format(logdir, subdir), graph=self._session.graph)
        else:
            self._writer = writer

    def train(self, data_dict, training_info={}, episode=None):

        # Step 1: update the running mean
        imaginary_dataset = training_info['imaginary_dataset']

        # Step 2: data processing
        if self.args.training_scheme in ['BC-PR', 'BC-PE']:
            data_dict['target_weight'] = data_dict['weight']  # for training
            data_dict['weight'] = 0.0 * data_dict['weight']  # for training

        elif self.args.training_scheme in ['BC-PI']:
            for key in ['start_state', 'weight']:
                data_dict[key] = \
                    np.concatenate([data_dict[key], imaginary_dataset[key]])
            data_dict['target_weight'] = data_dict['weight']  # for training
            data_dict['weight'] = 0.0 * data_dict['weight']  # for training

        else:
            raise NotImplementedError

        self._set_whitening_var(data_dict['whitening_stats'])
        self.optimize_weights(data_dict,
                              ['start_state', 'target_weight', 'weight'], episode)

        # synchronize the networks
        whitening_util.copy_whitening_var(data_dict['whitening_stats'],
                                          'state', 'target_state')
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator,
            data_dict['whitening_stats'], ['target_state']
        )
        if self.args.zero_weight == 'yes':
            logger.warning('Using Random Weights')
        else:
            self._set_weight(self._get_weight())
