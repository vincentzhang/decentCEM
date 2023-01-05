# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

from . import base_policy
from . import tf_networks
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
        assert self.args.training_scheme in ['BC-AR', 'BC-AI', 'BC-AE']
        assert self.args.cem_type in ['POPLINA-INIT', 'POPLINA-REPLAN']

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

        self._input_ph['target_action'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='target_action'
        )
        self._input_ph['noise'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='noise'
        )

        self._update_operator['loss'] = tf.reduce_mean(
            tf.square(self._input_ph['target_action'] -
                      self._tensor['action'])
        )

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
        ).minimize(self._update_operator['loss'])
        logger.info("policy training learning rate: {}".format(
            self.args.policy_lr)
        )

    def construct_tf_summary(self, logdir, get_display_obs_fn=None, writer=None):
        """ set up tf summary operators """
        if self.args.training_scheme in ['BC-AR', 'BC-AI']:
            self._construct_tf_summary_worker(logdir, get_display_obs_fn)
        elif self.args.training_scheme in ['BC-AE']:
            # TODO: can perhaps be merged into the worker case later
            self._construct_tf_summary_for_ensemble(logdir, get_display_obs_fn, writer)

    def _construct_tf_summary_for_ensemble(self, logdir, get_display_obs_fn=None, writer=None):
        ops = []
        ops.append(tf.summary.scalar(self._name_scope+'/loss', self._update_operator['loss']))
        state_mean_op = utils.make_summary_vec(self._name_scope+'/state_mean', self._whitening_operator['state_mean'])
        state_std_op = utils.make_summary_vec(self._name_scope+'/state_std', self._whitening_operator['state_std'])
        for i in state_mean_op:
            ops.append(i)
        for i in state_std_op:
            ops.append(i)
        action_op = utils.make_summary_vec(self._name_scope+'/action', self._tensor['action'][0, :])
        target_a_op = utils.make_summary_vec(self._name_scope+'/target_action', self._input_ph['target_action'][0, :])
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

    def _construct_tf_summary_worker(self, logdir, get_display_obs_fn=None): 
        ops = []
        ops.append(tf.summary.scalar('loss', self._update_operator['loss']))
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

    def _img_callback(self, feed_dict):
        # TODO, update
        pass
        #val_out, val_out2, misc_out = self._session.run([self._img_tensors, self._img_tensors2, self._tensor['per_state_loss']], feed_dict=feed_dict)
        # val_out contains the needed values for plot
        #img_op = self._get_img_op(val_out, val_out2, misc_out) 
        #img_summary_merged = tf.summary.merge(img_op)
        #img_summary = self._session.run(img_summary_merged)
        #return img_summary

    def train(self, data_dict, training_info={}, episode=None):
        
        # Step 1: update the running mean???
        self._set_whitening_var(data_dict['whitening_stats'])

        # Step 2: data processing
        if self.args.training_scheme == 'BC-AR':
            data_dict['target_action'] = data_dict['action']  # for training
            imaginary_dataset = training_info['imaginary_dataset']
            for key in ['noise']: #, 'sample_mu', 'sample_std']:
                data_dict[key] = imaginary_dataset[key]
            training_keys = ['start_state', 'target_action', 'noise']
        elif self.args.training_scheme == 'BC-AI':
            # add imaginary data to the dataset
            imaginary_dataset = training_info['imaginary_dataset']
            for key in ['start_state', 'action']:
                data_dict[key] = \
                    np.concatenate([data_dict[key], imaginary_dataset[key]])
            data_dict['target_action'] = data_dict['action']  # for training
            training_keys = ['start_state', 'target_action']
        elif self.args.training_scheme == 'BC-AE':
            # in this mode, we only train with imaginary data since the real data is only from the best worker
            imaginary_dataset = training_info['imaginary_dataset']
            for key in ['start_state', 'action']:
                data_dict[key] = imaginary_dataset[key]
            data_dict['target_action'] = data_dict['action']  # for training
            training_keys = ['start_state', 'target_action']
        else:
            raise NotImplementedError

        # the keys here for loading the placeholders
        self.optimize_weights(data_dict, training_keys, episode)
