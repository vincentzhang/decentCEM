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
        assert self.args.training_scheme in ['BC-AES']
        assert self.args.cem_type in ['POPLINA-INIT']

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

        self._MLP = tf_networks.MLP_MULTIHEAD(
            num_heads=self.args.ensemble_size,
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
        # this action is a list
        self._tensor['action'] = self._MLP(self._tensor['net_input'])

        local_loss = []
        for idx in range(self.args.ensemble_size):
            self._input_ph['target_action'+str(idx)] = tf.placeholder(
                tf.float32, [None, self._action_size], name='target_action'+str(idx)
            )
            
            if idx == 0: # this might be a bottleneck since it's faster if the action is flattened

                local_loss.append(tf.square(self._input_ph['target_action'+str(idx)] -
                        self._tensor['action'][idx]))
                #self._update_operator['loss'] = tf.reduce_mean(
                #tf.square(self._input_ph['target_action'+str(idx)] -
                #        self._tensor['action'][idx])
                #)
            else:
                local_loss.append(tf.square(self._input_ph['target_action'+str(idx)] -
                        self._tensor['action'][idx]))
                #self._update_operator['loss'] += tf.reduce_mean(
                #tf.square(self._input_ph['target_action'+str(idx)] -
                #        self._tensor['action'][idx])
                #)

        #self._update_operator['loss'] /= self.args.ensemble_size
        self._update_operator['loss'] = tf.reduce_mean(local_loss)

        opt =  tf.train.AdamOptimizer(learning_rate=self.args.policy_lr) 
        grads_and_vars = opt.compute_gradients(self._update_operator['loss'])
        grads_and_vars = [(g,v) for g, v in grads_and_vars if g is not None]
        layer_id = len(self.args.policy_network_shape) 
        weight = self.args.policy_lr_weight
        grads_and_vars = [(weight*g,v) if "layer_"+str(layer_id) not in v.name else (g,v) for g, v in grads_and_vars ]
        self._update_operator['update_op'] = opt.apply_gradients(grads_and_vars)
        #self._update_operator['update_op'] = tf.train.AdamOptimizer(
        #    learning_rate=self.args.policy_lr,
        #).minimize(self._update_operator['loss'])
        logger.info("policy training learning rate: {}".format(
            self.args.policy_lr)
        )

    def construct_tf_summary(self, logdir, get_display_obs_fn=None, writer=None):
        """ set up tf summary operators """
        ops = []
        ops.append(tf.summary.scalar(self._name_scope+'/loss', self._update_operator['loss']))
        state_mean_op = utils.make_summary_vec(self._name_scope+'/state_mean', self._whitening_operator['state_mean'])
        state_std_op = utils.make_summary_vec(self._name_scope+'/state_std', self._whitening_operator['state_std'])
        for i in state_mean_op:
            ops.append(i)
        for i in state_std_op:
            ops.append(i)

        for idx in range(self.args.ensemble_size):
            ops = self._construct_tf_summary_worker(ops, idx, logdir, get_display_obs_fn)

        self._merged = tf.summary.merge(ops)
        if writer is None:
            subdir = 'tensorlog'
            self._writer = tf.summary.FileWriter('{}/{}'.format(logdir, subdir), graph=self._session.graph)
        else:
            self._writer = writer
 
    def _construct_tf_summary_worker(self, ops, idx, logdir, get_display_obs_fn=None):
        action_op = utils.make_summary_vec(self._name_scope+'/action', self._tensor['action'][idx][0, :])
        target_a_op = utils.make_summary_vec(self._name_scope+'/target_action', self._input_ph['target_action'+str(idx)][0, :])
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
        return ops
        
    def _img_callback(self, feed_dict):
        pass

    def train(self, data_dict, training_info={}, episode=None):
        
        # Step 1: update the running mean???
        self._set_whitening_var(data_dict['whitening_stats'])

        # Step 2: data processing
        # in this mode, we only train with imaginary data since the real data is only from the best worker
        imaginary_dataset = training_info['imaginary_dataset']
        key_list = ['target_action'+str(idx) for idx in range(self.args.ensemble_size)]
        key_list.append('start_state')
        for key in key_list:
            data_dict[key] = imaginary_dataset[key]
        training_keys = key_list

        # the keys here for loading the placeholders
        self.optimize_weights(data_dict, training_keys, episode)
