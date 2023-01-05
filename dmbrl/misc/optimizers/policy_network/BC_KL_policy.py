# -----------------------------------------------------------------------------
#   @author:
# -----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import math as tfm 
import numpy as np
import time
from matplotlib import pyplot as plt

from . import base_policy
from . import tf_networks
from dmbrl.misc import logger
from dmbrl.modeling.layers import FC
from ... import utils

LOG_STD_MAX = 1
LOG_STD_MIN = -3


class policy_network(base_policy.base_policy_network):
    ''' @brief:
            It defines the policy network structure
        
        @self.args.training_scheme
            @BC-KL: (action space) behavior cloning with the distribution from the MPC reward
        '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size, var, action_limit, sample_size):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        assert self.args.training_scheme in ['BC-KL']
        assert self.args.cem_type in ['Gaussian']

        #self._var = var # variance of the policy net policy
        self._lb, self._ub = action_limit # each has a shape of (dU)
        self._sample_size = sample_size # this is the sample size of the action samples
        
        self._log_img = True # log images

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

        # this MLP predicts up to the second last layer, the two output mean and std is based on this
        self._MLP = tf_networks.MLP_TWOHEAD(
            dims=network_shape, scope=self._name_scope, train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        # fetch all the trainable variables
        self._set_var_list()

        self._MLP_structure_saved = False # flag if the MLP nns file has been created yet

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
        mu, log_std = self._MLP(normalized_start_state)
        #mu = tfm.clip_by_value_preserve_gradient(mu, self._lb, self._ub)
        #log_std = tfm.clip_by_value_preserve_gradient(log_std, LOG_STD_MIN, LOG_STD_MAX)    
        mu = tf.clip_by_value(mu, self._lb, self._ub)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)    
        std = tf.exp(log_std)

        return mu, std 

    def build_loss(self):

        self._build_ph()
        # add more input placeholder
        self._input_ph['action_samples'] = tf.placeholder(
            tf.float32, [None, self._sample_size, self._action_size], name='action_samples'
        ) # (N, popsize, dU)

        # add placeholder for returns and weights
        self._input_ph['returns'] = tf.placeholder(tf.float32, [None, self._sample_size, self._action_size], name='returns') 
        self._input_ph['weights'] = tf.placeholder(tf.float32, [None, self._sample_size, self._action_size], name='weights') 
        self._input_ph['target_mu'] = tf.placeholder(tf.float32, [None, self._action_size], name='target_mu') 
        self._input_ph['target_std'] = tf.placeholder(tf.float32, [None, self._action_size], name='target_std') 
        self._input_ph['target_pdf'] = tf.placeholder(tf.float32, [None, self._sample_size, self._action_size], name='target_pdf') 

        self._tensor, self._update_operator = {}, {}
        
        #self._tensor['pred_var'] =  tf.constant(self._var, shape=[self.args.minibatch_size], dtype=tf.float32, name='pred_var')  # hacky in the shape

        # construct the input to the forward network, we normalize the state
        # input, and concatenate with the action
        self._tensor['normalized_start_state'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['net_input'] = self._tensor['normalized_start_state']
        
        # the output policy of the network, shape (N, dU)
        self._tensor['action_mean'],  self._tensor['action_std']  = self._MLP(self._tensor['net_input'])
        self._tensor['action_mean'] = tf.clip_by_value(self._tensor['action_mean'][:, None, :], self._lb, self._ub) # shape -> (N, 1, dU) 
        self._tensor['action_std'] = tf.exp(tf.clip_by_value(self._tensor['action_std'][:, None, :], LOG_STD_MIN, LOG_STD_MAX)) # shape -> (N, 1, dU) 
        #self._tensor['action_mean'] = tfm.clip_by_value_preserve_gradient(self._tensor['action_mean'][:, None, :], self._lb, self._ub) # shape -> (N, 1, dU) 
        #self._tensor['action_std'] = tf.exp(tfm.clip_by_value_preserve_gradient(self._tensor['action_std'][:, None, :], LOG_STD_MIN, LOG_STD_MAX)) # shape -> (N, 1, dU) 

        # get predicted prob of the samples, 
        #policy_dist = tfd.TruncatedNormal(loc=self._tensor['action_mean'], scale=np.sqrt(self._var, dtype=np.float32), low=self._lb, high=self._ub)
        policy_dist = tfd.TruncatedNormal(loc=self._tensor['action_mean'], scale=self._tensor['action_std'], low=self._lb, high=self._ub)
        self.pred_pi = policy_dist.prob(self._input_ph['action_samples']) # get the probability of the action proposals (N, popsize, dU)
        self._tensor['pred_pi']= self.pred_pi / tf.reduce_sum(self.pred_pi, axis=1, keepdims=True) # (N, popsize, dU)
        self._input_ph['target_pi'] = tf.placeholder(
            tf.float32, [None, self._sample_size, self._action_size], name='target_pi'
        ) #N, popsize, dU

        pred_pi_pmf = tf.transpose(self._tensor['pred_pi'], [0, 2, 1]) # (N, popsize, dU) -> (N, dU, popsize)
        target_pi_pmf = tf.transpose(self._input_ph['target_pi'], [0, 2, 1]) # (N, popsize, dU) -> (N, dU, popsize)
        if self.args.loss_type == 'reverse': # reverse KL, (is the stop_gradient needed here?)
            KL_loss = tf.keras.losses.KLD(                        
                   pred_pi_pmf, target_pi_pmf) # KL is computed over popsize, so the output: (N, dU)
                   #pred_pi_pmf, tf.stop_gradient(target_pi_pmf)) # KL is computed over popsize, so the output: (N, dU)
        else: # forward KL
            KL_loss = tf.keras.losses.KLD(
                   target_pi_pmf, pred_pi_pmf) # KL is computed over popsize, so the output: (N, dU)
        self._update_operator['loss'] = tf.reduce_mean(KL_loss)

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.policy_lr,
        ).minimize(self._update_operator['loss'])
        logger.info("policy training loss function: {} KL".format(
            self.args.loss_type)
        )
        logger.info("policy training learning rate: {}".format(
            self.args.policy_lr)
        )

        # add an image op
        actions = self._input_ph['action_samples']  # N, popsize, dU, N is the minibatch size?
        pred = self._tensor['pred_pi']
        target = self._input_ph['target_pi']
        self._img_tensors = [actions, pred, target]
        self._img_tensors2 = [self._input_ph['returns'], self._input_ph['weights'], self._tensor['action_mean'], self._input_ph['target_mu'], 
                            self._tensor['action_std'], self._input_ph['target_std'], self.pred_pi, self._input_ph['target_pdf']]
        self._tensor['per_state_loss'] = tf.reduce_mean(KL_loss, axis=1) #(N,)
 
    def add_tf_summary(self):
        # TODO
        return True

    def construct_tf_summary(self, logdir, get_display_obs_fn=None):
        tf.summary.scalar('loss', self._update_operator['loss'])
        tf.summary.histogram('action_samples', self._input_ph['action_samples'][0, :, :])
        tf.summary.histogram('pred_pi', self._tensor['pred_pi'][0, :, :]) # show the first state 
        tf.summary.histogram('target_pi', self._input_ph['target_pi'][0, :, :])
        tf.summary.histogram('action_mean', self._tensor['action_mean'][0, :, :])
        tf.summary.histogram('target_std', self._input_ph['target_std'][0, :])
        tf.summary.histogram('action_std', self._tensor['action_std'][0, :, :])
        utils.make_summary_vec('action_mean', self._tensor['action_mean'][0, 0, :])
        utils.make_summary_vec('action_std', self._tensor['action_std'][0, 0, :])
        utils.make_summary_vec('target_std', self._input_ph['target_std'][0, :])
        utils.make_summary_vec('state_mean', self._whitening_operator['state_mean'])
        utils.make_summary_vec('state_std', self._whitening_operator['state_std'])
        if get_display_obs_fn is not None:
            logger.info("applied get_display_obs_fn")
            # turn state_mean to env-specific var for better visualization
            env_dict = get_display_obs_fn(self._whitening_operator['state_mean'])
            for key in env_dict.keys():
                tf.summary.scalar(key, env_dict[key][0])
        self._merged = tf.summary.merge_all()
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        subdir = 'tensorlog'
        self._writer = tf.summary.FileWriter('{}/{}'.format(logdir, subdir), graph=self._session.graph)

    def _get_img_op(self, vals, vals2, postfix=None, num_iter=10):
        """ get a list of image summary ops """
        actions, pred, target = vals
        returns, weights, pred_mu, target_mu, pred_std, target_std, pred_pdf, target_pdf = vals2
        img_op = []
        for i in range(num_iter):
            # for each state
            name_postfix = str(i) if postfix is None else "{:.3f}".format(postfix[i]) 
            #img_op.append(self._get_scatter_img_op(actions[i, :, 0], [pred[i, :, 0], target[i, :, 0]], name='action_dist_'+name_postfix, family='action_dist') )
            img_op.append(self._get_twinaxis_scatter_img_op(actions[i, :, 0], # target_mu [N, dU], pred_mu[N, 1, dU]
             y1=[ pred[i, :, 0], target[i, :, 0], pred_mu[i, 0, :], target_mu[i, :], pred_std[i], target_std[i], pred_pdf[i, :, 0], target_pdf[i, :, 0]], 
             y2=[ returns[i, :, 0], weights[i, :, 0]],
             name='action_dist_'+name_postfix, family="action_dist"))
        return img_op

    def _img_callback(self, feed_dict):
        val_out, val_out2, misc_out = self._session.run([self._img_tensors, self._img_tensors2, self._tensor['per_state_loss']], feed_dict=feed_dict)
        # val_out contains the needed values for plot
        img_op = self._get_img_op(val_out, val_out2, misc_out) 
        img_summary_merged = tf.summary.merge(img_op)
        img_summary = self._session.run(img_summary_merged)
        return img_summary

    def train(self, data_dict, training_info={}, episode=None):
        
        # Step 1: update the running mean???
        self._set_whitening_var(data_dict['whitening_stats'])

        # Step 2: data processing
        if self.args.training_scheme == 'BC-KL':
            self.optimize_weights(data_dict, ['start_state', 'action_samples', 'target_pi', 'returns', 'weights', 'target_mu', 'target_std', 'target_pdf'], episode) 
        else:
            raise NotImplementedError
