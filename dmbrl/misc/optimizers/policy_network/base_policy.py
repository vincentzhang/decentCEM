# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import os
import io
import ast
from scipy.io import savemat, loadmat
import pickle
from matplotlib import pyplot as plt

from . import whitening_util
from . import tf_utils
from . import tf_networks
from dmbrl.misc import logger


def limit_action(action, lb=-1, ub=1):

    return tf.minimum(tf.maximum(action, lb), ub)


class base_policy_network(object):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):
        self.args = args

        self._session = session
        self._name_scope = name_scope

        self._observation_size = observation_size
        self._action_size = action_size

        # self._task_name = args.task_name
        self._network_shape = args.policy_network_shape
        self._add_tf_summary = args.debug_policy

        self._npr = np.random.RandomState(args.seed)

        self._whitening_operator = {}
        self._whitening_variable = []

        self._log_img = False # this is set to True by subclass if to log images

    def build_network(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def _build_ph(self):

        # initialize the running mean and std (whitening)
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'state', self._observation_size
        )

        # initialize the input placeholder
        self._input_ph = {
            'start_state': tf.placeholder(
                tf.float32, [None, self._observation_size], name='start_state'
            )
        }

    def get_input_placeholder(self):
        return self._input_ph

    def get_weights(self):
        return None

    def set_weights(self, weights_dict):
        pass

    def forward_network(self, observation, weight_vec=None):
        normalized_start_state = (
            observation - self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']

        # the output policy of the network
        if weight_vec is None:
            action = self._MLP(normalized_start_state)
        else:
            action = self._MLP(normalized_start_state, weight_vec)

        action = limit_action(action)

        return action

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list + self._whitening_variable

        self._set_network_weights = tf_utils.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_utils.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._saver = tf.train.Saver(self._get_network_weights.get_var_dict())
        
    #def set_ckpt_path(self, logdir):
    #    self._ckpt_path = os.path.join(logdir, "policynet_ckpt")

    def save_net_graph(self, ckpt_path):
        # Write structure to file
        if not self._MLP_structure_saved:
            with open(os.path.join(ckpt_path, "%s.nns" % self._name_scope), "w+") as f:
                f.write("%s\n" % repr(self._MLP))
            self._MLP_structure_saved = True
    
    def load_net_graph(self, ckpt_path):
        net_graph_f = os.path.join(ckpt_path, "%s.nns" % self._name_scope)
        if self._MLP_structure_saved:
            with open(net_graph_f, "r") as f:
                kwargs = {}
                for line in f:
                    if "MLP" in line:
                        continue
                    key, val = line.strip().split("=")
                    kwargs[key] = val
                kwargs["scope"] = str(kwargs["scope"])
                kwargs["num_layers"] = int(kwargs["num_layers"])
                kwargs["activation"] = ast.literal_eval(kwargs["activation"])
                kwargs["normalization"] = ast.literal_eval(kwargs["normalization"])
                kwargs["train"] = bool(kwargs["train"])
                kwargs["dims"] = ast.literal_eval(kwargs["dims"])
                kwargs["init_data"] = ast.literal_eval(kwargs["init_data"])

            self._MLP = tf_networks.MLP(
                dims=kwargs["dims"], scope=kwargs["scope"], train=kwargs["train"],
                activation_type=kwargs["activation"], normalizer_type=kwargs["normalization"],
                init_data=kwargs["init_data"]
            )
            logger.info("Loaded policy network MLP structure from {}".format(net_graph_f))
        else:
            logger.info("MLP structure file {} doesn't exist".format(net_graph_f))

    def load_checkpoint(self, ckpt_path):
        with self._session.as_default():
            #if ckpt_path is None:
            #    ckpt_path = self._ckpt_path
            #ckpt_path = os.path.join(ckpt_path, "{}.ckpt".format(self._name_scope)) 
            latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
            self._saver.restore(self._session, latest_ckpt)
            logger.info("policy net {} loaded from {}".format(self._name_scope, latest_ckpt))

    def save_checkpoint(self, ckpt_path, step):
        with self._session.as_default():
            #if ckpt_path is None:
            #    ckpt_path = self._ckpt_path
            ckpt_path = os.path.join(ckpt_path, "{}.ckpt".format(self._name_scope)) 
            saved_path = self._saver.save(self._session, ckpt_path, global_step=step)
            logger.info("policy net {} saved to {}".format(self._name_scope, saved_path))

    def load_checkpoint_pkl(self, ckpt_path):
        with self._session.as_default():
            #params_dict = loadmat(os.path.join(ckpt_path, "{}.mat".format(self._name_scope)))
            with open(os.path.join(ckpt_path, "{}.pkl".format(self._name_scope)), 'rb') as fp:
                params_dict = pickle.load(fp)
            keys = params_dict.keys()
            self._set_network_weights({k: params_dict[k] for k in keys if not k.startswith("__")})

    def save_checkpoint_pkl(self, ckpt_path):
        with self._session.as_default():
            weights_dict = self._get_network_weights()
            #savemat(os.path.join(ckpt_path, "{}.mat".format(self._name_scope)), weights_dict)
            with open(os.path.join(ckpt_path, "{}.pkl".format(self._name_scope)), 'wb') as fp:
                pickle.dump(weights_dict, fp)

    def get_whitening_operator(self):
        return self._whitening_operator

    def _set_whitening_var(self, whitening_stats):
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator, whitening_stats, ['state']
        )

    def train(self, data_dict, replay_buffer, training_info={}):
        raise NotImplementedError

    def eval(self, data_dict):
        raise NotImplementedError

    def act(self, data_dict):
        raise NotImplementedError

    def optimize_weights(self, data_dict, training_keys, episode):

        test_set_id = np.arange(len(data_dict['start_state']))
        num_test_data = int(len(test_set_id) * self.args.pct_testset)
        self._npr.shuffle(test_set_id)
        test_set = {key: data_dict[key][test_set_id][:num_test_data]
                    for key in training_keys}
        train_set = {key: data_dict[key][test_set_id][num_test_data:]
                     for key in training_keys}
        test_error = old_test_error = np.inf

        # supervised training the behavior (behavior cloning)
        for epoch in range(self.args.policy_epochs):
            total_batch_len = len(train_set['start_state'])
            total_batch_inds = np.arange(total_batch_len)
            self._npr.shuffle(total_batch_inds)
            num_minibatch = \
                max(total_batch_len // self.args.minibatch_size, 1)
            train_error = []

            for start in range(num_minibatch):
                start = start * self.args.minibatch_size
                end = min(start + self.args.minibatch_size, total_batch_len)
                batch_inds = total_batch_inds[start: end]
                feed_dict = {self._input_ph[key]: data_dict[key][batch_inds]
                             for key in training_keys}

                error, _ = self._session.run(
                    [self._update_operator['loss'],
                    self._update_operator['update_op']], feed_dict=feed_dict
                )

                train_error.append(error)

            # see the test error
            feed_dict = {self._input_ph[key]: test_set[key]
                         for key in training_keys}

            test_error = self._session.run(
                self._update_operator['loss'], feed_dict=feed_dict
            )
            logger.info('Epoch %d; Train Error: %.6f; Test Error: %.6f' %
                        (epoch, np.mean(train_error), test_error))

            if test_error > old_test_error and epoch % 5 == 0:
                # TODO: MAKE A COUNTER HERE
                logger.info('Early stoping')
                break
            else:
                old_test_error = test_error
        
        if self.add_tf_summary():
            # log tf summary
            if episode is not None:
                feed_dict = {self._input_ph[key]: test_set[key]
                            for key in training_keys}
                summary = self._session.run(self._merged, feed_dict=feed_dict)
                self._writer.add_summary(summary, episode)
                if self._log_img and episode % 10 ==0:
                    # log the image every 10 episodes
                    img_summary = self._img_callback(feed_dict)
                    self._writer.add_summary(img_summary, episode)

    def _img_callback(self, feed_dict):
        pass

    def _get_img_op(self, vals):
        pass

    def _get_scatter_img_op(self, x, y, name, family="ScatterPlot"):
        """ generate a scatter plot and save it to an image summary"""
        # Generate the plot into a buffer
        if isinstance(y, list): # if y is a list, plot two curves
            #print("predict = {}".format(y[0]))
            #print("target = {}".format(y[1]))
            image_buf = self._get_overlay_plot_buf(x, y)
        else:
            image_buf = self._get_plot_buf(x, y)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(image_buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add img summary
        img_summary_op = tf.summary.image(name, image, max_outputs=1, family=family)
        return img_summary_op

    def _get_twinaxis_scatter_img_op(self, x, y1, y2, name, family="ScatterPlot"):
        """ generate a scatter plot with twinaxis and save it to an image summary"""
        # Generate the plot into a buffer
        image_buf = self._get_overlay_twinaxis_plot_buf(x, y1, y2)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(image_buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add img summary
        img_summary_op = tf.summary.image(name, image, max_outputs=1, family=family)
        return img_summary_op

    def _get_overlay_twinaxis_plot_buf(self, x, y1, y2, label=["pred", "target", "pred_pdf", "target_pdf", "returns", "weights"]):
        """ A helper function to create an overlay pyplot plot and save to buffer """
        fig = plt.figure()
        ax = fig.add_subplot( 2,1,1 )
        
        # ... draw plot here ...
        # plot of predicted vs target pmf
        # as well as the pdf?
        # y1: pred, target, pred_mu, target_mu, pred_var, target_var, pred_pdf, target_pdf (var not used for now)
        # y2: weights, returns
        pred_color = 'y'
        target_color = 'b'
        ln_pred = ax.scatter(x, y1[0],label=label[0], color=pred_color, alpha=0.3)
        ln_target = ax.scatter(x, y1[1],label=label[1], color=target_color, alpha=0.3)
        ax.set_ylabel('pmf')
        ax_2 = ax.twinx()
        ax_2.set_ylabel('pdf')
        ln_vertical_pred = ax_2.axvline(x=y1[2], color=pred_color, alpha=0.3)
        ln_vertical_target = ax_2.axvline(x=y1[3], color=target_color, alpha=0.3)
        ln_pred_pdf = ax_2.scatter(x, y1[6],label=label[2], color=pred_color, marker='x', s=80, alpha=0.3)
        ln_target_pdf = ax_2.scatter(x, y1[7],label=label[3], color=target_color, marker='x', s=80, alpha=0.3)
        lns = [ln_pred, ln_target, ln_pred_pdf, ln_target_pdf]
        labs = [l.get_label() for l in lns]
        ax_2.legend(lns, labs)

        ax1 = fig.add_subplot( 2,1,2 )
        ln_returns = ax1.scatter(x, y2[0], label=label[4], color='b',  alpha=0.3)
        ax1.set_ylabel('returns')
        #ax.set_ylim(0, 0.05)

        ax1_2 = ax1.twinx()
        #color = 'cyan'
        ln_weights = ax1_2.scatter(x, y2[1], label=label[5], color='r', marker='+', s=80, alpha=0.3)
        ax1_2.set_ylabel('weights')
        lns = [ln_returns, ln_weights]
        labs = [l.get_label() for l in lns]
        ax1_2.legend(lns, labs)

        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        return buf

    def _get_overlay_plot_buf(self, x, y, label=["pred", "target"]):
        """ A helper function to create an overlay pyplot plot and save to buffer """
        fig = plt.figure()
        ax = fig.add_subplot( 1,1,1 )
        
        # ... draw plot here ...
        ax.scatter(x, y[0],label=label[0])
        ax.scatter(x, y[1],label=label[1])
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        return buf     
    
    def _get_plot_buf(self, x, y):
        """ A helper function to create a pyplot plot and save to buffer """
        plt.figure()
        
        # ... draw plot here ...
        plt.scatter(x,y)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        return buf     

    def add_tf_summary(self):
        return self._add_tf_summary
    
    def get_tf_writer(self):
        return self._writer
