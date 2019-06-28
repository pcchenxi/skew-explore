"""
This code is modified from the implementation of "stable_baseline" 
(https://github.com/hill-a/stable-baselines) to match the interface 
for goal proposing environment
"""

import tensorflow as tf
import numpy as np
import warnings
import logging

from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy, mlp_extractor, nature_cnn
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm

def mlp_goal_encoder(goal_states, net_arch, act_fun):
    latent_goal = goal_states

    for idx, layer in enumerate(net_arch):
        if isinstance(layer, list):
            for idx, goal_layer_size in enumerate(layer):
                latent_goal = act_fun(linear(latent_goal, "goal_fc{}".format(idx), goal_layer_size, init_scale=np.sqrt(2)))        
        else:
            assert isinstance(layer, int), "Error: the net_arch list can only contain list and int"
            z_mu = tf.layers.dense(latent_goal, layer, name='goal_fc_mu')
            z_log_sigma_sq = tf.layers.dense(latent_goal, layer, name='goal_fc_sigma')

    return z_mu, z_log_sigma_sq


class GoalsConditionedMLPPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, goal_num = 1, goal_net_arch=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, goal_encoder='mlp', feature_extraction="mlp", **kwargs):
        super(GoalsConditionedMLPPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "mlp"))
        
        self.goal_encoder = goal_encoder
        # self._kwargs_check(feature_extraction, kwargs)
        self.name = "mlp_policy_"+goal_encoder

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]
        if goal_net_arch is None:
            goal_net_arch = [[64, 32], 2]

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            self.obs_goals = tf.placeholder(dtype=ob_space.dtype, shape=(None, ob_space.shape[0]), name='goal_states')
            obs_goals_reshape = self.obs_goals #tf.reshape(tensor=self.obs_goals, shape=(-1, self.goal_num * ob_space.shape[0]))

            if goal_encoder == "mlp_sample":
                logging.info('mlp encoder with z sampling')
                self.z_mu, self.z_log_sigma_sq = mlp_goal_encoder(obs_goals_reshape, goal_net_arch, act_fun)
                eps = tf.random_normal(
                    shape=tf.shape(self.z_log_sigma_sq),
                    mean=0, stddev=1, dtype=tf.float32)
                self.z_goal_sample = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
            if goal_encoder == "mlp":
                logging.info('mlp encoder with z mu')
                self.z_mu, self.z_log_sigma_sq = mlp_goal_encoder(obs_goals_reshape, goal_net_arch, act_fun)
                self.z_goal_sample = self.z_mu
            if goal_encoder == "no_encoder" or goal_encoder == 'no_goal_proposing':
                logging.info('no encoder for goal obs')
                self.z_goal_sample = tf.stop_gradient(self.obs_goals)

            # self.z_goal_input = tf.placeholder(dtype=ob_space.dtype, shape=self.z_mu.shape, name='input_z_goal')
            self.z_goal_input = tf.placeholder(dtype=ob_space.dtype, shape=self.z_goal_sample.shape, name='input_z_goal')
            self.use_input_z = tf.placeholder_with_default(False, shape=(), name='use_input_z')

            def use_sample(): return self.z_goal_sample
            def use_input(): return self.z_goal_input
            self.z_goal = tf.cond(self.use_input_z, use_input, use_sample)

            if goal_encoder == 'no_goal_proposing':
                latent = tf.layers.flatten(self.processed_obs)
            else:
                latent = tf.concat([tf.layers.flatten(self.processed_obs), self.z_goal], 1)
            logging_info = 'latent shape' + str(latent.shape)
            logging.info(logging_info)

            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(latent, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        if goal_encoder == "mlp_sample":
            kl_coef = 0.01
            latent_loss = -0.5 * tf.reduce_sum(
                1 + self.z_log_sigma_sq - tf.square(self.z_mu) - 
                tf.exp(self.z_log_sigma_sq), axis=1)

            self.latent_loss = tf.reduce_mean(latent_loss) * kl_coef
        else:
            self.latent_loss = 0

        self._setup_init()

    def step(self, obs, z_goal_input, goals, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})
        else:
            action, value, neglogp, z_goal_sample = self.sess.run([self.action, self.value_flat, self.neglogp, self.z_goal],
                                                   {self.obs_ph: obs, self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, z_goal_input, goals, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})

    def value(self, obs, z_goal_input, goals, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})

    def get_z_goal(self, obs_goals):
        return self.sess.run(self.z_goal_sample, {self.obs_goals: obs_goals})
        # return self.sess.run([self.z_goal_sample, self.z_mu, self.z_log_sigma_sq], {self.obs_goals: obs_goals})



class GoalsConditionedLSTMPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=150, reuse=False, layers=None, goal_num = 1, goal_net_arch=None, 
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, goal_encoder='mlp', feature_extraction="mlp",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(GoalsConditionedLSTMPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "mlp"))

        self.goal_encoder = goal_encoder
        # self._kwargs_check(feature_extraction, kwargs)
        self.name = "lstm_policy_"+goal_encoder

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            self.obs_goals = tf.placeholder(dtype=ob_space.dtype, shape=(None, ob_space.shape[0]), name='goal_states')
            obs_goals_reshape = self.obs_goals #tf.reshape(tensor=self.obs_goals, shape=(-1, self.goal_num * ob_space.shape[0]))                
                            
            if goal_encoder == "mlp_sample":
                logging.info('mlp encoder with z sampling')
                self.z_mu, self.z_log_sigma_sq = mlp_goal_encoder(obs_goals_reshape, goal_net_arch,act_fun)
                eps = tf.random_normal(
                    shape=tf.shape(self.z_log_sigma_sq),
                    mean=0, stddev=1, dtype=tf.float32)
                self.z_goal_sample = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
            if goal_encoder == "mlp":
                logging.info('mlp encoder with z mu')
                self.z_mu, self.z_log_sigma_sq = mlp_goal_encoder(obs_goals_reshape, goal_net_arch,act_fun)
                self.z_goal_sample = self.z_mu
            if goal_encoder == "no_encoder" or goal_encoder == 'no_goal_proposing':
                self.z_goal_sample = tf.stop_gradient(self.obs_goals)
                
            self.z_goal_input = tf.placeholder(dtype=ob_space.dtype, shape=self.z_goal_sample.shape, name='input_z_goal')
            self.use_input_z = tf.placeholder_with_default(False, shape=(), name='use_input_z')

            def use_sample(): return self.z_goal_sample
            def use_input(): return self.z_goal_input
            self.z_goal = tf.cond(self.use_input_z, use_input, use_sample)


            if goal_encoder == 'no_goal_proposing':
                latent = tf.layers.flatten(self.processed_obs)
            else:
                latent = tf.concat([tf.layers.flatten(self.processed_obs), self.z_goal], 1)
            logging.info('latent shape %f' % latent.shape)

            if net_arch is None:  # Legacy mode
                if layers is None:
                    layers = [64, 64]
                else:
                    warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = latent #tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                            layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

                self._value_fn = value_fn
            else:  # Use the new net_arch parameter
                if layers is not None:
                    warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
                if feature_extraction == "cnn":
                    raise NotImplementedError()

                # latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                    layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                            list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                            list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)


        if goal_encoder == "mlp_sample":
            kl_coef = 0.01
            latent_loss = -0.5 * tf.reduce_sum(
                1 + self.z_log_sigma_sq - tf.square(self.z_mu) - 
                tf.exp(self.z_log_sigma_sq), axis=1)

            self.latent_loss = tf.reduce_mean(latent_loss) * kl_coef
        else:
            self.latent_loss = 0
                  
        self._setup_init()

    def step(self, obs, z_goal_input, goals, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask,
                                 self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask,
                                 self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})

    def proba_step(self, obs, z_goal_input, goals, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask,
                                                self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})

    def value(self, obs, z_goal_input, goals, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask,
                                                self.obs_goals:goals, self.z_goal_input: z_goal_input, self.use_input_z:True})

    def get_z_goal(self, obs_goals):
        return self.sess.run(self.z_goal_sample, {self.obs_goals: obs_goals})
        # return self.sess.run([self.z_goal_sample, self.z_mu, self.z_log_sigma_sq], {self.obs_goals: obs_goals})


class LSTMPolicy(GoalsConditionedLSTMPolicy):
    def __init__(self, *args, **kwargs):
        super(LSTMPolicy, self).__init__(*args, **kwargs,
                                        goal_net_arch=[[64, 32], 2], goal_num = 1,
                                        net_arch=['lstm', dict(pi=[64, 32],
                                                        vf=[64, 32])],
                                        goal_encoder='no_encoder')

class LSTMPolicyRND(GoalsConditionedLSTMPolicy):
    def __init__(self, *args, **kwargs):
        super(LSTMPolicyRND, self).__init__(*args, **kwargs,
                                        goal_net_arch=[[64, 32], 2], goal_num = 1,
                                        net_arch=['lstm', dict(pi=[64, 32],
                                                        vf=[64, 32])],
                                        goal_encoder='no_goal_proposing')