import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class Actor():
    def __init__(self, state_dim, action_dim, action_max, learning_rate = 0.001, tau = 0.05, log_std_min = -20, log_std_max = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.tau = tau
        self.action_max = action_max
        self.log_std_min = -20
        self.log_std_max = 2

        self.tfd = tfp.distributions

        self.actor = self._bulid_model(state_dim, action_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _bulid_model(self, state_dim, action_dim, units=(400, 300, 100)):   
        state = Input(shape=state_dim)
        x = Dense(units[0], name="L0", activation="relu")(state)
        for index in range(1, len(units)):
            x = Dense(units[index], name="L{}".format(index), activation="relu")(x)

        actions_mean = Dense(action_dim, name="Out_mean")(x)
        actions_std = Dense(action_dim, name="Out_std")(x)

        model = Model(inputs=state, outputs=[actions_mean, actions_std])
        return model

    def act(self, state, test=False, use_random=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)
        if use_random:
            a = tf.random.uniform(shape=[1, self.action_dim], minval = -1, maxval = 1, dtype=tf.float64)
            a = a * self.action_max
        else:
            means, log_stds = self.actor.predict(state)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)

            a, _ = self.process_actions(means, log_stds, test=test)
        return a

    def process_actions(self, mean, log_std, test=False, eps=1e-6):
        std = tf.math.exp(log_std)
        raw_actions = mean

        if not test:
            raw_actions += tf.random.normal(shape=mean.shape, dtype=tf.float64) * std

        log_prob_u = self.tfd.Normal(loc=mean, scale=std).log_prob(raw_actions)
        actions = tf.math.tanh(raw_actions)
        # ******** probability densitiy (Enforcing action bounds)******** #
        log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps), axis=1, keepdims=True)
        
        actions = actions * self.action_max

        return actions, log_prob

    def save(self, name):
        self.actor.save(name)

    def load(self, name):
        self.actor = tf.keras.models.load_model(name)