import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

class Critic():
    def __init__(self, state_dim, action_dim, learning_rate = 0.001, tau = 0.05):
        self.tau = tau
        self.learning_rate = learning_rate
        
        self.Q = self._bulid_model(state_dim, action_dim)

        self.Q_target = self._bulid_model(state_dim, action_dim)
        self.Q_target.set_weights(self.Q.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _bulid_model(self, state_dim, action_dim, units=(400, 200, 100)): # with batch normalization
        inputs = [Input(shape=state_dim), Input(shape=action_dim
        )]
        concat = Concatenate(axis=-1)(inputs)
        x = Dense(units[0], name="Hidden0", activation="relu")(concat)
        for index in range(1, len(units)):
            x = Dense(units[index], name="Hidden{}".format(index), activation="relu")(x)

        output = Dense(1, name="Out_QVal")(x)
        model = Model(inputs=inputs, outputs=output)

        return model

    def target_update(self):
        new_weight = (1-self.tau)*np.array(self.Q_target.get_weights()) + self.tau*np.array(self.Q.get_weights())
        self.Q_target.set_weights(new_weight)