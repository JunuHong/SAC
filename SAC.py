import tensorflow as tf
import numpy as np
from collections import deque
import random
from tensorflow.keras.optimizers import Adam
from Actor_Net import Actor
from Critic_Net import Critic

class SAC_agent():
    def __init__(self, state_dim, action_dim, action_max,
                 learning_rate = 0.001, tau = 0.05, gamma = 0.95,
                 update_every = 5, max_memory = 1_000_000, batch_size = 128, 
                 log_std_min = -20, log_std_max = 2):

        tf.keras.backend.set_floatx('float64')

        self.memory = deque(maxlen=max_memory)

        self.memory = deque(maxlen=max_memory)
        self.batch_size = batch_size

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        self.update_counter = 0
        self.update_every = update_every

        # actor-critic nets
        self.actor = Actor(state_dim, action_dim, action_max, 
                           learning_rate, tau, log_std_min, log_std_max)
        
        self.Q1 = Critic(state_dim, action_dim, learning_rate, tau)     
        self.Q2 = Critic(state_dim, action_dim, learning_rate, tau)

        # temperature variable
        self.target_entropy = -np.prod(action_dim)
        self.log_alpha = tf.Variable(0., dtype=tf.float64)
        self.alpha = tf.Variable(0., dtype=tf.float64)
        self.alpha.assign(tf.exp(self.log_alpha))
        self.alpha_optimizer = Adam(learning_rate=learning_rate)

        # Tensorboard
        self.summaries = {}

    def remember(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append([state, action, reward, next_state, done])

    def minibatch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        s = np.asarray(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]

        with tf.GradientTape(persistent=True) as tape:
            # next state action log probs
            means, log_stds = self.actor.actor(next_states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            next_actions, log_probs = self.actor.process_actions(means, log_stds)

            # Q loss
            current_q_1 = self.Q1.Q([states, actions])
            current_q_2 = self.Q2.Q([states, actions])
            next_q_1 = self.Q1.Q_target([next_states, next_actions])
            next_q_2 = self.Q2.Q_target([next_states, next_actions])
            next_q_min = tf.math.minimum(next_q_1, next_q_2)
            state_values = next_q_min - self.alpha * log_probs
            target_qs = tf.stop_gradient(rewards + state_values * self.gamma * (1. - dones))
            Q1_loss = tf.reduce_mean(0.5 * tf.math.square(current_q_1 - target_qs))
            Q2_loss = tf.reduce_mean(0.5 * tf.math.square(current_q_2 - target_qs))

            # current state action log probs
            means, log_stds = self.actor.actor(states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            actions, log_probs = self.actor.process_actions(means, log_stds)

            # actor loss
            current_q_1 = self.Q1.Q([states, actions])
            current_q_2 = self.Q2.Q([states, actions])
            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            actor_loss = tf.reduce_mean(self.alpha*log_probs - current_q_min)

            # temperature loss
            alpha_loss = -tf.reduce_mean((self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)))

        critic_grad = tape.gradient(Q1_loss, self.Q1.Q.trainable_variables)  # compute actor gradient
        self.Q1.optimizer.apply_gradients(zip(critic_grad, self.Q1.Q.trainable_variables))

        critic_grad = tape.gradient(Q2_loss, self.Q2.Q.trainable_variables)  # compute actor gradient
        self.Q2.optimizer.apply_gradients(zip(critic_grad, self.Q2.Q.trainable_variables))

        actor_grad = tape.gradient(actor_loss, self.actor.actor.trainable_variables)  # compute actor gradient
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.actor.trainable_variables))
        
        self.update_counter += 1

        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        self.alpha.assign(tf.exp(self.log_alpha))

        if self.update_counter == self.update_every:
            self.Q1.target_update()
            self.Q2.target_update()
            self.update_counter = 0

        self.summaries['Q1_loss'] = Q1_loss
        self.summaries['Q2_loss'] = Q2_loss
        self.summaries['actor_loss'] = actor_loss
        self.summaries['alpha_loss'] = alpha_loss
        self.summaries['alpha'] = self.alpha
        self.summaries['log_alpha'] = self.log_alpha

        return self.summaries