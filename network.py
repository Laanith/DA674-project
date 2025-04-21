# coding:utf-8
# network.py

import numpy as np
import tensorflow as tf
from constant import FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH, ENTROPY_BETA


class A3CFF(tf.keras.Model):
    def __init__(self, num_actions):
        super(A3CFF, self).__init__(name="a3cff")
        self.num_actions = num_actions

        # Convolution layers
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=8, strides=4, activation='relu', padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')

        # Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(num_actions)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """ Forward pass """
        x = tf.cast(inputs, tf.float32) / 255.0  # Normalization like in Atari DeepMind agents
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)

        policy = tf.nn.softmax(self.policy_logits(x))
        value = self.value(x)

        return policy, tf.squeeze(value)

    def compute_loss(self, states, actions, discounted_rewards):
        """
        Compute the A3C loss based on policy and value predictions
        
        Args:
            states: Input states
            actions: Actions taken
            discounted_rewards: Target rewards (with discount factor applied)
            
        Returns:
            total_loss: Combined policy and value loss
        """
        policies, values = self(states)
        
        # Convert actions to one-hot encoding
        actions_one_hot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
        
        # Calculate responsible probabilities (probabilities of taken actions)
        responsible_probs = tf.reduce_sum(policies * actions_one_hot, axis=1)
        
        # Log probability and entropy with numerical stability
        log_prob = tf.math.log(tf.clip_by_value(responsible_probs, 1e-10, 1.0))
        entropy = -tf.reduce_sum(policies * tf.math.log(tf.clip_by_value(policies, 1e-10, 1.0)), axis=1)
        
        # Calculate advantage
        advantage = discounted_rewards - values
        
        # Calculate losses
        policy_loss = -log_prob * tf.stop_gradient(advantage)  # Stop gradient for stable training
        entropy_loss = -ENTROPY_BETA * entropy
        value_loss = 0.5 * tf.square(advantage)
        
        # Combine all losses
        total_loss = tf.reduce_mean(policy_loss + entropy_loss + value_loss)
        
        return total_loss