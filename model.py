import numpy as np
import tensorflow as tf
from preprocess import *


class PoemGenerator(tf.keras.Model):
    def __init__(self):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.
        """
        super(PoemGenerator, self).__init__()

        # Numerical Hyperparameters
        self.batch_size = 128
        self.learning_rate = 0.001
        self.LSTM_size = 128
        self.hidden_size = 100
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        # self.embedding_layer = tf.keras.layers.Embedding(
        # self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.gru = tf.keras.layers.LSTM(
            self.rnn_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(
            self.hidden_size, activation='relu')
        # self.dense_2 = tf.keras.layers.Dense(
        # self.vocab_size, activation='softmax')

    @tf.function
    def call(self)
