# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""

# Import packages
import tensorflow.keras as tf
import numpy as np
from config import Config

# Model game
class Model(Config):
    def __init__(self, model1=None, model2=None):

        # Import config
        Config.__init__(self)

        # Define models
        self._model1 = model1 if model1 is not None else None
        self._model2 = model2 if model2 is not None else None

        # Define the placeholders
        self._states = None
        self._actions = None

        # Define the output operations
        self._logits = None
        self._loss = None
        self._optimizer = None
        self._var_init = None

        # Initialize the loss history
        self._losses1 = []
        self._losses2 = []

        # Set up the models
        self._model1 = self.define_model()
        self._model2 = self.define_model()

    def define_model(self):
        model = tf.models.Sequential([
            tf.layers.Dense(50, activation=tf.layers.LeakyReLU(alpha=self.learningRate), input_shape=[self.numStates]),
            tf.layers.Dense(50, activation=tf.layers.LeakyReLU(alpha=self.learningRate)),
            tf.layers.Dense(self.numActions, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self.learningRate))
        return model

    def predict_one(self, state):
        prediction1 = self._model1.predict(state.reshape(1, self.numStates), verbose=0)[0]
        prediction2 = self._model2.predict(state.reshape(1, self.numStates), verbose=0)[0]
        prediction  = [np.min(prediction1[i], prediction2[i]) for i in range(len(prediction1))]
        return prediction

    def predict_batch(self, states):
        return self._model1.predict(states, verbose=0), self._model2.predict(states, verbose=0)

    def train_batch(self, x_batch, y1_batch, y2_batch):
        # Train batch
        log1 = self._model1.fit(x_batch, y1_batch, verbose=0)
        log2 = self._model2.fit(x_batch, y2_batch, verbose=0)

        # Add losses to log
        self._losses1 += log1.history.get('loss')
        self._losses2 += log2.history.get('loss')

    def mutate(self, model):

        # For each layer, copy model weights and mutate with chanceOfMutation
        new_layers = []
        for l in range(len(model.get_weights())):

            # Retrieve weights
            layer = model.get_weights()[l]

            # Mutate weights
            random_mutation_probs = np.random.rand(*layer.shape)
            random_mutation_probs = np.where(random_mutation_probs < self.chanceOfMutation,
                                             (np.random.rand() - 0.5) / 2, 0)
            new_layer = layer + random_mutation_probs
            new_layers += [new_layer]

        # Clone model 1 and set new, mutated weights
        copy = tf.models.clone_model(model)
        copy.set_weights(new_layers)

        return copy

