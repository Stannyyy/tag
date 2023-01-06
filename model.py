# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""

# Import packages
import tensorflow.keras as tf
from config import retrieve_config
import numpy as np

# Extract all variables from the config file
config = retrieve_config()

# Model game
class Model:
    def __init__(self, num_states, num_actions):
        self._num_states  = num_states
        self._num_actions = num_actions
        self._batch_size  = config['BATCH_SIZE']
        # define the placeholders
        self._states    = None
        self._actions   = None
        self._model     = None
        # the output operations
        self._logits    = None
        self._loss      = None
        self._optimizer = None
        self._var_init  = None
        self._alpha     = config['ALPHA']
        # losses
        self._losses = []
        # now setup the model
        self.define_model()

    def define_model(self):
        # Define model
        model = tf.models.Sequential()
        model.add(tf.layers.Dense(100, activation='relu', input_shape=[self._num_states]))        
        model.add(tf.layers.Dense(self._num_actions,activation='linear'))
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self._alpha))
        self._model = model
        
    def predict_one(self, state, verbose=0):
        prediction = self._model.predict(state.reshape(1, self._num_states), verbose = 0)[0]
        if verbose:
            prediction = [round(p,2) for p in prediction]
            print(prediction)
        return prediction
    
    def predict_batch(self, states):
        return self._model.predict(states, verbose=0)
    
    def train_batch(self, x_batch, y_batch):
        log = self._model.fit(x_batch, y_batch, verbose=0)
        self._losses += log.history.get('loss')

    def mutate(self):
        copy = tf.models.clone_model(self._model)
        new_layers = []
        for l in copy.get_weights():
            if len(l.shape) == 1:
                random_mutation_probs = np.random.rand(l.shape[0])
            else:
                random_mutation_probs = np.random.rand(l.shape[0],l.shape[1])
            random_mutation_probs = np.where(random_mutation_probs < config['CHANCE_OF_MUTATION'], np.random.rand(),0)
            print(random_mutation_probs)
            new_layer = l + random_mutation_probs
            new_layers += [new_layer]
        copy.set_weights(new_layers)

        return copy

