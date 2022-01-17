# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:13:26 2021

@author: StannyGoffin
"""

# Import packages
import tensorflow.keras as tf

# Model game
class Model:
    def __init__(self, num_states, num_actions, BATCH_SIZE, ALPHA):
        self._num_states  = num_states
        self._num_actions = num_actions
        self._batch_size  = BATCH_SIZE
        # define the placeholders
        self._states    = None
        self._actions   = None
        self._model     = None
        # the output operations
        self._logits    = None
        self._loss      = None
        self._optimizer = None
        self._var_init  = None
        self._alpha     = ALPHA
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
        prediction = self._model.predict(state.reshape(1, self._num_states))[0]
        if verbose:
            prediction = [round(p,2) for p in prediction]
            print(prediction)
        return prediction
    
    def predict_batch(self, states):
        return self._model.predict(states,verbose=0)
    
    def train_batch(self, x_batch, y_batch):
        log = self._model.fit(x_batch, y_batch,verbose=0)
        self._losses += log.history.get('loss')