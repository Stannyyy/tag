# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
import math
import time
from model import Model


# Player
class Player(Model):

    def __init__(self, name, render=False, just_like = None):

        # Import model
        Model.__init__(self)

        # Identifying variables
        self._name = name if just_like is None else just_like._name + name

        # Model variables
        self._eps = self.maxEpsilon if just_like is None else just_like._eps

        # Experience variables (carrying over using just_like)
        self._steps = 0 if just_like is None else just_like._steps
        self._samples = [] if just_like is None else just_like._samples.copy()

        # Render variables
        self._render = render

        # Sample variables
        self._state = np.array([])
        self._sample = []

        # Collection variables
        self._reward_store = []
        self._timeline = []

        # State variables
        self._reward = 0
        self._tot_reward = 0

    def choose_action(self, options):

        # Use chance to see whether to explore or exploit
        chance_value = random.random()
        if chance_value < self._eps:
            choice = random.sample(options, k=1)[0]
        else:
            prediction = self.predict_one(self._state)
            prediction = [p if i in options else -np.inf for i, p in enumerate(prediction)]
            choice = np.argmax(prediction)

        return choice

    def choose_action_video(self):
        x

    def set_state(self, state):
        self._state = state
    
    def add_sample(self):
        self._samples += [self._sample]
        if len(self._samples) > self.maxMemory:
            self._samples = self._samples[-self.maxMemory:]
        self._sample = []

    def learn_by_replay(self):

        # Make random batch
        batch = random.sample(self._samples, k=self.batchSize)

        # Predict Q(s,a) given the batch of states
        states = np.array([val[0] for val in batch])
        q_s_a1, q_s_a2 = self._model.predict_batch(states)

        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        next_states = np.array([(np.zeros(self.numStates) if val[3] is None else val[3]) for val in batch])
        q_s_a_d1, q_s_a_d2 = self._model.predict_batch(next_states)

        # Set up training arrays
        x = np.zeros((len(batch), self.numStates))
        y1 = np.zeros((len(batch), self.numActions))
        y2 = np.zeros((len(batch), self.numActions))

        # Now loop over batch
        for i, b in enumerate(batch):

            # Extract sample
            state, action, reward, next_state, options = b[0], b[1], b[2], b[3], b[4]

            # Get the corrected q values for all actions in state
            corrected_q1 = q_s_a1[i]
            corrected_q2 = q_s_a2[i]

            # Update the q value for action
            if next_state is None:
                # In this case, the game completed after action, so there is no max Q(s',a') prediction possible
                corrected_q1[action] = reward
                corrected_q2[action] = reward
            else:
                # Clipped Double Q-learning
                prediction_next_state = np.amin([np.amax(q_s_a_d1[i][options]), np.amax(q_s_a_d2[i][options])])

                corrected_q1[action] = reward + self.discountFactor * prediction_next_state
                corrected_q2[action] = reward + self.discountFactor * prediction_next_state

            x[i] = state
            y1[i] = corrected_q1
            y2[i] = corrected_q2

        self._model.train_batch(x, y1, y2)
        self.update_epsilon()
        
    def update_epsilon(self):
        self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._lambda * self._steps)
        self._steps += 1

    def new_game(self):
        self._tot_reward = 0
