# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
import math
from config import retrieve_config

# Extract all variables from the config file
config = retrieve_config()

# Player
class Player:
    def __init__(self, model, env, render=False):
        self._env          = env
        self._model        = model
        self._render       = render
        self._max_eps      = config['MAX_EPSILON']
        self._min_eps      = config['MIN_EPSILON']
        self._eps          = self._max_eps
        self._gamma        = config['GAMMA']
        self._lambda       = config['LAMBDA']
        self._steps        = 0
        self._reward_store = []
        self._turn_store   = []
        self._diff_memory  = []
        self._max_memory   = config['MAX_MEMORY']
        self._samples      = []
        self._tot_reward   = 0
        self._random       = False
        self._last_env_now = env
        self._last_move    = -1
        self._last_reward  = 0
        self._last_env_aft = env
        self._last_options = []
        self._reward       = 0
        self._move         = -1
        self._options      = []
        self._next_state   = []
        self._can_move     = True
        
    def choose_and_do_action(self,game,turn):
        options = game.what_options(turn)
        if random.random() < self._eps:
            choice, reward = game.random_move(turn,options)

        else:
            prediction = self._model.predict_one(self._env, verbose=0)
            for i in [0,1,2,3,4,5,6,7]:
                if i not in options:
                    prediction[i] = -np.inf
            choice  = np.argmax(prediction)
            reward  = game.move(turn,choice)
        
        self._reward     += reward
        self._tot_reward += reward
        return choice, reward
    
    def add_sample(self,sample):
        self._samples += [sample]
        if len(self._samples) > self._max_memory:
            self._samples = self._samples[-self._max_memory:]
    
    def make_batch(self,batch_size):
        batch = random.sample(self._samples,k=batch_size)
        return batch
        
    def learn_by_replay(self,verbose=False):
        batch = self.make_batch(self._model._batch_size)
        states = np.array([val[0] for val in batch]) 
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state, options = b[0], b[1], b[2], b[3], b[4]

            # get the current q values for all actions in state
            current_q = q_s_a[i]
            if next_state is None:
                q_s_a_d[i] = [0]*len(q_s_a_d[i])
            if verbose:
                print('\n')
                print('Current q: '+str([round(q,2) for q in current_q]))
                print('Current q: '+str(round(current_q[action],2)))
                print('Max q: '+str(round(max(current_q),2)))

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i][options])
                
            x[i] = state
            y[i] = current_q
        self._model.train_batch(x, y)
        self.update_epsilon()
        
    def update_epsilon(self):
        self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._lambda * self._steps)
        self._steps += 1