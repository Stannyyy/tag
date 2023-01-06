# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:39 2021

@author: StannyGoffin
"""

# Import packages
import random 
import time
import numpy as np
from config import retrieve_config

# Extract all variables from the config file
config = retrieve_config()

# Moderate game
class Moderator:
    def __init__(self, game, players):
        self._game         = game
        self._players      = players
        self._render       = config['RENDER']
        self._render_speed = config['RENDER_SPEED']
        self._n_players    = len(self._players)
        self._batch_size   = config['BATCH_SIZE']
    
    def next_turn(self,order_turns,turn):
        idx = order_turns.index(turn)
        idx += 1
        if idx == len(order_turns):
            idx = 0
        turn = order_turns[idx]
        return turn
    
    def nonrandom_players(self):
        nonrandom_players_ = []
        for p in range(self._n_players):
            if self._players[p]._random == False:
                nonrandom_players_ += [p]
        return nonrandom_players_
    
    def determine_and_assign_rewards(self):
        reward_taggers = 0
        reward_runners = 0
        for p in range(self._n_players):
            if self._game._taggers[p]:
                reward_taggers += self._players[p]._reward
            else:
                reward_runners += self._players[p]._reward
        for p in range(self._n_players):
            if self._game._taggers[p]:
                self._players[p]._reward = reward_taggers - reward_runners
            else:
                self._players[p]._reward = reward_runners - reward_taggers
        
    def play(self,render):
        
        # Initialize game
        turn       = 0
        game_ended = False
        self._game.random_game()
        
        # New round of turns
        order_turns = random.sample(range(self._n_players),k=self._n_players)
        
        # For convenience, make list of non-random players
        nonrandom_players_ = self.nonrandom_players()
        
        # Start game
        moves = 0
        while game_ended == False:
            
            # New round of moves, set reward to 0
            for p in range(self._n_players):
                self._players[p]._reward = 0
                self._players[p]._can_move = True
            
            # Play turns
            for turn in order_turns:
    
                # Render
                if render:
                    self._game.render()
                    time.sleep(self._render_speed)            
            
                # The state is composed of x coordinates, y coordinates and whether or not the current player is a tagger
                state = np.array(self._game._x_list + self._game._y_list + [self._game._taggers[turn]])
                
                # If the runner has not already been caught
                if game_ended == False:
                    # Make a move!
                    move, reward = self._players[turn].choose_and_do_action(self._game,turn)
                    moves += 1
                    
                    # Save values for upcoming sample
                    self._players[turn]._env    = state
                    self._players[turn]._move   = move
                    self._players[turn]._reward = reward
                else:
                    self._players[turn]._reward = 0
                    self._players[turn]._can_move = False
                
                # Determine next turn
                next_turn_ = self.next_turn(order_turns, turn)

                # Determine if game ended and determine next state
                if (np.abs(reward)>self._game._grid_size)|(moves>20):
                    self._players[turn]._next_state = None
                    game_ended = True
                else:
                    # In the next state, it's the next players turn
                    self._players[turn]._next_state = np.array(self._game._x_list + self._game._y_list + [self._game._taggers[next_turn_]])
                
                # Determine options to check future q value
                if game_ended == False:
                    self._players[turn]._options = self._game.what_options(next_turn_)

            # Add inverted rewards from the other team to each player
            self.determine_and_assign_rewards()

            # Add samples to memory for the players that are non-random
            for p in nonrandom_players_:
                if self._players[p]._can_move:
                    sample = (self._players[p]._env, self._players[p]._move, self._players[p]._reward, self._players[p]._next_state, self._players[p]._options)
                    self._players[p].add_sample(sample)
            
            # Learn             
            if game_ended:
                
                # Render
                if render:
                    self._game.render()
                    time.sleep(self._render_speed)
                
                # All players learn!
                for p in nonrandom_players_: 
                    
                    # Only start learning once memory has reached batch size
                    if len(self._players[p]._samples) > self._batch_size:
                        
                        # Learn!
                        self._players[p].learn_by_replay()
                    
                    # Reset player 
                    self._players[p]._reward_store.append(np.float(self._players[p]._tot_reward))
                    self._players[p]._tot_reward = 0

                # Now other team is the taggers
                self._game._taggers = [t == False for t in self._game._taggers]
        