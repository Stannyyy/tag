# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:14:39 2021

@author: StannyGoffin
"""

# Import packages
import random 
import time
import numpy as np
from config import Config
from game import Game

# Moderate game
class Moderator(Config):

    def __init__(self, players):

        # Import config
        Config.__init__(self)

        # Moderator variables
        self._players = players
        self._order_turns = random.sample(range(self.numPlayers), k=self.numPlayers)
        self._turn = self._order_turns[0]
        self._turn_count = 0
        self._game_continues = True
        self._game = None  # Only fill this with a game class once you hit play()

    def next_turn(self):
        idx = self._order_turns.index(self._turn)
        idx += 1
        if idx == len(self._order_turns):
            idx = 0
        self._turn = self._order_turns[idx]
        
    def play(self):
        
        # Initialize game
        self._turn_count = 0
        self._game = Game()

        # Render
        self._game.render()
        text = self.write_video_text()
        self._game.save(text, prefix=str(self._turn_count))

        # Reset players
        [p.new_game() for p in self._players]

        # Start game
        while self._game._ended == False:

            # Get options for player
            options = self._game.what_options(self._turn)

            # Align game and player state
            self._players[self._turn].set_state(self._game._x_list + self._game._y_list +
                                                [self._turn, int(self._game._taggers[self._turn])])

            # Append next state and its options to sample (idx 3 and 4 of sample)
            if len(self._players[self._turn]._sample) < 5:
                self._players[self._turn]._sample += [self._players[self._turn]._state,
                                                      options]
            if len(self._players[self._turn]._sample) == 5:
                self._players[self._turn].add_sample()

            # Make a move!
            choice = self._players[self._turn].choose_action(options)
            reward = self._game.move(self._turn, choice)
            self._turn_count += 1

            # # Describe choice
            # self.describe_choice(choice)

            # Render
            self._game.render()

            # Next players turn
            self.next_turn()

            # Append to sample (idx 0, 1, 2 of sample)
            self._players[self._turn]._reward = reward
            self._players[self._turn]._tot_reward += reward
            self._sample = [self._players[self._turn]._state, choice, reward]

            text = self.write_video_text()
            self._game.save(text, prefix=str(self._turn_count+1))

            # Determine if game ended and determine next state
            if self._game._ended | (self._turn_count >= 50):
                for p in range(self.numPlayers):
                    self._players[p]._next_state = None
                self._game._ended = True

        # Learn
        if self._game._ended:

            # Create video
            game_name = ' is playing against '.join([p._name for p in self._players])
            self._game.record(game_name)

            # All players learn!
            for p in range(self.numPlayers):

                # Only start learning once memory has reached batch size
                if len(self._players[p]._samples) > self.batchSize:

                    # Learn!
                    self._players[p].learn_by_replay()

                # Reset player
                self._players[p]._reward_store.append(np.float(self._players[p]._tot_reward))
                self._players[p]._tot_reward = 0

            # Now other team is the taggers
            self._game._taggers = [t == False for t in self._game._taggers]

    def write_video_text(self):
        print('taggers:')
        print(self._game._taggers)
        print([i for i, x in enumerate(self._game._taggers) if x][0])
        text = 'Is tagger info: ' + str([i for i, x in enumerate(self._game._taggers) if x][0]) + '\n' + \
                'Reward player 0: ' + str(self._players[0]._reward) + '\n' + \
                'Reward player 1: ' + str(self._players[1]._reward) + '\n' + \
                'Total reward player 0: ' + str(self._players[0]._tot_reward) + '\n' + \
                'Total reward player 1: ' + str(self._players[1]._tot_reward) + '\n' + \
                'Turns: ' + str(self._turn_count)
        return text

    def describe_choice(self, choice):
        print({False:'o',True:'x'}.get(self._game._taggers[self._turn]))
        print({0:'up', 1:'down', 2:'left', 3:'right',
               4:'up left', 5:'up right', 6:'down left', 7:'down right'}.get(choice))
