# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:07:36 2021

@author: StannyGoffin
"""

# Import packages
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import own packages
from config import retrieve_config
from tag import Tag
from model import Model
from player import Player
from competition import competition

# Extract all variables from the config file
config = retrieve_config()

# GET ALGORITHM GOING!
# Set up model
num_states  = 2 * config['NUM_PLAYERS'] + 1
num_actions = 8

# Set up competition

# Set up game
game = Tag(config['GRID_SIZE'], config['NUM_PLAYERS'], config['NUM_TAGGERS'])

# Create initial 10 players
players = [Player(Model(num_states, num_actions), np.array(game._x_list + game._y_list)),
           Player(Model(num_states, num_actions), np.array(game._x_list + game._y_list)),
           Player(Model(num_states, num_actions), np.array(game._x_list + game._y_list)),
           Player(Model(num_states, num_actions), np.array(game._x_list + game._y_list)),
           Player(Model(num_states, num_actions), np.array(game._x_list + game._y_list))]

# Now train for 50 generations
for generation in range(10):

    # Announce generation
    print('Now training generation',generation)

    # Competition
    players, winning_players = competition(game,players)

    # Mutate
    # Copy number one, three times into the new player list, number two, two times into the new player list
    players = [Player(players[winning_players[0]]._model._model, np.array(game._x_list + game._y_list)),
               Player(players[winning_players[0]]._model.mutate(), np.array(game._x_list + game._y_list)),
               Player(players[winning_players[0]]._model.mutate(), np.array(game._x_list + game._y_list)),
               Player(players[winning_players[1]]._model._model, np.array(game._x_list + game._y_list)),
               Player(players[winning_players[1]]._model.mutate(), np.array(game._x_list + game._y_list))]
