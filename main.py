# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:07:36 2021

@author: StannyGoffin
"""

# SET VARIABLES
# Reinforcement learning model variables
MAX_EPSILON  = 1
MIN_EPSILON  = 0
LAMBDA       = 0.001
ALPHA        = 0.01
GAMMA        = 0.1
BATCH_SIZE   = 100
MAX_MEMORY   = 5000

# Game variables
GRID_SIZE    = 10

# Player variables
NUM_PLAYERS  = 2
IS_RANDOM    = [False,True] # List of len(NUM_PLAYERS) saying which are random
NUM_TAGGERS  = 1

# Do you want to render the game (I advice to do this only after the model had time to form)
RENDER       = False # If True, every PRINT_EVERY episodes, an episode will be rendered
RENDER_SPEED = 0.3     # Players move every ~ seconds in rendering window

# Experiment variables
NUM_EPISODES = 100000 # Max number of episodes to play, if you want to keep playing until MIN_LOSS is reached, make this number high
PRINT_EVERY  = 10  # Shows a plot of rewards per player every ~ episodes
MIN_LOSS     = 0.9 # Stop when loss < MIN_LOSS, if you want to keep playing until NUM_EPISODES, make this number high

# Render episodes at the end of training
N_GAMES_SHOW = 2000 # Render is always True here, if you don't want to render at the end, assign 0

# Import packages
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import scipy
import matplotlib.pyplot as plt
import time

# Import own packages
from tag import Tag
from model import Model
from player import Player
from random_player import Random_Player 
from moderator import Moderator
        
# GET ALGORITHM GOING!
# Set up model
num_states  = 2 * NUM_PLAYERS + 1
num_actions = 8

# Set up game
game = Tag(GRID_SIZE, NUM_PLAYERS, NUM_TAGGERS)

# Set up players
players = []
for player in IS_RANDOM:
    
    # New model per player
    model = Model(num_states, num_actions, BATCH_SIZE, ALPHA)

    # Set up session
    if player: # True means random player
        players += [Random_Player(np.array(game._x_list + game._y_list))]
    else: # False means 
        players += [Player(model, np.array(game._x_list + game._y_list), 
                           MAX_MEMORY, MAX_EPSILON, MIN_EPSILON,
                           GAMMA, LAMBDA)]

# Set up moderator
modertr = Moderator(game,players,BATCH_SIZE,RENDER,RENDER_SPEED)



##################                         
### Experiment ###                         
##################

# Initialize
cnt = 0
stt   = time.time()

# Loop - Ctrl+C to interrupt loop and debug
loss_check = True
while (cnt < NUM_EPISODES) & (loss_check):
   
    if (cnt % PRINT_EVERY == 0)&(cnt!=0)&(cnt!=PRINT_EVERY):
        
        # Test
        plyrs   = []
        av_loss = []
        reached = []
        eps     = []
        for plyr, rndm in enumerate(IS_RANDOM):
            if plyr == False:
                # If another PRINT_EVERY episodes are played, show graph
                plt.plot(scipy.ndimage.filters.gaussian_filter1d(modertr._players[plyr]._model._losses, len(modertr._players[plyr]._model._losses) /10))
                pl_loss = np.array(modertr._players[plyr]._model._losses[-1*PRINT_EVERY:]).mean().round(1)
                av_loss = av_loss + [pl_loss]
                reached += [pl_loss <= MIN_LOSS]
                eps     = eps + [round(modertr._players[plyr]._eps,2)]
                plyrs  += ['Player '+str(plyr+1)]
        
        if all(reached):
            loss_check = False
            for plyr, rndm in enumerate(IS_RANDOM):
                modertr._players[plyr]._eps = 0 # Set amount of random moves to 0 for rendering
            
        plt.xlabel("# Episodes")
        plt.ylabel("Loss")
        plt.legend(plyrs)
        time_now = '-'.join([('{0:0'+str(max(len(str(t)),2))+'d}').format(t) for i, t in enumerate(time.localtime()[0:5])])
        plt.savefig("Plot of tag game played "+time_now)
        plt.show()
        plt.close("all")
        end = time.time()
        print('This round, s elapsed: '+str(round(end-stt))+', av loss: '+str(av_loss)+', eps: '+str(eps))
        stt = time.time()
        
        # Show one episode
        modertr.play(RENDER)
        cnt += 1
    
    # Play episode! & time it
    modertr.play(False)
    cnt += 1

# See how agents are behaving
for i in range(N_GAMES_SHOW):
    modertr.play(True)
if (N_GAMES_SHOW != 0)|(RENDER):
    game.shut_down_GUI()

# Remove all plot imgs
[os.remove(file) for file in os.listdir(os.getcwd()) if file.endswith('.png')]
