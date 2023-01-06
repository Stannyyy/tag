# Import packages
import random
import numpy as np
import pandas as pd
from moderator import Moderator
from arena import Arena
from config import retrieve_config

# Extract all variables from the config file
config = retrieve_config()

# Start a competition and m
def competition(game,players):
    # Now loop over all combinations
    round_x_list = list(range(len(players)))
    round_y_list = list(range(len(players)))
    random.shuffle(round_x_list)
    random.shuffle(round_y_list)
    for round_x in round_x_list:
        for round_y in round_y_list:

            if round_x > round_y:

                # Announce game
                print('===')
                print('Players',round_x,'and',round_y,'are playing a game')

                # Set up moderator and arena
                modertr = Moderator(game,players=[players[round_x],players[round_y]])
                arena = Arena(modertr)

                # Compete in arena
                arena.play_and_learn()

                # Extract players
                players[round_x] = arena.modertr._players[0]
                players[round_y] = arena.modertr._players[1]

    # Rank and select top 2
    ranked_players = [round(np.mean(p._model._losses[-config['PRINT_EVERY']:]),2) for p in players]

    # Create pandas to order
    ranked_players = pd.DataFrame({'Players':range(len(players)),'Losses':ranked_players}).sort_values('Losses')
    winning_players = list(ranked_players.Players[0:2].values)

    return players, winning_players