# Import packages
import time
import matplotlib.pyplot as plt
import scipy
import numpy as np
from config import retrieve_config

# Extract all variables from the config file
config = retrieve_config()

# Arena
class Arena:
    def __init__(self, modertr):
        self.cnt = 0
        self.stt = time.time()
        self.loss_check = True
        self.modertr = modertr
        self.NUM_EPISODES = config['NUM_EPISODES']
        self.PRINT_EVERY = config['PRINT_EVERY']
        self.IS_RANDOM = config['IS_RANDOM']
        self.RENDER = config['RENDER']
        self._round = 0

    def play_and_learn(self):

        # Loop for number of episodes
        while self.cnt < self.NUM_EPISODES:

            if (self.cnt % self.PRINT_EVERY == 0) & (self.cnt != 0) & (self.cnt != self.PRINT_EVERY):

                # Test
                av_loss_1 = np.array(self.modertr._players[0]._model._losses[-1 * self.PRINT_EVERY:]).mean().round(1)
                av_loss_2 = np.array(self.modertr._players[1]._model._losses[-1 * self.PRINT_EVERY:]).mean().round(1)
                eps_1 = [round(self.modertr._players[0]._eps, 3)]
                eps_2 = [round(self.modertr._players[1]._eps, 3)]

                # Print learning status
                self.end = time.time()
                print('Round',str(self.cnt), 'out of',self.NUM_EPISODES, round(self.end - self.stt), 'sec collapsed')
                print('Player 1 = av loss: ' + str(av_loss_1) + ', eps: ' + str(eps_1))
                print('Player 2 = av loss: ' + str(av_loss_2) + ', eps: ' + str(eps_2))
                self.stt = time.time()

                # Show one episode
                self.modertr.play(self.RENDER)
                self.cnt += 1

            # Play episode! & time it
            self.modertr.play(False)
            self.cnt += 1

        self._round += 1