# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:11:00 2021

@author: StannyGoffin
"""

# Import packages
import random
import numpy as np
from config import Config
from PIL import Image, ImageDraw, ImageFont
import copy
import cv2
import glob
import os

# Game
class Game(Config):
    def __init__(self):

        # Import config
        Config.__init__(self)

        # Game variables
        self._x_list = [-1] * self.numPlayers
        self._y_list = [-1] * self.numPlayers
        self._taggers = [True] * self.numTaggers + [False] * (self.numPlayers - self.numTaggers)
        self._options = [0, 1, 2, 3,  # 0:up, 1:down, 2:left, 3:right,
                         4, 5, 6, 7]  # 4:up left, 5:up right, 6:down left, 7:down right
        self._ended = False

        # Initialize game
        self.init_random_game()

        # Initialize render
        self._rendered = ''
    
    def init_random_game(self):
        self._x_list = [-1] * self.numPlayers
        self._y_list = [-1] * self.numPlayers

        for i in range(self.numPlayers):
            while True:
                x = int(np.floor(random.random()*self.gridSize))
                y = int(np.floor(random.random()*self.gridSize))
                check = [True for i in range(self.numPlayers) if (self._x_list[i] == x) & (self._y_list[i] == y)]
                if len(check) == 0:
                    self._x_list[i] = x
                    self._y_list[i] = y
                    break
        self._ended = False
    
    # Move options        
    def what_options(self, turn):

        # Get x and y position of the player whose turn it is
        x = self._x_list[turn]
        y = self._y_list[turn]

        # For different scenario's, rule out options
        options = copy.deepcopy(self._options)
        if y == 0:
            options[0] = -1
            options[4] = -1
            options[5] = -1
        if y == self.gridSize - 1:
            options[1] = -1
            options[6] = -1
            options[7] = -1
        if x == 0:
            options[2] = -1
            options[4] = -1
            options[6] = -1
        if x == self.gridSize - 1:
            options[3] = -1
            options[5] = -1
            options[7] = -1
            
        options = [o for o in options if o != -1]
        return options
    
    def move(self, turn, choice):    
        x = self._x_list[turn]
        y = self._y_list[turn]
        if choice == 0:
            y = y - 1
        elif choice == 1:
            y = y + 1
        elif choice == 2:
            x = x - 1
        elif choice == 3:
            x = x + 1
        elif choice == 4:
            y = y - 1
            x = x - 1
        elif choice == 5:
            y = y - 1
            x = x + 1
        elif choice == 6:
            y = y + 1
            x = x - 1
        elif choice == 7:
            y = y + 1
            x = x + 1
            
        self._y_list[turn] = y
        self._x_list[turn] = x
        
        reward = int(round(self.what_reward(turn)))
        return reward

    def what_reward(self, turn):

        # Get state of player whose turn it is
        is_tagger = self._taggers[turn]
        print('istagger g117:',is_tagger)
        print('turn 118',turn)
        x = self._x_list[turn]
        y = self._y_list[turn]

        # Check if player is in the same spot as another player
        in_same_spot = [i for i in range(self.numPlayers) if (self._x_list[i] == x) & (self._y_list[i] == y) & (i != turn)]

        # Usually, a tagger gets some punishment for each move, a runner gets some reward for each move
        if is_tagger:
            reward = -0.9 * self.gridSize * 2  # Negative bias
        else:
            reward = 0.9 * self.gridSize

        # When the tagger caught the runner, the tagger gets a large reward and the runner a large punishment
        for caught in in_same_spot:

            caught_is_tagger = self._taggers[caught]
            
            if is_tagger != caught_is_tagger:
                if is_tagger:
                    reward = 11 * self.gridSize
                else:
                    reward = -11 * self.gridSize * 2

                self._ended = True
        print('reward')
        print(reward)

        return reward

    def render(self):

        playing_field = np.full(shape=(self.gridSize, self.gridSize), fill_value=' ')
        for tagger in np.where(self._taggers)[0].tolist():
            x_tagger = self._x_list[tagger]
            y_tagger = self._y_list[tagger]
            playing_field[x_tagger, y_tagger] = 'x'

        for runner in np.where([t == False for t in self._taggers])[0].tolist():
            x_runner = self._x_list[runner]
            y_runner = self._y_list[runner]
            if playing_field[x_runner, y_runner] == 'x':
                playing_field[x_runner, y_runner] = '%'
            else:
                playing_field[x_runner, y_runner] = 'o'

        # print(playing_field.T)
        self._rendered = playing_field.T

    def save(self, text, prefix):
        img = Image.new('RGB', (160, 160), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((1, 1),
               text + '\n ' + str(self._rendered).replace('[', '').replace(']', ''),
               fill=(0, 0, 0),
               font=ImageFont.load_default())
        while len(prefix) < 3:
            prefix = '0'+prefix
        img.save(self.savePath+prefix+".png")

    def record(self,game_name):

        img_array = []
        for filename in glob.glob(self.savePath+'*.png'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
            os.remove(filename)

        out = cv2.VideoWriter(self.savePath+game_name+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

