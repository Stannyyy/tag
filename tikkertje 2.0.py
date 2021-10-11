# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:07:36 2021

@author: StannyGoffin
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:53:21 2020

@author: StannyGoffin
"""

# SET VARIABLES
# Reinforcement learning model variables
MAX_EPSILON  = 1
MIN_EPSILON  = 0
LAMBDA       = 0.001
GAMMA        = 0.99
BATCH_SIZE   = 50
MAX_MEMORY   = 5000

# Game variables
GRID_SIZE    = 3

# Player variables
NUM_PLAYERS  = 2
IS_RANDOM    = [False,False] # List of len(NUM_PLAYERS) saying which are random, like [False, True, False, False]
NUM_TAGGERS  = 1

# Do you want to render the game (I advice to do this only after the model had time to form)
RENDER       = False
RENDER_SPEED = 0.1# Prints move every ~ seconds

# Experiment variables
NUM_EPISODES = 10001
PRINT_EVERY  = 1000 # Shows a plot of rewards per player every ~ episodes


# Import packages
import numpy as np
import pandas as pd
import random 
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior() 
tf.get_logger().setLevel('INFO')
import rendering
import scipy
import matplotlib.pyplot as plt
import time

# Game
class Tag:
    def __init__(self, GRID_SIZE, NUM_PLAYERS, NUM_TAGGERS):
        self._grid_size = GRID_SIZE
        self._num_play  = NUM_PLAYERS
        self._x_list    = [-1] * NUM_PLAYERS
        self._y_list    = [-1] * NUM_PLAYERS
        self._p_list    = list(range(NUM_PLAYERS))
        self._taggers   = [True] * NUM_TAGGERS + [False] * (NUM_PLAYERS - NUM_TAGGERS)
        self._i_list    = random.sample(range(NUM_PLAYERS),k=NUM_PLAYERS)
        self._turn      = self._i_list[0]
        self._options   = [0,1,2,3,4,5,6,7] # ['up','down','left','right','upleft','upright','downleft','downright']
        self._df        = pd.DataFrame(-1, columns=range(self._grid_size), index=range(self._grid_size))
        self._viewer    = None
        self.random_game()
    
    def random_game(self):
        self._x_list   = [-1] * self._num_play
        self._y_list   = [-1] * self._num_play
        self._df       = pd.DataFrame(-1, columns=range(self._grid_size), index=range(self._grid_size))
        
        for i in self._i_list:
            while True:
                x = int(np.floor(random.random()*(self._grid_size)))
                y = int(np.floor(random.random()*(self._grid_size)))
                check = [True for i in self._i_list if (self._x_list[i] == x) & (self._y_list[i] == y)]
                if len(check) == 0:
                    self._x_list[i] = x
                    self._y_list[i] = y
                    break
            self._df.loc[y,x] = self._p_list[i]
        self.what_options()
        
    # Print game
    def print_game(self):
        df = self._df.replace(-1,'')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)

    def refresh_df(self):
        self._df = pd.DataFrame(-1, columns=range(self._grid_size), index=range(self._grid_size))
        for i in self._i_list:
            self._df.loc[self._y_list[i],self._x_list[i]] = self._p_list[i]
    
    # Move options
    def what_options(self):
        x = self._x_list[self._turn]
        y = self._y_list[self._turn]
        self._options = [0,1,2,3,4,5,6,7] # ['up','down','left','right','upleft','upright','downleft','downright']
        if y == 0:
            self._options[0] = -1
            self._options[4] = -1
            self._options[5] = -1
        if y == self._grid_size - 1:
            self._options[1] = -1
            self._options[6] = -1
            self._options[7] = -1
        if x == 0:
            self._options[2] = -1
            self._options[4] = -1
            self._options[6] = -1
        if x == self._grid_size - 1:
            self._options[3] = -1
            self._options[5] = -1
            self._options[7] = -1
        
        self._options = [o for o in self._options if o != -1]
        
    def future_options(self,choice):
        x = self._x_list[self._turn]
        y = self._y_list[self._turn]
        options = [0,1,2,3,4,5,6,7] # ['up','down','left','right','upleft','upright','downleft','downright']
        
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
            
        if y == 0:
            options[0] = -1
            options[4] = -1
            options[5] = -1
        if y == self._grid_size - 1:
            options[1] = -1
            options[6] = -1
            options[7] = -1
        if x == 0:
            options[2] = -1
            options[4] = -1
            options[6] = -1
        if x == self._grid_size - 1:
            options[3] = -1
            options[5] = -1
            options[7] = -1
            
        options = [o for o in options if o != -1]
        return options
    
    def random_move(self):
        choice = random.sample(self._options,k=1)[0]
        
        reward = self.move(choice)
        return choice, reward
    
    def move(self, choice):    
        x = self._x_list[self._turn]
        y = self._y_list[self._turn]
        if choice == 0: # ['up','down','left','right','upleft','upright','downleft','downright']
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
        
        if (x > (self._grid_size-1)) | (y > (self._grid_size-1)) | (x < 0) | (y < 0):
            print('Position out of bounds after move')
            print('X after move '+str(x))
            print('Y after move '+str(y))
            print('Choice: '+str(['up','down','left','right','upleft','upright','downleft','downright'][choice]))
            print('Options: '+str(self._options))
            raise Exception
            
        self._y_list[self._turn]   = y
        self._x_list[self._turn]   = x
        self.refresh_df()
        
        reward = self.what_reward()
        return reward
    
    def what_reward(self):
        is_tagger    = self._taggers[self._turn]
        x            = self._x_list[self._turn]
        y            = self._y_list[self._turn]
        in_same_spot = [i for i in self._i_list if (self._x_list[i] == x) & (self._y_list[i] == y) & (i != self._turn)] 
        
        if is_tagger:
            reward = self._grid_size * - 0.1
        else:
            reward = self._grid_size * 0.1
        
        for caught in in_same_spot:
            caught_is_tagger = self._taggers[caught]
            
            if is_tagger != caught_is_tagger:
                if is_tagger:
                    reward = self._grid_size * 100
                else:
                    reward = self._grid_size * -100

        return reward
    
    def next_turn(self):
        # Move to next turn
        where_now      = self._i_list.index(self._turn)
        if where_now  == len(self._i_list)-1:
            where_now  = -1
        self._turn     = self._i_list[where_now+1]
        
        # Calculate options
        self.what_options()

    
    # Render GUI
    def render(self,v_func=[1,1,1,1]):
        screen_size  = 400
        grid_size    = screen_size / (self._grid_size+2)
    
        if self._viewer is None:
            self._viewer = rendering.Viewer(screen_size, screen_size)
        
        # self._viewer.window.clear()
        self._viewer.geoms = []
                
        for l in range(len(self._x_list)):

            x = self._x_list[l]
            y = self._y_list[l]
            t = self._taggers[l]
                    
            if t == 1:
                team1 = rendering.make_circle(10)
                team1.add_attr(rendering.Transform(translation=(grid_size * (x+1)+l*3, screen_size - grid_size * (y+1)+l*3)))
                team1.set_color(1, 1, 0)
                self._viewer.add_geom(team1)
            elif t == 0:
                team0 = rendering.make_circle(10)
                team0.add_attr(rendering.Transform(translation=(grid_size * (x+1)+l*3, screen_size - grid_size * (y+1)+l*3)))
                team0.set_color(0, 0, 1)
                self._viewer.add_geom(team0)

        return self._viewer.render(return_rgb_array='human' == 'rgb_array')
    
    # Shut down GUI
    def shut_down_GUI(self):
        self._viewer.close()


# Model game
class Model:
    def __init__(self, num_states, num_actions):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = BATCH_SIZE
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._loss = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()
        # losses
        self._losses = []

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        self._loss = tf.losses.huber_loss(self._q_s_a, self._logits)
        self._optimizer = tf.train.GradientDescentOptimizer(LAMBDA).minimize(self._loss)
        self._var_init = tf.global_variables_initializer()
        
    def predict_one(self, state, sess):
        prediction = sess.run(self._logits, feed_dict={self._states:state.reshape(1, self._num_states)})
        return prediction
    
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})
    
    def train_batch(self, sess, x_batch, y_batch):
        [_,loss]=sess.run([self._optimizer,self._loss], feed_dict={self._states: x_batch, self._q_s_a: y_batch})
        self._losses += [round(loss,1)]


# Play
class Player:
    def __init__(self, sess, model, env, max_memory, max_eps, min_eps,
                 decay, render=True):
        self._sess         = sess
        self._env          = env
        self._model        = model
        self._render       = render
        self._max_eps      = max_eps
        self._min_eps      = min_eps
        self._decay        = decay
        self._eps          = self._max_eps
        self._steps        = 0
        self._reward_store = []
        self._turn_store   = []
        self._diff_memory  = []
        self._max_memory   = max_memory
        self._samples      = []
        self._tot_reward   = 0
        self._random       = False
        self._last_env_now = env
        self._last_move    = -1
        self._last_reward  = 0
        self._last_env_aft = env
        self._last_options = []
        self._test_reward_list = []
        self._test_reward  = 0
        
    def choose_action(self,game,):
        if random.random() < self._eps:
            choice, reward = game.random_move()

        else:
            prediction = self._model.predict_one(self._env,self._sess)            
            for i in [0,1,2,3,4,5,6,7]:
                if i not in game._options:
                    prediction[0,i] = -np.inf
            choice  = np.argmax(prediction)
            reward  = game.move(choice)
            
        return choice, reward

    def define_v_function(self,game):
        prediction = self._model.predict_one(self._env,self._sess)    
        return(prediction)
        
    def learn_by_replay(self):
        batch = self.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch]) 
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state, options = b[0], b[1], b[2], b[3], b[4]
            
            # get the current q values for all actions in state
            current_q = q_s_a[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i][options])
                
            x[i] = state
            y[i] = current_q
            
        self._model.train_batch(self._sess, x, y)

    def add_sample(self, sample):
        self._samples.append(sample)
        self._last_env_now = sample[0]
        self._last_move    = sample[1]
        self._last_reward  = sample[2]
        self._last_env_aft = sample[3]
        self._last_options = sample[4]
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
        
# Play
class Random_Player:
    def __init__(self, sess,env, render=True):
        self._sess         = sess
        self._env          = env
        self._steps        = 0
        self._reward_store = []
        self._turn_store   = []
        self._tot_reward   = 0
        self._random       = True
    
    def choose_action(self,game):
        choice, reward = game.random_move()
        self._tot_reward  += reward
        return choice, reward
        

# Moderate game
class Moderator:
    def __init__(self, game, players):
        self._game         = game
        self._players      = players

    def run(self,render):
        turn       = 0 # For analysis
        game_ended = False
        
        # New game
        self._game.random_game()
        
        while game_ended == False:

            # Reset order of turns every round of turns
            if (turn !=0) & (turn % NUM_PLAYERS == 0):
                self._game._i_list = random.sample(range(NUM_PLAYERS),k=NUM_PLAYERS)
                self._game._turn   = self._game._i_list[0]
                self._game.what_options()
                
            # Make numpy array of current environment
            env_now     = np.array(self._game._x_list + self._game._y_list + [self._game._taggers[self._game._turn]])
            self._players[self._game._turn]._env = env_now
            
            # Act and add sample if non-random player, act if random player
            if self._players[self._game._turn]._random == False:
                
                # Act
                move, reward = self._players[self._game._turn].choose_action(self._game)
                
                # If game ended, next environment is None 
                if (np.abs(reward)>self._game._grid_size):
                    env_aft_move = None
                    game_ended = True
                else:
                    env_aft_move = np.array(self._game._x_list + self._game._y_list + [self._game._taggers[self._game._turn]])
                options      = self._game.future_options(move)
                
                # Normalize environment
                x_list = [round(x / (self._game._grid_size-1),1) for x in self._game._x_list]
                y_list = [round(y / (self._game._grid_size-1),1) for y in self._game._y_list]
                env_now = np.array(x_list + y_list + [self._game._taggers[self._game._turn]])
                
                # Add sample
                self._players[self._game._turn]._env = env_now
                self._players[self._game._turn].add_sample((env_now, move, reward, env_aft_move, options))
                self._players[self._game._turn]._tot_reward += reward
                
            else:
                # Act
                move, reward = self._players[self._game._turn].choose_action(self._game)
            
            # Move the agent to the next state
            if render:
                self._game.render()
                time.sleep(RENDER_SPEED)

            # If the game is done, let all players add sample and learn
            if game_ended:
                for i in range(len(self._players)):

                    # Assign rewards to everybody and learn!
                    if (self._players[i]._random == False):
                        
                        # Don't have to add sample when its players turn
                        if i == self._game._turn:
                            reward_now = 0
                        else:
                            # If player in the other team, reward is mirrored
                            if self._game._taggers[i] != self._game._taggers[self._game._turn]:
                                reward_now = reward * -1
                            else:
                                reward_now = reward
                    
                        # Update some parameters
                        self._players[i]._env = env_now
                        self._players[i]._tot_reward += reward
                        
                        # Update rewards for the other players than the one who's turn turn it was
                        if reward_now != 0:
                            self._players[i].add_sample((self._players[i]._last_env_now, self._players[i]._last_move, reward_now, self._players[i]._last_env_aft, self._players[i]._last_options))
                        
                        # Learn!
                        self._players[i].learn_by_replay()

                        # Set new epsilon
                        self._players[i]._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._players[i]._steps)
                        self._players[i]._steps += 1
                    
                    # Reset player 
                    self._players[i]._reward_store.append(np.float(self._players[i]._tot_reward))
                    self._players[i]._turn_store.append(turn)
                    self._players[i]._tot_reward = 0
                    
            # Next turn, but first three turns are for runner
            if turn >= 3:
                self._game.next_turn()
            else:
                self._game.what_options()
            turn = turn + 1 # For analysis
        
        # Now other team is the taggers
        self._game._taggers = [t == False for t in self._game._taggers]
        
        # Reset order of turn and make sure the runners get to go first
        self._game._i_list = random.sample(range(NUM_PLAYERS),k=NUM_PLAYERS)
        self._game._turn   = [p for i,p in enumerate(self._game._i_list) if self._game._taggers[i] == False][0]
        self._game.what_options()
    
        
        
# GET ALGORITHM GOING!
# Set up model
num_states  = 2 * NUM_PLAYERS + 1
num_actions = 8

# Set up game
game = Tag(GRID_SIZE, NUM_PLAYERS, NUM_TAGGERS)

# Set up players
if type(IS_RANDOM) == bool:
    if IS_RANDOM:
        IS_RANDOM = [True]*NUM_PLAYERS
    else:
        IS_RANDOM = [False]*NUM_PLAYERS

players = []
for player in IS_RANDOM:
    
    # New model per player
    model = Model(num_states, num_actions)

    # Set up session
    sess = tf.Session()
    sess.run(model._var_init)
    if player: # True means random player
        players += [Random_Player(sess, np.array(game._x_list + game._y_list))]
    else: # False means 
        players += [Player(sess, model, np.array(game._x_list + game._y_list), 
                           MAX_MEMORY, MAX_EPSILON, MIN_EPSILON,LAMBDA)]

# Set up moderator
modertr = Moderator(game,players)

# # Helps when debugging
# self    = modertr 

# Play all the episodes by modertr.run() - Show every PRINT_EVERY episodes
# what the total reward per player is over time
cnt = 0

# You can always stop by Ctrl + c and run manually from here
# try:
while cnt < NUM_EPISODES:
    # print(cnt)
    if (cnt % PRINT_EVERY == 0)&(cnt!=0):
        # Test
        plyrs = []
        for plyr, rndm in enumerate(IS_RANDOM):
            # If another PRINT_EVERY episodes are played, show graph
            plt.plot(scipy.ndimage.filters.gaussian_filter1d(modertr._players[plyr]._model._losses, len(modertr._players[plyr]._model._losses) / 40))
            plyrs += ['Player '+str(plyr+1)]
            
        plt.xlabel("Test runs")
        plt.ylabel("Loss")
        plt.xlim([0, len(modertr._players[plyr]._model._losses)])
        plt.legend(plyrs)
        time_now = '-'.join([('{0:0'+str(max(len(str(t)),2))+'d}').format(t) for i, t in enumerate(time.localtime()[0:5])])
        plt.savefig("Plot of tag game played "+time_now)
        plt.show()
        plt.close("all")
    
    # Play episode!
    modertr.run(RENDER)

    cnt += 1
# except Exception as e:
#     if RENDER:
#         game.shut_down_GUI()
#     print(e)

# See how agents are behaving - Run manually
NUM_EPISODES_SHOW = 100
for i in range(NUM_EPISODES_SHOW):
    modertr.run(render=True)
# game.shut_down_GUI()

# Remove all plot imgs
[os.remove(file) for file in os.listdir(os.getcwd()) if file.endswith('.png')]
