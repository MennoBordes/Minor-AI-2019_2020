import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from collections import deque
import argparse
from time import sleep
from datetime import datetime
from tqdm import tqdm

import numpy as np
import random
import os
import gym
import gym_xplane

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

REPLAY_MEMORY_SIZE = 50000
LEARNING_RATE = 0.01
MIN_REWARD = -1000000000000

WAYPOINT_FILE = 'Routes/flight_straight_1.json'
WAYPOINT_START_LAND = False

# SETUP ENVIRONMENT
parser = argparse.ArgumentParser()
parser.add_argument('--clientAddr', help='client host address', default='0.0.0.0')
parser.add_argument('--xpHost', help='xplane host address', default='127.0.0.1')
parser.add_argument('--xpPort', help='xplane port', default=49009)
parser.add_argument('--clientPort', help='client port', default=1)
args = parser.parse_args()


env = gym.make('xplane-gym-v0')
# Create waypoints to target
env.remove_waypoints()
# env.add_waypoints('Routes/EHAM-LEVC_amsterdam-valencia.json')
# env.add_waypoints('Routes/EHAM-LZIB_amsterdam-bratislava.json')
env.add_waypoints(WAYPOINT_FILE, land_start=WAYPOINT_START_LAND)

# SEED environment
env.action_space.seed(0)


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()





class AI_Cruise:
    def __init__(self):
        # Main model
        self.NAME = 'AI_CRUISE'
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def create_model(self):
        """
        Creates a model
        :return: The created model
        """
        # model = Sequential()
        # print(env.observation_space)
        # # model.add(Dense(10, input_shape=env.observation_space, activation='relu'))
        #
        # model.add(Dense(10, input_shape=(11,), activation='relu', name='input'))
        # # model.add(Dense(15, activation='sigmoid', name='hidden'))
        # model.add(Dense(7, activation='sigmoid', name='output'))
        # model.compile(SGD(lr=LEARNING_RATE), 'binary_crossentropy', metrics=["accuracy"])
        # # print(model.summary())

        print('*****************************************')
        print('*****************************************')
        # Create input layer
        m_input = tf.keras.layers.Input(shape=(11,), name='input')
        # Create hidden layer
        h_1 = tf.keras.layers.Dense(18, activation='relu', name='hidden')(m_input)
        # Create multiple output layers
        out_1 = tf.keras.layers.Dense(4, activation='sigmoid', name='out_1')(h_1)
        out_2 = tf.keras.layers.Dense(1, activation='softmax', name='out_2')(h_1)
        out_3 = tf.keras.layers.Dense(2, activation='sigmoid', name='out_3')(h_1)
        # Create list of outputs
        outputs = [out_1, out_2, out_3]
        # Create model
        model = tf.keras.Model(inputs=m_input, outputs=outputs, name="AI_CRUISE")
        model.compile(
            optimizer=SGD(lr=LEARNING_RATE),
            loss='binary_crossentropy'
        )
        print(model.summary())

        print('*****************************************')
        print('*****************************************')

        return model

    def update_replay_memory(self, transition):
        """
        Adds the step data to a memory replay array
        :param transition: (observation_space, action, reward, new_observation_spoce, done)
        """
        self.replay_memory.append(transition)

    # def train(self, terminal_state, step):
    #     # Start training only if certain number of samples is already saved
    #     if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
    #         return
    #
    #     # Get a minibatch of random samples from memory replay table
    #     minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    #
    #     # Get current states from minibatch, then query NN model for Q values
    #     current_states = np.array([transition[0] for transition in minibatch])/255
    #     current_qs_list = self.model.predict(current_states)
    #
    #     # Get future states from minibatch, then query NN model for Q values
    #     # When using target network, query it, otherwise main network should be queried
    #     new_current_states = np.array([transition[3] for transition in minibatch])/255
    #     future_qs_list = self.target_model.predict(new_current_states)
    #
    #     X = []
    #     Y = []
    #
    #     # Enumerate batches
    #     for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
    #         if not done:
    #             max_future_q = np.max(future_qs_list[index])
    #             new_q = reward + DISCOUNT * max_future_q
    #         else:
    #             new_q = reward
    #
    #         current_qs = current_qs_list[index]
    #         current_qs[action] = new_q
    #
    #         X.append(current_state)
    #         Y.append(current_qs)
    #
    #     self.model.fit(np.array(X))

    def get_qs(self, state):
        """
        Get Q values given current observation space (environment state)
        :param state: The current observation state
        :return: The predicted actions
        """

        state_array = np.array(state)
        # print('state_array: {}'.format(state_array))
        steering, gear, flaps = self.model.predict(state_array.reshape(-1, *state_array.shape))

        # Convert landing gear from float to int
        gear = gear.astype('int')
        # print('steering: {}'.format(steering))
        # print('gear: {}'.format(gear))
        # print('flaps: {}'.format(flaps))

        m_list = []
        for l in steering[0]:
            m_list.append(l)

        for l in gear[0]:
            m_list.append(l)

        for l in flaps[0]:
            m_list.append(l)

        return m_list


# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

agent = AI_Cruise()

agent2 = RandomAgent(env.action_space)

ep_rewards = [-2000]


AGGREGATE_STATS_EVERY = 2

EPISODES = 5
episode = 0

# while episode < EPISODES:
#     try:
# for episode in range(EPISODES):
# Using tqdm for tracking info about the episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Restarting episode
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    env.remove_waypoints()
    env.add_waypoints(WAYPOINT_FILE, WAYPOINT_START_LAND)


    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # action = agent2.act(current_state)
        # print('random: {}'.format(action))

        if np.random.random() > epsilon:
            # Get predicted action
            action = agent.get_qs(current_state)
        else:
            action = env.action_space.sample()
            print(action)
            # print('else')

        # action = agent.get_qs(current_state)
        # print('prediction: {}'.format(action))


        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        current_state = new_state

        sleep(0.05)
    print('episode_reward: {}'.format(episode_reward))

    ep_rewards.append(episode_reward)

    # Add episode reward to a list and log the stats (only for every n episodes)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        # Save model, but only when min reward is greater or equal to set value
        if min_reward >= MIN_REWARD:
            now = datetime.now()
            now_format = now.strftime("%Y-%m-%dT%H-%M-%S")
            agent.model.save(f'train_models/{agent.NAME}__{now_format}__{episode}episode__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    # except Exception as e:
    #     print("Error: {} \nErrorValue: {}".format(e.__class__, str(e)))

env.close()
