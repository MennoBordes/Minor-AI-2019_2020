import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from collections import deque
import argparse
from time import sleep

import numpy as np
import random
import os
import gym
import gym_xplane

REPLAY_MEMORY_SIZE = 50000

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()


class GlobalAI(object):
    """
        The global AI which is supposed to choose which of the sub-ai's
        to use at which point in the flight.
    """
    def __init__(self):
        pass


class AITakeoff(object):
    """
        The AI which is solely focused on taking off from the runway.
        Through the help of waypoints
    """
    def __init__(self, environment=gym.make('xplane-gym-v0'), learning_rate=0.01,
                 discount=0.95, exploration_rate=1.0):

        # Main model
        self.model = self.create_model(lr=learning_rate)

        # Target model
        self.target_model = self.create_model(lr=learning_rate)
        self.target_model.set_weights(self.model.get_weights())

        # An array with the last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate

        self.input_count = environment.observation_space
        self.output_count = environment.action_space

        self.session = tf.Session()

        self.session.run(self.initializer)

    def create_model(self, lr):
        model = Sequential()

        model.add(Dense(10, input_shape=len(env.observation_space), activation='relu'))
        model.add(Dense(len(env.action_space), activation='sigmoid'))
        model.compile(SGD(lr=lr), "binary_crossentropy", metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        temp = self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        return temp

    def get_next_action(self, state):
        if random.Random() > self.exploration_rate:
            return self.greedy_action(state)
        else:
            return self.random_action()


class AIFlight(object):
    """
        The AI which is solely focused on navigating from point A to point B.
        Through the help of waypoints
    """
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.observation_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class AILanding(object):
    """
        The AI which is solely focused on landing the plane on the runway.
        Through the help of waypoints
    """
    def __init__(self):
        pass


if __name__ == '__main__':
    # SETUP ENVIRONMENT
    parser = argparse.ArgumentParser()
    parser.add_argument('--clientAddr', help='client host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='xplane host address', default='127.0.0.1')
    parser.add_argument('--xpPort', help='xplane port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=1)
    args = parser.parse_args()

    env = gym.make('xplane-gym-v0')
    env.reset()
    # Create waypoints to target
    env.remove_waypoints()
    # env.add_waypoints('Routes/EHAM-LEVC_amsterdam-valencia.json')
    # env.add_waypoints('Routes/EHAM-LZIB_amsterdam-bratislava.json')
    env.add_waypoints('Routes/flight_straight_1.json', land_start=False)

    # SEED environment
    env.action_space.seed(0)

    agent = RandomAgent(env.action_space)

    EPISODES = 1
    episode = 0

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)

            episode_reward += reward
            # print('reward: {}'.format(episode_reward))

            state = new_state
            # print("State: ", state, "Reward: ", reward)
            # print("Done: ", done, "New state: ", new_state)

            sleep(0.05)
        print('episode_reward: {}'.format(episode_reward))
    env.close()
