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
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import random
import os
import gym
import gym_xplane

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 5  # 100
MINIBATCH_SIZE = 1  # 64
UPDATE_TARGET_EVERY = 5
LEARNING_RATE = 0.01
MIN_REWARD = -9000

WAYPOINT_FILE = 'Routes/EHAM_amsterdam_approach.json'
WAYPOINT_START_LAND = False

# SETUP ENVIRONMENT
parser = argparse.ArgumentParser()
parser.add_argument('--clientAddr', help='client host address', default='0.0.0.0')
parser.add_argument('--xpHost', help='xplane host address', default='127.0.0.1')
parser.add_argument('--xpPort', help='xplane port', default=49009)
parser.add_argument('--clientPort', help='client port', default=1)
args = parser.parse_args()

env = gym.make('xplane-gym-v0', clientAddr='0.0.0.0', xpHost='127.0.0.1', xpPort=49009, clientPort=1)
# Create waypoints to target
env.remove_waypoints()
env.add_waypoints(WAYPOINT_FILE, WAYPOINT_START_LAND)

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()

class ModifiedTensorBoard(TensorBoard):
    # Override init
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

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

class AIFlight(object):
    """
        The AI which is solely focused on navigating from point A to point B.
        Through the help of waypoints
    """


class AILanding(object):
    """
        The AI which is solely focused on landing the plane on the runway.
        Through the help of waypoints
    """

    def __init__(self, environment=gym.make('xplane-gym-v0', clientAddr='0.0.0.0', xpHost='192.168.0.1', xpPort=49009, clientPort=1), learning_rate=0.01,
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

    def create_model(self, lr):

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
        model = tf.keras.Model(inputs=m_input, outputs=outputs, name="AI_LANDING")
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

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.get_qs(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255

        future_qs_list = self.get_qs(new_current_states, target_model=True)
        X = []
        y1 = []
        y2 = []
        y3 = []

        # Enumerate batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # if not done:
            #     max_future_q = np.max(future_qs_list[index])
            #     new_q = reward + DISCOUNT * max_future_qð
            #     print('max_fut_q: {} \nnew_q: {}'.format(max_future_q, new_q))
            #     return
            # else:
            #     new_q = reward

            # current_qs = current_qs_list[index]
            # current_qs = (current_qs_list[0][index], current_qs_list[1][index], current_qs_list[2][index])
            # print('current_qs: {} \nnew_q: {} action: {}'.format(current_qs, new_q, action))
            # current_qs[action] = new_q
            # print('current_qs_list: {}'.format(current_qs_list[0]))
            X.append(current_state)
            y1.append(current_qs_list[0][index])
            y2.append(current_qs_list[1][index])
            y3.append(current_qs_list[2][index])

        self.model.fit(
            {"input": np.array(X) / 255},
            {
                "out_1": np.array(y1),
                "out_2": np.array(y2),
                "out_3": np.array(y3)
            },
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False)  # ,
        # callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state, target_model=False):
        """
        Get Q values given current observation space (environment state)
        :param target_model: To be executed on the target model, default = False
        :param state: The current observation state
        :return: The predicted actions
        """

        state_array = np.array(state)
        # print('state_array: {}'.format(state_array))
        steering, gear, flaps = self.model.predict(state_array.reshape(-1, *state_array.shape)) \
            if target_model \
            else self.target_model.predict(state_array.reshape(-1, *state_array.shape))

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

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

agent = AILanding()

agent2 = RandomAgent(env.action_space)

ep_rewards = []

AGGREGATE_STATS_EVERY = 2

EPISODES = 50
episode = 0

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
        if np.random.random() > epsilon:
            # Get predicted action
            # action = agent.get_qs(current_state)
            steering, gear, flaps = agent.model.predict(current_state)
            action = [steering[0], steering[1], steering[2], steering[3], round(gear[0].astype('int')), flaps[0],
                      flaps[1]]
        else:
            # Get random action
            r_action = env.action_space.sample()
            steering = [r_action[0], r_action[1], r_action[2], r_action[3]]
            gear = [r_action[4]]
            flaps = [r_action[5], r_action[6]]
            # reshape random action to correct format
            action = [r_action[0], r_action[1], r_action[2], r_action[3], round(r_action[4]).astype('int'), r_action[5],
                      r_action[6]]

        new_state, reward, done, _ = env.step(action, AIType=AI_type.Cruise)
        # print('step: {} action: {} reward: {}'.format(step, action, reward))

        episode_reward += reward

        # Every step update replay memory and train main network
        # agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.update_replay_memory((current_state, (steering, gear, flaps), reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

        sleep(0.05)

    # Append episode reward to a list and log stats (every given number of episodes)
    episode_reward = round(episode_reward, 1)
    # print('\nepisode: {} episode_reward: {}\n'.format(episode, episode_reward))
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
            agent.model.save(f'train_models/{agent.NAME}__{now_format}__{episode}episode__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__.h5')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    print("\n")
    for index, reward in enumerate(ep_rewards):
        print('Episode: {} Reward: {}'.format(index, reward))
    print("\n")
    # except Exception as e:
    #     print("Error: {} \nErrorValue: {}".format(e.__class__, str(e)))

env.close()