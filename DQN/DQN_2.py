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
from gym_xplane.envs.xplane_env import AI_type
from DQN.current_training_model import current_training
import DQN.graph as graph

# Cruise model
from ai_cruise import AI_Cruise
# Landing model
from ai_landing import AI_Landing

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 100  # 100
MINIBATCH_SIZE = 32  # 64
UPDATE_TARGET_EVERY = 5
LEARNING_RATE = 0.01
MIN_REWARD = -50

# === Check which model is being trained
# Waypoint and checkpoint files
if current_training == AI_type.Cruise:
    CURRENT_MODEL = AI_type.Cruise
    WAYPOINT_FILE = 'Routes/flight_straight_1.json'
    WAYPOINT_START_LAND = False
    CHECKPOINT_PATH = 'training_2/cp-{date}.ckpt'
    agent = AI_Cruise(LEARNING_RATE=LEARNING_RATE, DISCOUNT=DISCOUNT,
                      MINIBATCH_SIZE=MINIBATCH_SIZE, REPLAY_MEMORY_SIZE=REPLAY_MEMORY_SIZE,
                      UPDATE_COUNTER=UPDATE_TARGET_EVERY, checkpoint_path=CHECKPOINT_PATH)

elif current_training == AI_type.Landing:
    CURRENT_MODEL = AI_type.Landing
    WAYPOINT_FILE = 'Routes/EHAM_amsterdam_approach.json'
    WAYPOINT_START_LAND = False
    CHECKPOINT_PATH = 'training_3/cp-{date}.ckpt'

    #     TODO  UPDATE TO YOUR OWN MODEL
    agent = AI_Landing(LEARNING_RATE=LEARNING_RATE, DISCOUNT=DISCOUNT,
                      MINIBATCH_SIZE=MINIBATCH_SIZE, REPLAY_MEMORY_SIZE=REPLAY_MEMORY_SIZE,
                      UPDATE_COUNTER=UPDATE_TARGET_EVERY, checkpoint_path=CHECKPOINT_PATH)
else:
    CURRENT_MODEL = AI_type.TakeOff
    WAYPOINT_FILE = 'DQN\Routes\EHAM_amsterdam_approach.json'
    WAYPOINT_START_LAND = True
    CHECKPOINT_PATH = 'training_4/cp-{date}.ckpt'

    #     TODO  UPDATE TO YOUR OWN MODEL
    agent = AI_Cruise(LEARNING_RATE=LEARNING_RATE, DISCOUNT=DISCOUNT,
                      MINIBATCH_SIZE=MINIBATCH_SIZE, REPLAY_MEMORY_SIZE=REPLAY_MEMORY_SIZE,
                      UPDATE_COUNTER=UPDATE_TARGET_EVERY, checkpoint_path=CHECKPOINT_PATH)

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
# env.action_space.seed(0)

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Load existing weights
latest_weights = tf.train.latest_checkpoint("cp-2020-06-25T14-08-05.ckpt")
agent.model.load_weights(latest_weights)
agent.target_model.load_weights(latest_weights)

ep_rewards = []
highest_reward = -100

AGGREGATE_STATS_EVERY = 2

EPISODES = 1
episode = 0

# while episode < EPISODES:
#     try:
# for episode in range(EPISODES):
# Using tqdm for tracking info about the episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    try:

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
            try:
                if np.random.random() > epsilon:
                    # Get predicted action
                    # action = agent.get_qs(current_state)
                    steering, gear, flaps = agent.model.predict(current_state)
                    action = [steering[0], steering[1], steering[2], steering[3],
                              round(gear[0].astype('int')), flaps[0],
                              flaps[1]]
                else:
                    # Get random action
                    r_action = env.action_space.sample()
                    steering = [r_action[0], r_action[1], r_action[2], r_action[3]]
                    gear = [r_action[4]]
                    flaps = [r_action[5], r_action[6]]
                    # reshape random action to correct format
                    action = [r_action[0], r_action[1], r_action[2], r_action[3],
                              round(r_action[4]).astype('int'), r_action[5], r_action[6]]

                new_state, reward, done, _ = env.step(action, AIType=CURRENT_MODEL)

                episode_reward += reward

                # Every step update replay memory and train main network
                agent.update_replay_memory((current_state, (steering, gear, flaps), reward, new_state, done))
                agent.train(done, step)

                current_state = new_state
                step += 1
                sleep(0.1)
            except Exception as e:
                print(f'state: {new_state}')
                print(f"Error: {e.__class__} \nErrorValue: {str(e)}")

        # Append episode reward to a list and log stats (every given number of episodes)
        episode_reward = round(episode_reward, 1)
        ep_rewards.append(episode_reward)

        graph.checktime()
        graph.check_fuel()
        #print
        # Save model if score is higher than previous highest score
        if episode_reward > highest_reward:
            # Set new highest reward
            highest_reward = episode_reward
            now = datetime.now()
            now_format = now.strftime("%Y-%m-%dT%H-%M-%S")
            agent.model.save(
                f'train_models/{agent.NAME}__{now_format}__'
                f'{episode}episode__'
                f'{episode_reward:_>7.2f}reward__.h5')

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

            # Save model, but only when min reward is greater or equal to set value
            if min_reward >= MIN_REWARD:
                now = datetime.now()
                now_format = now.strftime("%Y-%m-%dT%H-%M-%S")
                agent.model.save(
                    f'train_models/{agent.NAME}__{now_format}__'
                    f'{episode}episode__'
                    f'{max_reward:_>7.2f}max_'
                    f'{average_reward:_>7.2f}avg_'
                    f'{min_reward:_>7.2f}min__.h5')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        print(f'\nCurrent episode: {episode} reward: {episode_reward}')
        print(f'Highest episode: {ep_rewards.index(max(ep_rewards)) + 1} reward: {max(ep_rewards)}')
        # for index, reward in enumerate(ep_rewards):
        #     print(f'Episode: {index} Reward: {reward}')
    except Exception as e:
        ep_rewards.append(-999)
        print(f"Error: {e.__class__} \nErrorValue: {str(e)}")

env.close()
