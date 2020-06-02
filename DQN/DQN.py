# import tensorflow as tf
import argparse
from time import sleep

import numpy as np
import random
import os
import gym
import gym_xplane

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


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
