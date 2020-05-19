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
    # env = gym.make('xplane-gym-v0', clientAddr=args.clientAddr, xpHost=args.xpHost,
    #                xpPort=args.xpPort, clientPort=args.clientPort)
    env = gym.make('xplane-gym-v0')
    # env.reset()
    # env.clientAddr = args.clientAddr
    # env.xpHost = args.xpHost
    # env.xpPort = args.xpPort
    # env.clientPort = args.xpPort
    # env.reset()

    # Create waypoints to target
    env.remove_waypoints()
    # env.add_waypoints('Routes/EHAM-LEVC_amsterdam-valencia.json')
    env.add_waypoints('Routes/EHAM-LZIB_amsterdam-bratislava.json')

    # SEED environment
    env.action_space.seed(0)

    agent = RandomAgent(env.action_space)

    EPISODES = 5
    episode = 0

    for episode in range(EPISODES):
        state = env.reset()
        done = False

        # itera = 0
        # for itera in range(100):
        #     print(itera)
        while not done:
            action = agent.act()
            new_state, reward, done, _ = env.step(action)

            # print("State: ", state, "Reward: ", reward)
            # print("Done: ", done, "New state: ", new_state)

            sleep(0.05)
    env.close()
