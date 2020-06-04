import tensorflow as tf
import argparse
from time import sleep

import numpy as np
import random
import os
import gym
import gym_xplane

if __name__ == '__main__':
    # SETUP ENVIRONMENT
    env = gym.make('xplane-gym-v0')

    # Create waypoints to target
    env.remove_waypoints()
    env.add_waypoints('Routes\EHAM_amsterdam_landing.json')

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

            print("State: ", state, "Reward: ", reward)
            print("Done: ", done, "New state: ", new_state)

            sleep(0.05)
    env.close()
