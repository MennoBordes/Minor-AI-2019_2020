# import tensorflow as tf
import argparse
import numpy as np
import random
import os
import gym
import gym_xplane

'''
CWD = os.getcwd()
# print(CWD)


class DQModel(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int):
        super(DQModel, self).__init__()
        self.ml1 = tf.keras.layers.Input


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(16,)))
model.add(tf.keras.layers.Dense(32))

print(model)
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clientAddr', help='client host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='xplane host address', default='127.0.0.1')
    parser.add_argument('--xpPort', help='xplane port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=1)

    args = parser.parse_args()

    # env = gym.make('xplane-gym-v0', clientAddr=args.clientAddr, xpHost=args.xpHost,
    #                xpPort=args.xpPort, clientPort=args.clientPort)
    env = gym.make('xplane-gym-v0')
    env.clientAddr = args.clientAddr
    env.xpHost = args.xpHost
    env.xpPort = args.xpPort
    env.clientPort = args.xpPort

    print(env)
    env.reset()
    env.close()
