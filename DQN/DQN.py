import tensorflow as tf
import numpy as np
import random
import os

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
