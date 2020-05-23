from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import XPlaneConnect.xpc as xpc
import json


from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class PyEnvironment(object):

  def reset(self):
    """Return initial_time_step."""
    self._current_time_step = self._reset()
    return self._current_time_step

  def step(self, action):
    """Apply action and return new time_step."""
    if self._current_time_step is None:
        return self.reset()
    self._current_time_step = self._step(action)
    return self._current_time_step

  def current_time_step(self):
    return self._current_time_step

  def time_step_spec(self):
    """Return time_step_spec."""

  @abc.abstractmethod
  def observation_spec(self):
    """Return observation_spec."""

  @abc.abstractmethod
  def action_spec(self):
    """Return action_spec."""

  @abc.abstractmethod
  def _reset(self):
    """Return initial_time_step."""

  @abc.abstractmethod
  def _step(self, action):
    """Apply action and return new time_step."""
    self._current_time_step = self._step(action)
    return self._current_time_step

class XPlaneEnv(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=100, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 0:
      self._episode_ended = True
    elif 0 < action < 100:
      with xpc.XPlaneConnect() as client:
        command = 'sim/joystick/yoke_pitch_ratio'
        client.sendDREF(command, 1)
      self._state += 1

    if self._episode_ended or self._state == 10:
      reward = 1
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
        np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

def add_waypoints(self, json_path):
  waypoints = []

  with open(json_path) as json_file:
    nodes = json.load(json_file)
    data = nodes['nodes']
    for index, data in enumerate(data):
      if index is 0:
        # Set first waypoint to starting position
        waypoints.append(self.position[0])
        waypoints.append(self.position[1])
        waypoints.append(self.position[2])

        # Add waypoints for Schiphol end of runway 18R
        waypoints.append(52.3286247253418)  # Latitude
        waypoints.append(4.708907604217529)  # Longitude
        waypoints.append(150)  # Altitude
        continue
      # Add waypoints from file
      waypoints.append(data['lat'])
      waypoints.append(data['lon'])
      waypoints.append((data['alt'] / 3.2808))

  self.waypoints = waypoints
  XplaneENV.CLIENT.sendWYPT(op=1, points=waypoints)

def remove_waypoints(self):
  XplaneENV.CLIENT.sendWYPT(op=3, points=[])
  pass

environment = XPlaneEnv()
utils.validate_py_environment(environment, episodes=5)
