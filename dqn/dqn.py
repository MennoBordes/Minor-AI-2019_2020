import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from environment.xplane_env import XPlaneEnv

q_net = q_network.QNetwork(
  XPlaneEnv.observation_spec(),
  XPlaneEnv.action_spec(),
  fc_layer_params=(100,))

agent = dqn_agent.DqnAgent(
  XPlaneEnv.time_step_spec(),
  XPlaneEnv.action_spec(),
  q_network=q_net,
  optimizer=optimizer,
  td_errors_loss_fn=common.element_wise_squared_loss,
  train_step_counter=tf.Variable(0))

agent.initialize()