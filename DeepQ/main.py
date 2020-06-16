import gym
import numpy as np
from custom_gym.envs.myxpc.utils import observation
from dqn import Agent

gym_env = gym.make('custom_gym:Xplane-v0')
lr = 0.001
gam = 0.99
n_games = 3
agent = Agent(learning_rate=lr, gamma=gam, epsilon=1.0, 
    input_dims=observation().shape, n_actions=16, batch_size=30, file_name='dq_model_1.h5')
scores = []
eps_hist = []

for i in range(n_games):
    done = False
    score = 0 
    observation = gym_env.reset()
    print("Game", i)
    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, done = gym_env.step(action)
        score = score + reward
        agent.store_transition(observation, action, reward, new_observation, done)
        observation = new_observation
        agent.learn()
    eps_hist.append(agent.epsilon)
    scores.append(score)
    
print(scores)
agent.save_model()
    

