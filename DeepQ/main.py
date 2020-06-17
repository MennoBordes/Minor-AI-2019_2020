# import gym
# import numpy as np
# from custom_gym.envs.myxpc.utils import observation
# from dqn import Agent
# import time

# gym_env = gym.make('custom_gym:Xplane-v0')
# lr = 0.001
# gam = 0.99
# n_games = 200
# agent = Agent(learning_rate=lr, gamma=gam, epsilon=1.0, 
#     input_dims=observation().shape, n_actions=15, batch_size=70, file_name='saved_models/dq_model_1.h5')
# scores = []
# eps_hist = []
# agent.load_model()

# for i in range(n_games):
#     done = False
#     score = 0 
#     observation = gym_env.reset()
#     time.sleep(2)
#     observation_checkpoints = np.array([observation[0:2]])
#     step_counter = 0
#     print("GAME ITERATION ", i)
#     while not done:
#         action = agent.choose_action(observation)
#         new_observation, reward, done = gym_env.step(action)
#         step_counter = step_counter + 1
#         score = score + reward
#         agent.store_transition(observation, action, reward, new_observation, done)
#         observation = new_observation
#         agent.learn()
#         # This if statement checks if the airplane is stuck 
#         observation_checkpoints = np.append(observation_checkpoints, [new_observation[0:2]], axis=0)
#         print(observation_checkpoints)
#         print("stepcounter is", step_counter)
#         if step_counter % 30 == 0:
#             if np.array_equal(observation_checkpoints[step_counter - 30], observation_checkpoints[step_counter - 1]):
#                 done = True
#         agent.save_model()
      
#     eps_hist.append(agent.epsilon)
#     scores.append(score)
    
# print(scores)

    

