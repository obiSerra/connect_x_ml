
import json
import os
from datetime import datetime

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from kaggle_environments import make
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn as nn

from connectx.common.utils import get_win_percentages, print_win_percentages, update_model_data, get_agent, TqdmCallback, save_model_data
from connectx.common.lookahead import multistep_agent_factory
from connectx.common.ConnectFourGym import ConnectFourGym


import importlib

module = importlib.import_module('models/si_model.py')


learner = module.learner

# Basic Training

timesteps = 10e3
iterations = 10
version = 0
for i in range(1, iterations + 1):
    print(f'Version {i} training vs Random')
    learner.learn(timesteps)

    version = int(timesteps * i)

    learner.save(version)
    agent = learner.get_agent(learner)
    print("Vs Random:")
    results_random = get_win_percentages(agent, "random")
    print_win_percentages(results_random)
    print("Vs Lookahead:")

    results_look = get_win_percentages(agent, multistep_agent_factory())
    print_win_percentages(results_look)

    update_model_data(learner.model_name, version, results_random, results_look)


# Vs Lookahead Training

prev_version = version
env = ConnectFourGym(agent2=multistep_agent_factory())

learner.load_model_version(env, version)

timesteps = 10e3
iterations = 10

for i in range(1, iterations + 1):
    print(f'Version {i} training vs Lookahead')
    learner.learn(timesteps)

    version = int(timesteps * i) + prev_version

    learner.save(version)
    agent = learner.get_agent(learner)
    print("Vs Random:")
    results_random = get_win_percentages(agent, "random")
    print_win_percentages(results_random)

    print("Vs Lookahead:")
    results_look = get_win_percentages(agent, multistep_agent_factory())
    print_win_percentages(results_look)

    update_model_data(learner.model_name, version, results_random, results_look)


# Vs Previous Agent Training

prev_version = version

agent = learner.get_agent(learner)
env = ConnectFourGym(agent2=agent)

learner.load_model_version(env, version)

timesteps = 10e3
iterations = 10

for i in range(1, iterations + 1):
    print(f'Version {i} training vs prev agent')
    learner.learn(timesteps)

    version = int(timesteps * i) + prev_version

    learner.save(version)
    agent = learner.get_agent(learner)
    print("Vs Random:")
    results_random = get_win_percentages(agent, "random")
    print_win_percentages(results_random)

    print("Vs Lookahead:")
    results_look = get_win_percentages(agent, multistep_agent_factory())
    print_win_percentages(results_look)

    update_model_data(learner.model_name, version, results_random, results_look)
    env = ConnectFourGym(agent2=agent)
    learner.load_model_version(env, version)



print("Training Done")
# print("Learing-done")

# th.set_printoptions(profile="full")

# agent_path = os.path.join(os.path.dirname(
#     __file__), f'{model_name}_state_dictionary.py')

# state_dict = learner.policy.to('cpu').state_dict()
# state_dict = {
#     'conv1.weight': state_dict['features_extractor.conv1.weight'],
#     'conv1.bias': state_dict['features_extractor.conv1.bias'],
#     'fc1.weight': state_dict['features_extractor.fc1.weight'],
#     'fc1.bias': state_dict['features_extractor.fc1.bias'],
#     'fc2.weight': state_dict['features_extractor.fc2.weight'],
#     'fc2.bias': state_dict['features_extractor.fc2.bias'],

#     'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
#     'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],
#     'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
#     'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],
#     'policy3.weight': state_dict['mlp_extractor.policy_net.4.weight'],
#     'policy3.bias': state_dict['mlp_extractor.policy_net.4.bias'],

#     'action.weight': state_dict['action_net.weight'],
#     'action.bias': state_dict['action_net.bias'],
# }

# with open(agent_path, mode='w') as file:
#     #file.write(f'\n    data = {learner.policy._get_data()}\n')
#     file.write(f'from torch import tensor\n\n' +
#                f'state_dict = {state_dict}\n')
