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

from connectx.common.utils import get_win_percentages, print_win_percentages, get_agent, TqdmCallback, save_model_data, update_model_data
from connectx.common.lookahead import multistep_agent_factory
from connectx.common.ConnectFourGym import ConnectFourGym

model_name = "test_model_pipeline"

env = ConnectFourGym()


class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 42):
        super(Net, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(294, 140)
        self.fc2 = nn.Linear(140, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [dict(pi=[42, 42, 42], vf=[42, 42, 42])],
    'features_extractor_class': Net,
}

model_params = {
    "batch_size": 100,
    "n_steps": 1000,
    "policy_kwargs": policy_kwargs,
    # "clip_range": 0.3,
}


class Model():
    def __init__(self, env, model_name, model_params):
        self.model_name = model_name
        self.model_params = model_params

        self.log_dir = "logs/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_dir = f"models/{model_name}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.learner = PPO('MlpPolicy', env, verbose=0,
                           tensorboard_log=self.log_dir,
                           **self.model_params)

        print(learner.policy)

        save_model_data(model_name, model_params, self.learner)

    def load_model_version(self, env, version):
        self.learner = PPO('MlpPolicy', env, verbose=0,
                           tensorboard_log=self.log_dir,
                           **self.model_params)
        self.learner = self.learner.load(f"{self.model_dir}/{version}", env=env)

    def learn(self, timesteps):
        self.learner.learn(total_timesteps=timesteps,
                           tb_log_name=self.model_name,
                           callback=TqdmCallback(timesteps),
                           reset_num_timesteps=False)

    def save(self, version):
        learner.save(f"{self.model_dir}/{version}")

    def get_agent(self):
        return get_agent(self.learner)


learner = Model(env, model_name, model_params)


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
