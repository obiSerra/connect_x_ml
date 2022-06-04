import gym
from kaggle_environments import make

import os
import numpy as np
import torch as th
from torch import nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from connectx.common.utils import get_win_percentages, get_agent, TqdmCallback
from connectx.common.lookahead import multistep_agent_factory
from connectx.common.ConnectFourGym import ConnectFourGym


env = ConnectFourGym()

model_name = "model_test"
log_dir = "logs/"

os.makedirs(log_dir, exist_ok=True)
model_dir = f"saved_models/{model_name}"
os.makedirs(log_dir, exist_ok=True)



class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3 = nn.Linear(384, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        return x


policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [64, dict(pi=[32, 16], vf=[32, 16])],
    'features_extractor_class': Net,
}

model_params = {
    # "batch_size": 100,
    # "n_steps": 1000,
    "policy_kwargs": policy_kwargs,
    # "clip_range": 0.3,
}


learner = PPO('MlpPolicy', env, verbose=0,
              tensorboard_log=log_dir,
              **model_params)
print(learner.policy)

timesteps = 10e3
iterations = 10
for i in range(1, iterations + 1):
    print(f'iteration {i}')
    learner.learn(total_timesteps=timesteps,
                  tb_log_name=model_name,
                  callback=TqdmCallback(timesteps),
                  reset_num_timesteps=False)

    version = int(timesteps * i)

    learner.save(f"{model_dir}/{version}")
    agent = get_agent(learner)
    print("Vs Random:")
    get_win_percentages(agent, "random")
    print("Vs Lookahead:")
    get_win_percentages(agent, multistep_agent_factory())

    if i > iterations / 2:
        print("Train vs Lookahead")
        env = ConnectFourGym(agent2=multistep_agent_factory())
        learner = PPO('MlpPolicy', env, verbose=0,
                      tensorboard_log=log_dir,
                      **model_params)
        learner = learner.load(f"{model_dir}/{version}", env=env)


print("Learing-done")

th.set_printoptions(profile="full")

agent_path = os.path.join(os.path.dirname(__file__), 'state_dictionary.py')

state_dict = learner.policy.to('cpu').state_dict()
state_dict = {
    'conv1.weight': state_dict['features_extractor.conv1.weight'],
    'conv1.bias': state_dict['features_extractor.conv1.bias'],
    'conv2.weight': state_dict['features_extractor.conv2.weight'],
    'conv2.bias': state_dict['features_extractor.conv2.bias'],
    'fc3.weight': state_dict['features_extractor.fc3.weight'],
    'fc3.bias': state_dict['features_extractor.fc3.bias'],

    'shared1.weight': state_dict['mlp_extractor.shared_net.0.weight'],
    'shared1.bias': state_dict['mlp_extractor.shared_net.0.bias'],

    'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
    'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],
    'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
    'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],

    'action.weight': state_dict['action_net.weight'],
    'action.bias': state_dict['action_net.bias'],
}

with open(agent_path, mode='w') as file:
    #file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(f'from torch import tensor\n\n' +
               f'state_dict = {state_dict}\n')
