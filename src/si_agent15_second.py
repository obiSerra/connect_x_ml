import os
import random

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gym import spaces
from kaggle_environments import make
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from tqdm import tqdm

from connect_x_ppo_model import ConnectXPPOModel
from utils.lookahead import multistep_agent_factory


class ConnectFourGym:
    def __init__(self, adv_agent="random", rewards=(1, -1, -10)):
        self.adv_agent = adv_agent
        self.rewards = rewards

        ks_env = self._set_env()
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.new_shape = (1, self.rows, self.columns)
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=self.new_shape, dtype=np.int)
        self.reward_range = (min(rewards), max(rewards))
        self.spec = None
        self.metadata = None

    def _set_env(self):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([self.adv_agent, None])
        return ks_env

    def reset(self):
        self.obs = self.env.reset()
        return self.get_board()

    def get_board(self):
        return np.array(self.obs['board']).reshape(*self.new_shape)

    def change_reward(self, old_reward, done):
        if old_reward == 1:
            return self.rewards[0]
        elif done:
            return self.rewards[1]
        else:
            reward = 1 / (self.rows * self.columns * self.obs['step'])
            return reward

    def step(self, action):
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:
            reward, done, _ = self.rewards[2], True, {}
        return self.get_board(), reward, done, _


class FeatureExtractorNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 42):
        super(FeatureExtractorNet, self).__init__(
            observation_space, features_dim)

    def forward(self, x):
        x = nn.Flatten()(x)
        return x


policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [dict(pi=[42, 42, 42, 42, 42, 42, 42],
                      vf=[7, 7, 7, 7, 7, 7, 7])],
    'features_extractor_class': FeatureExtractorNet,
    'normalize_images': False
}

model_params = {
    "batch_size": 100,
    "n_steps": 1000,
    "policy_kwargs": policy_kwargs,
    "clip_range": 0.3,
}

model_name = os.path.basename(__file__).replace('.py', '')


if __name__ == '__main__':
    connectx = ConnectXPPOModel(ConnectFourGym, model_name, model_params)

    basic_iters = 10
    basic_timesteps = 10e3
    adv_iters = 20
    adv_timesteps = 10e4

    connectx.learn('random', iters=basic_iters, timesteps=basic_timesteps)

    last_version = str(int(basic_iters * basic_timesteps))
    print("Running adv training")
    connectx.learn(multistep_agent_factory(), iters=adv_iters,
                   timesteps=adv_timesteps,
                   load_version=last_version, update_name="_vs_adv")
