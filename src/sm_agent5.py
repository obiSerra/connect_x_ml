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
        self.env = ks_env.train([None, self.adv_agent])
        return ks_env

    def _evaluate_line(self, line, mine=1, oth=2, neu=0):
        """Evaluate a line of 4 possible elements

        Args:
            line: array

        """
        mines = len([m for m in line if m == mine])
        oths = len([o for o in line if o == oth])
        neus = len([n for n in line if n == neu])
        if mines == 3 and neus == 1:
            return 0.7
        elif oths == 3 and neus == 1:
            return -1
        elif mines == 2 and neus == 2:
            return 0.2
        elif oths == 2 and neus == 2:
            return -0.5
        else:
            return 0

    def _point_lines(self, board, start):
        """Given a board and a starting point, return the valid axes.

            eg:
            given a board [[ 0, 1, 2, 3, 4] with start 1 (second element - value 1)
                            [ 0, 5, 6, 7, 8],
                            [ 0, 9,10,11,12],
                            [ 0,13,14,15,16]]

            it will create a sub array 4x4 (if there are enough elements) starting from `start` index
            and it will return the horizontal and vertical axes: [1, 2, 3, 4], [1, 5, 9, 13]
            and the 2 diagonals: [1, 6, 11, 16] and [4, 7, 10, 13]

            All the axes that have less than 4 elements will be discarded

        Args:
            start: int, the index of the element to start

        """

        r, c = np.unravel_index(start, board.shape)
        v_connect = board[r:r + 4, c]
        h_connect = board[r, c:c + 4]
        d_connect = np.diagonal(board[r:r + 4, c:c + 4])
        dr_connect = np.diagonal(np.rot90(board[r:r + 4, c:c + 4]))
        axes = [axe for axe in [v_connect, h_connect,
                                d_connect, dr_connect] if len(axe) == 4]
        return axes

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
            board_points = 0.01
            board = np.array(self.obs['board']).reshape(6, 7)
            mine = self.obs['mark']
            oth = 2 if mine == 1 else 1
            for i in range(board.size):
                axes = self._point_lines(board, i)

                points = [self._evaluate_line(a, mine=mine, oth=oth)
                          for a in axes]

                board_points += sum(points)

            board_points = board_points / 100
            return board_points

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
    "clip_range": 0.05,
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

    last_version = str(int(adv_iters * adv_timesteps))
    print("Running negamax training")
    connectx.learn("negamax", iters=adv_iters,
                   timesteps=adv_timesteps,
                   load_version=last_version, update_name="_vs_negamax")