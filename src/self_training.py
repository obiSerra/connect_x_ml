import os
import sys
import math
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from lookahead import multistep_agent_factory
from utils import (ConnectFourGymV5, TqdmCallback, agent_factory,
                   get_win_percentages)


def test_agent_results(agent):
    adv_agent = multistep_agent_factory()
    vs_random = get_win_percentages(agent1=agent, agent2="random")
    vs_adv = get_win_percentages(agent1=agent, agent2=adv_agent)
    vs_negamax = get_win_percentages(agent1=agent, agent2="negamax")
    return (vs_random, vs_adv, vs_negamax)


def score_progress(results):
    vs_random, vs_adv, vs_negamax = results
    total_points = math.floor((vs_random[0][0] - vs_random[0][1]) * 10) * 0.1
    total_points += math.floor((vs_adv[0][0] - vs_adv[0][1]) * 10) * 0.5
    total_points += math.floor((vs_negamax[0][0] - vs_negamax[0][1]) * 10)
    return total_points


def print_progress(results, name, version=""):
    vs_random, vs_adv, vs_negamax = results
    score = score_progress(results)
    print("vs random")
    print(f" win: {vs_random[0][0]} | invalid: {vs_random[0][1]}")
    print("vs adv_agent")
    print(
        f" win: {vs_adv[0][0]} | invalid: {vs_adv[0][1]}")
    print("vs negamax")
    print(
        f" win: {vs_negamax[0][0]} | invalid: {vs_negamax[0][1]}")
    print(f"score: {score}")

    lines = [f"{name},",
             f"{version},",
             f"{vs_random[0][0]},",
             f"{vs_random[0][1]},",
             f"{vs_adv[0][0]},",
             f"{vs_adv[0][1]},",
             f"{vs_negamax[0][0]},",
             f"{vs_negamax[0][1]},",
             f"{score},",
             f"{datetime.now()}",
             "\n"
             ]
    # Writing to file
    with open(f"data/progress_{name}.csv", "a") as file1:
        # Writing data to a file
        file1.writelines(lines)

    return vs_negamax[0][0]


def evaluate_model():
    pass

# Setup


TIMESTEPS = int(10e3)
starting_iter = 0
iters = 0
max_iters = 1000
model_name = "self_improv_1"

# Log setup
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Model setup

MODEL_DIR = f"models/{model_name}"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def env_factory(adv_agent):
    env = ConnectFourGymV5(agent2=adv_agent)
    env.reset()
    return env


def PPO_model_factory(env, version=None, params={}):
    model = PPO('MlpPolicy', env, verbose=0,
                tensorboard_log=logdir,
                **params)
    if version is not None:
        model = model.load(f"{MODEL_DIR}/{version}", env=env)
    return model


class FeatureExtractorNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 21):
        super(FeatureExtractorNet, self).__init__(
            observation_space, features_dim)
        # edge extraction
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 9, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(9, 12, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(12, 6, kernel_size=3, stride=1)
        # ridurre pesantemente feature dimentions
        self.fc3 = nn.Linear(36, features_dim)

    def forward(self, x):
        # print("in", x)
        # print(x.size())
        x = x / 2  # input between 0/1
        # print("in2", x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        # print("out", x)
        return x


policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [dict(pi=[21, 21, 21, 21], vf=[21, 21, 21, 21])],
    'features_extractor_class': FeatureExtractorNet,
}

next_agent = "random"
# next_agent = multistep_agent_factory()
env = env_factory(next_agent)
model_params = {
    "batch_size": 84,
    "n_steps": 512,
    "policy_kwargs": policy_kwargs,
    # learning_rate=2.5e-4,
    # clip_range=0.2,
    # n_epochs=20,
}

model = PPO_model_factory(env, params=model_params)

print("Model Policy:")
print("")
print(model.policy)
print("")

# saving the policy
with open(f"data/policy_{model_name}.txt", "w") as file:
    file.write(str(model.policy))

# creating the csv file
with open(f"data/progress_{model_name}.csv", "w") as file:
    lines = ["model,"
             "version,",
             "vs random win,",
             "vs random invalid,",
             "vs adv_agent win,",
             "vs adv_agent invalid,",
             "vs negamax win,",
             "vs negamax invalid,",
             "score,",
             "saved on",
             "\n"]
    # Writing to file

    file.writelines(lines)

agents = []
print("Start training")

while iters < max_iters:
    iters += 1
    version = TIMESTEPS * (iters + starting_iter)
    print(f"Iteration: {iters}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,
                callback=TqdmCallback(TIMESTEPS), tb_log_name=model_name)

    model.save(f"{MODEL_DIR}/{version}")
    print(f"Done model version: {version}")
    agent = agent_factory(model)
    print("Evaluating model...")
    progress = test_agent_results(agent)
    scores = score_progress(progress)
    print_progress(progress, model_name, version=version)
    vs_random, vs_adv, vs_negamax = progress

    if vs_random[0][0] < 0.7:
        next_agent = "random"
    else:
        next_agent = agent

    if vs_adv[0][0] >= 0.8:
        print("DOOOOOONE")
        break

    model = PPO_model_factory(env=env_factory(
        next_agent), version=version)
