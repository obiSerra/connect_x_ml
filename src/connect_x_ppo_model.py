import json
import math
import os
import random
from datetime import datetime

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from scipy import signal
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from utils.lookahead import multistep_agent_factory
from utils.utils import TqdmCallback, agent_factory, get_win_percentages


class ConnectModel:

    def __init__(self, gym_class, model_name, model_params):
        self.gym_class = gym_class
        self.origin_name = model_name
        self.model_name = model_name
        self.model_params = model_params
        self.init_model()

    def init_model(self):
        self.model_dir = f"models/{self.model_name}"
        self.logdir = "logs"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def test_agent_results(self, agent):
        adv_agent = multistep_agent_factory()
        vs_random = get_win_percentages(agent1=agent, agent2="random")
        vs_adv = get_win_percentages(agent1=agent, agent2=adv_agent)
        # vs_negamax = get_win_percentages(agent1=agent, agent2="negamax")
        vs_negamax = [[0, 0], [0, 0]]

        return (vs_random, vs_adv, vs_negamax)

    def score_progress(self, results):
        vs_random, vs_adv, vs_negamax = results
        total_points = math.floor(
            (vs_random[0][0] - vs_random[0][1]) * 10) * 0.1
        total_points += math.floor((vs_adv[0][0] - vs_adv[0][1]) * 10) * 0.5
        total_points += math.floor((vs_negamax[0][0] - vs_negamax[0][1]) * 10)
        return total_points

    def print_progress(self, results, version=""):
        vs_random, vs_adv, vs_negamax = results
        score = self.score_progress(results)
        print("vs random")
        print(f" win: {vs_random[0][0]} | invalid: {vs_random[0][1]}")
        print("vs adv_agent")
        print(
            f" win: {vs_adv[0][0]} | invalid: {vs_adv[0][1]}")
        print("vs negamax")
        print(
            f" win: {vs_negamax[0][0]} | invalid: {vs_negamax[0][1]}")
        print(f"score: {score}")

        lines = [f"{self.model_name},",
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
        with open(f"data/progress_{self.model_name}.csv", "a") as file1:
            # Writing data to a file
            file1.writelines(lines)

        return vs_negamax[0][0]

    def generate_env(self, adv_agent):
        env = self.gym_class(adv_agent=adv_agent)
        env.reset()
        return env

    def _save_model_data(self):

        print(f"Model: {self.model_name}")
        print("Model Policy:")
        print("")
        print(self.model.policy)
        print("")
        print(self.gym_class)
        print("")

        with open(f"data/policy_{self.model_name}.txt", "w") as file:
            file.write("Params:\n")
            file.write(json.dumps(str(self.model_params)))
            file.write("\n")
            file.write(str(self.gym_class))
            file.write("\n")
            file.write(str(self.model.policy))

        # creating the csv file
        with open(f"data/progress_{self.model_name}.csv", "w") as file:
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

    def get_agent(self):
        def agent(obs, config):
            # Use the best model to select a column
            col, _ = self.model.predict(
                np.array(obs['board']).reshape(1, 6, 7))
            # Check if selected column is valid
            is_valid = (obs['board'][int(col)] == 0)
            # If not valid, select random move.
            if is_valid:
                return int(col)
            else:
                return random.choice(
                    [col for col in range(config.columns)
                     if obs.board[int(col)] == 0])

        return agent

    def learn(self, adv_agent, iters=-1, timesteps=10e3, load_version=None,
              update_name=''):
        self.load_model(adv_agent, version=load_version)
        if update_name != '':
            self.model_name = self.origin_name + update_name
            self.init_model()
        self._save_model_data()
        i = 0
        while iters < 0 or i < iters:

            self.model.learn(
                total_timesteps=timesteps, reset_num_timesteps=False,
                callback=TqdmCallback(timesteps),
                tb_log_name=self.model_name)
            i += 1
            version = int(timesteps * i)
            self.model.save(f"{self.model_dir}/{version}")

            print(f"Done model {self.model_name} version {version}")
            agent = self.get_agent()
            print("Evaluating model...")
            progress = self.test_agent_results(agent)
            self.print_progress(progress, version=version)

    def self_learn(self, start_agent="random",
                   iters=-1, timesteps=10e3, load_version=None,
                   update_name=''):
        print("Self learning")
        self.load_model(start_agent, version=load_version)
        if update_name != '':
            self.model_name = self.origin_name + update_name
            self.init_model()
        self._save_model_data()
        i = 0
        while iters < 0 or i < iters:

            self.model.learn(
                total_timesteps=timesteps, reset_num_timesteps=False,
                callback=TqdmCallback(timesteps),
                tb_log_name=self.model_name)
            i += 1
            version = int(timesteps * i)
            self.model.save(f"{self.model_dir}/{version}")

            print(f"Done model {self.model_name} version {version}")
            agent = self.get_agent()
            print("Evaluating model...")
            progress = self.test_agent_results(agent)
            self.print_progress(progress, version=version)
            self.load_model(agent, version=version)


class ConnectXPPOModel(ConnectModel):

    def load_model(self, adv_agent, version=None):
        env = self.generate_env(adv_agent)
        model = PPO('MlpPolicy', env, verbose=0,
                    tensorboard_log=self.logdir,
                    **self.model_params)
        if version is not None:
            model = model.load(f"{self.model_dir}/{version}", env=env)

        self.model = model


class ConnectXDQNModel(ConnectModel):

    def load_model(self, adv_agent, version=None):
        env = self.generate_env(adv_agent)
        model = DQN('MlpPolicy', env, verbose=0,
                    tensorboard_log=self.logdir,
                    **self.model_params)
        if version is not None:
            model = model.load(f"{self.model_dir}/{version}", env=env)

        self.model = model
