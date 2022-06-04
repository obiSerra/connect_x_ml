
from datetime import datetime
import numpy as np
from kaggle_environments import make, evaluate
from torch import tensor
import random
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from collections import namedtuple
import json
import logging


LOGGING_FILE = "debug.log"

logging.basicConfig(filename=LOGGING_FILE,
                    encoding='utf-8', level=logging.DEBUG)


Results = namedtuple(
    'Results', "agent1_wins agent2_wins agent1_inv agent2_inv")


def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate(
        "connectx", [agent1, agent2], config, [], n_rounds // 2)
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b, a] for [a, b]
                 in evaluate("connectx", [agent2, agent1], config, [], n_rounds - n_rounds // 2)]

    agent_1_wins = np.round(outcomes.count([1, -1]) / len(outcomes), 2)
    agent_2_wins = np.round(outcomes.count([-1, 1]) / len(outcomes), 2)
    agent_1_invalid = outcomes.count([None, 0])
    agent_2_invalid = outcomes.count([0, None])
    Results = namedtuple(
        'Results', "agent1_wins agent2_wins agent1_inv agent2_inv")

    return Results(agent_1_wins, agent_2_wins, agent_1_invalid, agent_2_invalid)


def print_win_percentages(results: Results):

    print("Agent 1 Win Percentage:", results.agent1_wins)
    print("Agent 2 Win Percentage:", results.agent2_wins)
    print("Number of Invalid Plays by Agent 1:", results.agent1_inv)
    print("Number of Invalid Plays by Agent 2:", results.agent2_inv)


def get_agent(model):
    def agent(obs, config):
        try:
            obs = tensor(obs["board"]).reshape(
                1, 1, config.rows, config.columns).float()

            obs = obs / 2
            board_2d = obs.reshape(config.rows, config.columns)

            action, _ = model.predict(obs)

            is_valid = any(board_2d[:, int(action)] == 0)

            if not is_valid:
                moves = []
                for c in range(1, config.columns):

                    if any(board_2d[:, c] == 0):
                        moves.append(c)
                return random.choice(moves)
            return int(action)
        except Exception as e:
            logging.error(f"{e}")
    return agent


class TqdmCallback(BaseCallback):
    def __init__(self, timesteps, verbose=0):
        super(TqdmCallback, self).__init__(verbose)
        self.timesteps = timesteps
        self.pbar = tqdm(total=self.timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


def save_model_data(model_name, model_params, learner):
    with open(f"data/policy_{model_name}.txt", "w") as file:
        file.write("Params:\n")
        file.write(json.dumps(str(model_params)))
        file.write("\n")
        file.write("\n")
        file.write(str(learner.policy))

    # creating the csv file
    with open(f"data/progress_{model_name}.csv", "w") as file:
        lines = ["model,"
                 "version,",
                 "vs random win,",
                 "vs random invalid,",
                 "vs look win,",
                 "vs look invalid,",
                 "saved on",
                 "\n"]
        # Writing to file

        file.writelines(lines)


def update_model_data(model_name, version, results_random, results_look):
    with open(f"data/progress_{model_name}.csv", "a") as file:
        lines = [f"{model_name},",
                 f"{version},",
                 f"{results_random.agent1_wins},",
                 f"{results_random.agent1_inv},",
                 f"{results_look.agent1_wins},",
                 f"{results_look.agent1_inv},",
                 f"{datetime.now()}",
                 "\n"]
    # Writing to file

        file.writelines(lines)
