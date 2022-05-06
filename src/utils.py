import random

import numpy as np
from gym import spaces
from kaggle_environments import evaluate, make
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate(
        "connectx", [agent1, agent2], config, [], n_rounds // 2)
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b, a] for [a, b]
                 in evaluate(
                     "connectx",
        [agent2, agent1], config, [], n_rounds - n_rounds // 2)]

    win_perc_agent1 = np.round(
        outcomes.count([1, -1]) / len(outcomes), 2)
    invalid_agent1 = outcomes.count([None, 0])
    invalid_agent2 = outcomes.count([0, None])
    win_perc_agent2 = np.round(
        outcomes.count([-1, 1]) / len(outcomes), 2)
    ret = [[win_perc_agent1, invalid_agent1],
           [win_perc_agent2, invalid_agent2]]

    return ret


def print_win_percentages(agent1, agent2, n_rounds=100):
    [[win_perc_agent1, invalid_agent1], [win_perc_agent2, invalid_agent2]
     ] = get_win_percentages(agent1, agent2, n_rounds=n_rounds)
    print("Agent 1 Win Percentage:", win_perc_agent1)
    print("Agent 2 Win Percentage:", win_perc_agent2)
    print("Number of Invalid Plays by Agent 1:", invalid_agent1)
    print("Number of Invalid Plays by Agent 2:", invalid_agent2)


class ConnectFourGym:
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(self.rows, self.columns, 1), dtype=np.int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-100, 10)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 10
        elif done:  # The opponent won the game
            return -10
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -100, True, {}
        return np.array(self.obs['board']).reshape(
            self.rows, self.columns, 1), reward, done, _


class ConnectFourGymV2:
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(self.rows, self.columns, 1), dtype=np.int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 2)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(self.rows, self.columns, 1)

    def change_reward(self, old_reward, done):
        if old_reward == 1:
            return 2
        elif done:  # The opponent won the game
            return -2
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(
            self.rows, self.columns, 1), reward, done, _


class ConnectFourGymV3:
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.new_shape = (1, self.rows, self.columns)
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=self.new_shape, dtype=np.int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 2)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(*self.new_shape)

    def change_reward(self, old_reward, done):
        if old_reward == 1:
            return 2
        elif done:
            return -2
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(
            *self.new_shape), reward, done, _


class ConnectFourGymV4:
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.new_shape = (1, self.rows, self.columns)
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=self.new_shape, dtype=np.int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 2)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(*self.new_shape)

    def change_reward(self, old_reward, done):
        if old_reward == 1:
            return 2
        elif done:
            return -2
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(
            *self.new_shape), reward, done, _


def agent_factory(model):
    def agent(obs, config):
        # Use the best model to select a column
        col, _ = model.predict(np.array(obs['board']).reshape(1, 6, 7))
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
