import random

import numpy as np
from gym import spaces
from kaggle_environments import evaluate, make
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


def get_win_percentages(agent1, agent2, n_rounds=100, start_as=-1):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    if start_as == -1:
        # Agent 1 goes first (roughly) half the time
        outcomes = evaluate(
            "connectx", [agent1, agent2], config, [], n_rounds // 2)
        # Agent 2 goes first (roughly) half the time
        outcomes += [[b, a] for [a, b]
                     in evaluate(
            "connectx",
            [agent2, agent1], config, [], n_rounds - n_rounds // 2)]

    elif start_as == 0:
        outcomes = evaluate(
            "connectx", [agent1, agent2], config, [], n_rounds)

    elif start_as == 1:
        outcomes = [[b, a] for [a, b]
                    in evaluate(
            "connectx",
            [agent2, agent1], config, [], n_rounds)]

    else:
        print(f"start_as!! {start_as} not supported; use -1 | 0 | 1")
        return ""

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
    def __init__(self, adv_agent="random", min_reward=-10, max_reward=2):
        self.adv_agent = adv_agent
        self.agents = [None, self.adv_agent]
        self.max_reward = max_reward
        self.min_reward = min_reward
        ks_env = self._set_env()
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.new_shape = (1, self.rows, self.columns)
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=self.new_shape, dtype=np.int)
        self.reward_range = (min_reward, max_reward)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def _set_env(self, invert_agents=False):
        ks_env = make("connectx", debug=True)
        if invert_agents:
            self.agents = [self.agents[1], self.agents[0]]
        self.env = ks_env.train(self.agents)
        return ks_env

    def reset(self):
        self.obs = self.env.reset()
        self._set_env(invert_agents=True)
        return self.get_board()

    def get_board(self):
        board = np.array(self.obs['board']).reshape(*self.new_shape)

        if self.obs['mark'] == 2:
            with np.nditer(board, op_flags=['readwrite']) as it:
                for x in it:
                    if x == 2:
                        x[...] = 1
                    elif x == 1:
                        x[...] = 2
        return board

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return self.max_reward
        elif done:  # The opponent won the game
            return -self.max_reward
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = self.min_reward, True, {}
        return self.get_board(), reward, done, _


class AggroGym(ConnectFourGym):
    def __init__(self, adv_agent="random"):
        super().__init__(adv_agent=adv_agent, min_reward=-10, max_reward=2)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 2.7 - (self.obs['step'] * (1 / (self.rows * self.columns)))
        elif done:  # The opponent won the game
            return -self.max_reward
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


class AltAggroGym(ConnectFourGym):
    def __init__(self, adv_agent="random"):
        super().__init__(adv_agent=adv_agent, min_reward=-10, max_reward=2)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            min_moves = 6 + self.obs['mark']
            extra_moves = self.obs['step'] - min_moves
            penalty = extra_moves * 1 / 42
            return 2 - penalty
        elif done:  # The opponent won the game
            return -self.max_reward
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


class AggroAltBoard(ConnectFourGym):
    def __init__(self, adv_agent="random"):
        super().__init__(adv_agent=adv_agent, min_reward=-10, max_reward=2)

    def get_board(self):
        board = np.array(self.obs['board']).reshape(*self.new_shape)
        board = board * self.obs['mark']
        return board

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            min_moves = 6 + self.obs['mark']
            extra_moves = self.obs['step'] - min_moves
            penalty = extra_moves * 1 / 42
            return 2 - penalty
        elif done:  # The opponent won the game
            return -self.max_reward
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


class AggroSecondTrain(ConnectFourGym):
    def __init__(self, adv_agent="random"):
        super().__init__(adv_agent=adv_agent, min_reward=-10, max_reward=2)

    def _set_env(self):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([self.agents[1], self.agents[0]])
        return ks_env

    def reset(self):
        self.obs = self.env.reset()
        return self.get_board()

    def get_board(self):
        board = np.array(self.obs['board']).reshape(*self.new_shape)
        return board

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            min_moves = 6 + self.obs['mark']
            extra_moves = self.obs['step'] - min_moves
            penalty = extra_moves * 1 / 42
            return 2 - penalty
        elif done:  # The opponent won the game
            return -self.max_reward
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


class SiAgentGym(ConnectFourGym):
    def __init__(self, adv_agent="random"):
        super().__init__(adv_agent=adv_agent, min_reward=-10, max_reward=2)

    def _set_env(self):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([self.agents[1], self.agents[0]])
        return ks_env

    def reset(self):
        self.obs = self.env.reset()
        return self.get_board()

    def get_board(self):
        board = np.array(self.obs['board']).reshape(*self.new_shape)
        return board

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            min_moves = 6 + self.obs['mark']
            extra_moves = self.obs['step'] - min_moves
            penalty = extra_moves * 1 / 42
            return 2 - penalty
        elif done:  # The opponent won the game
            return -self.max_reward
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


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


def double_agent_factory(model1, model2):
    def agent(obs, config):
        # Use the best model to select a column
        print(obs['mark'])
        if obs['mark'] == 1:
            col, _ = model1.predict(np.array(obs['board']).reshape(1, 6, 7))
        else:
            col, _ = model2.predict(np.array(obs['board']).reshape(1, 6, 7))
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
