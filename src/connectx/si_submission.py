def agent(obs, config):
    import sys
    import os
    import importlib
    import numpy as np
    import torch as th
    from torch import nn as nn
    import torch.nn.functional as F
    from torch import tensor
    import random

    import logging
    logging.basicConfig(filename='eval.log',
                        encoding='utf-8', level=logging.DEBUG)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 7, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(294, 140)
            self.fc2 = nn.Linear(140, 42)
            self.policy1 = nn.Linear(42, 42)
            self.policy2 = nn.Linear(42, 42)
            self.policy3 = nn.Linear(42, 42)
            self.action = nn.Linear(42, 7)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = nn.Flatten()(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.policy1(x))
            x = F.relu(self.policy2(x))
            x = F.relu(self.policy3(x))
            x = self.action(x)
            x = x.argmax()
            return x

    from connectx.si_model_state_dictionary import state_dict
    model = Net()
    model = model.float()
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model = model.eval()
    obs = tensor(obs["board"]).reshape(
        1, 1, config.rows, config.columns).float()

    obs = obs / 2
    board_2d = obs.reshape(config.rows, config.columns)

    # logging.info("board")
    #
    action = model(obs)
    # logging.info(f"action {action}")

    is_valid = any(board_2d[:, int(action)] == 0)

    if not is_valid:
        moves = []
        for c in range(1, config.columns):

            if any(board_2d[:, c] == 0):
                moves.append(c)
        logging.info("board")
        logging.info(board_2d)
        logging.info("mod")
        logging.info(moves)
        logging.info(f"action {action}")
        return random.choice(moves)
    return int(action)
