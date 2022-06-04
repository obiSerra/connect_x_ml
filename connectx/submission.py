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
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc3 = nn.Linear(384, 512)
            self.shared1 = nn.Linear(512, 64)
            self.policy1 = nn.Linear(64, 32)
            self.policy2 = nn.Linear(32, 16)
            self.action = nn.Linear(16, 7)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = nn.Flatten()(x)
            x = F.relu(self.fc3(x))
            x = F.relu(self.shared1(x))
            x = F.relu(self.policy1(x))
            x = F.relu(self.policy2(x))
            x = self.action(x)
            x = x.argmax()
            return x

    from connectx.state_dictionary import state_dict
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
