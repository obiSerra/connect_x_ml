import numpy as np
from torch import tensor

import logging
LOGGING_FILE = "debug.log"

logging.basicConfig(filename=LOGGING_FILE,
                    encoding='utf-8', level=logging.DEBUG)

def evaluate_line(line, mine=1, oth=2, neu=0):
    """Evaluate a line of 4 possible elements

    Args:
        line: array

    """

    mines = len([m for m in line if m == mine])
    neus = len([n for n in line if n == neu])
    oths = len([o for o in line if o == oth])

    if mines == 4:
        return 10e6
    elif oths == 4:
        return -10e6
    elif mines == 3 and neus == 1:
        return 1
    elif oths == 3 and neus == 1:
        return -100
    else:
        return 0


def point_lines(np_board_2d, start):
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
        np_board_2d: np.array 
        start: int, the index of the element to start

    """
    r, c = np.unravel_index(start, np_board_2d.shape)
    v_connect = np_board_2d[r:r + 4, c]
    h_connect = np_board_2d[r, c:c + 4]
    d_connect = np.diagonal(np_board_2d[r:r + 4, c:c + 4])
    dr_connect = np.diagonal(np.rot90(np_board_2d[r:r + 4, c:c + 4]))
    axes = [axe for axe in [v_connect, h_connect,
                            d_connect, dr_connect] if len(axe) == 4]
    return axes


def evaluate_board(np_board_2d, mine=1, oth=2, neu=0):
    """Calculate a value score for a given board.

    Args:
        np_board_2d: np.array with 6,7 shape

    Returns:
        int, the board score value

    """
    board_points = 0
    for i in range(np_board_2d.size):
        axes = point_lines(np_board_2d, i)
        points = sum([evaluate_line(a, mine=mine, oth=oth, neu=neu)
                     for a in axes])
        board_points += points
    return board_points


def simulate_move(board_2d, current_col, val=1):
    row_index = np.where(board_2d[:, current_col] == 0)[0][-1]
    board_2d[row_index, current_col] = val
    return board_2d


def all_moves(board, configuration, val=1):
    values = []
    for col in range(configuration.columns):
        if len([c for c in board[:, col] if c == 0]) > 0:
            board_copy = np.copy(board)
            new_board = simulate_move(board_copy, col, val)

            values.append((col, new_board))
    return values


def minmax(next_possibilities):

    # Optimization: discard all boards with negative values
    next_evaluated = []
    for (col, new_board) in next_possibilities:
        value = evaluate_board(new_board)
        next_evaluated.append((col, new_board, value))
    next_evaluated.sort(key=lambda x: x[-1], reverse=True)

    return next_evaluated


def look_ahead(board, configuration, depth, current=0, actions=[],
               mine=1, oth=2, memo={}):

    if current >= depth:
        return [actions]

    # hash = hash_board(board, mine)
    # if hash in memo:
    #     next_possibilities = memo[hash]
    # else:
    next_possibilities = all_moves(board, configuration, val=mine)
    # memo[hash] = next_possibilities

    futures = []

    # for (col, new_board, board_value) in next_evaluated[:10]:
    for (col, new_board, board_value) in minmax(next_possibilities):
        look_result = []
        look_result = look_ahead(new_board, configuration, depth,
                                 current=current + 1,
                                 actions=actions +
                                 [{'col': col, 'board': new_board,
                                     'value': board_value}],
                                 mine=oth, oth=mine, memo=memo)
        futures += look_result
    return futures


def select_bests(possibilities):
    bests = []
    best_val = -10e100
    for strategy in possibilities:
        s = strategy[-1]
        if s['value'] > best_val:
            best_val = s['value']
            bests = [[s['col'] for s in strategy]]
        elif s['value'] == best_val:
            bests.append([s['col'] for s in strategy])
    return bests


def multistep_agent_factory(move_predict=1):
    def multistep_agent(obs, config):
        try:
            from random import choice
            step_predict = move_predict
            current_board = obs.board

            # obs = tensor(obs["board"]).reshape(
            #     1, 1, config.rows, config.columns).float()

            board_2d = np.array(current_board).reshape(
                config.rows, config.columns)

            possibilities = look_ahead(board_2d, config, step_predict)
            bests = select_bests(possibilities)
        #    if obs.board[int(col)] == 0
            valid_moves = [b[0]
                        for b in bests if obs.board[int(b[0])] == 0]
            return choice(valid_moves)
        except Exception as e:
            logging.error(e)
    return multistep_agent
