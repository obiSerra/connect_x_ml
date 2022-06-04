import numpy as np


class SimulatedBoard():
    def __init__(self, board, steps=[], mine=1, oth=2, neu=0):
        self.board = board
        self.done = False
        self.score = None
        self.steps = steps
        self.mine = mine
        self.oth = oth
        self.neu = neu
        self._evaluate_board(mine=mine, oth=oth, neu=neu)

    def to_dict(self):
        return {'board': self.board,
                'steps': self.steps,
                'score': self.score}

    def _point_lines(self, start):
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
        r, c = np.unravel_index(start, self.board.shape)
        v_connect = self.board[r:r + 4, c]
        h_connect = self.board[r, c:c + 4]
        d_connect = np.diagonal(self.board[r:r + 4, c:c + 4])
        dr_connect = np.diagonal(np.rot90(self.board[r:r + 4, c:c + 4]))
        axes = [axe for axe in [v_connect, h_connect,
                                d_connect, dr_connect] if len(axe) == 4]
        return axes

    def _evaluate_board(self, mine=1, oth=2, neu=0):
        """Calculate a value score for a given board.

        Args:
            self.board: np.array with 6,7 shape

        Returns:
            int, the board score value

        """
        score = 0
        for i in range(self.board.size):
            axes = self._point_lines(i)
            lines = [self._evaluate_line(a, mine=mine, oth=oth, neu=neu)
                     for a in axes]

            if 10e6 in lines:
                self.done = True
                self.score = 10e6
                return score
            elif -10e6 in lines:
                self.done = True
                self.score = -10e6
                return score
            else:
                points = sum(lines)
                score += points
        self.score = score

    def _evaluate_line(self, line, mine=1, oth=2, neu=0):
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
            return 10
        elif oths == 3 and neus == 1:
            return -100
        else:
            return 0


class TreeEval():
    def __init__(self, rows=6, cols=7):
        self.cols = cols
        self.rows = rows
        self.new_shape = (self.rows, self.cols)
        self.to_keep = 4

    def _reshape_board(self, obs_board):
        return np.array(obs_board).reshape(*self.new_shape)

    def _possible_moves(self, board):
        z_rows, z_cols = np.where(board == 0)
        possible_pos = {}
        for i in range(len(z_cols)):
            possible_pos[str(z_cols[i])] = [z_cols[i], z_rows[i]]
        return possible_pos.values()

    def _remove_duplicates(self, boards):
        return list({str(b.board): b for b in boards}.values())

    def simulate_multiple_steps(self, obs_board, own_mark, adv_mark,
                                steps=1, min_max=True):
        board = self._reshape_board(obs_board)

        i = 0
        boards = [SimulatedBoard(board, mine=own_mark, oth=adv_mark)]
        marks = [own_mark, adv_mark]

        while i < steps:
            res = []

            for b in boards:
                if not b.done:
                    res += self.simulate_next_step(b, marks[0])
                else:
                    res.append(b)
            marks = [marks[1], marks[0]]
            i += 1

            if min_max is True:
                scored_boards = self._remove_duplicates(res)

                scored_boards.sort(key=lambda x: x.score)
                keep = [s for s in scored_boards[-self.to_keep:]]

                boards = keep
            else:
                boards = self._remove_duplicates(res)

        return [SimulatedBoard(b.board, steps=b.steps, mine=own_mark, oth=adv_mark) for b in boards]

    def simulate_next_step(self, board, mark):
        moves = self._possible_moves(board.board)
        new_boards = []
        for m in moves:
            b = board.board.copy()
            b[m[1], m[0]] = mark
            new_boards.append(SimulatedBoard(
                b, steps=[*board.steps, m],
                mine=board.mine if board.mine == mark else board.oth,
                oth=board.oth if board.mine == mark else board.mine))
        return new_boards


# @ pytest.fixture
# def mock_board():
#     return [0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0,
#             0, 0, 1, 2, 1, 0, 0,
#             0, 0, 2, 2, 2, 1, 2,
#             2, 1, 1, 1, 2, 1, 1]


# @ pytest.fixture
# def obs_mock(mock_board):
#     return {'remainingOverageTime': 60,
#             'mark': 2,
#             'step': 15,
#             'board': mock_board}


# @ pytest.mark.parametrize("test_input, expected", [([0, 0, 0, 0, 0, 0, 0,
#                                                    0, 0, 0, 0, 0, 0, 0,
#                                                    0, 0, 0, 0, 0, 0, 0,
#                                                    0, 0, 1, 2, 1, 0, 0,
#                                                    0, 0, 2, 2, 2, 1, 2,
#                                                    2, 1, 1, 1, 2, 1, 1], [[0, 4], [1, 4], [2, 2], [3, 2], [4, 2], [5, 3], [6, 3]]),
#                                                    ([0, 0, 0, 0, 0, 0, 0,
#                                                      0, 0, 0, 0, 0, 0, 0,
#                                                      0, 0, 0, 0, 0, 0, 0,
#                                                      0, 0, 0, 2, 1, 0, 0,
#                                                      0, 0, 0, 2, 2, 1, 2,
#                                                      0, 1, 0, 1, 2, 1, 1], [[0, 5], [1, 4], [2, 5], [3, 2], [4, 2], [5, 3], [6, 3]]),
#                                                    ([0, 0, 0, 1, 0, 0, 0,
#                                                      0, 0, 0, 2, 0, 0, 0,
#                                                      0, 0, 0, 1, 0, 0, 0,
#                                                      0, 0, 0, 2, 1, 0, 0,
#                                                      0, 0, 0, 2, 2, 1, 2,
#                                                      0, 1, 0, 1, 2, 1, 1], [[0, 5], [1, 4], [2, 5], [4, 2], [5, 3], [6, 3]])
#                                                    ]
#                           )
# def test_valid_cols(test_input, expected):
#     ev = TreeEval()
#     board = test_input

#     result = ev._possible_moves(ev._reshape_board(board))
#     assert list(result) == expected


# # def test_simulate(obs_mock):
# #     ev = TreeEval()
# #     board = obs_mock['board']

# #     mark = obs_mock['mark']
# #     result = ev.simulate_next_step(ev._reshape_board(board), mark)
# #     assert len(result) == 7


# # def test_simulate_multi(obs_mock):
# #     ev = TreeEval()
# #     board = obs_mock['board']

# #     mark = obs_mock['mark']
# #     result = ev.simulate_multiple_steps(board, mark, 1, steps=1, min_max=False)
# #     assert str(result) == str(
# #         ev.simulate_next_step(ev._reshape_board(board), mark))


# def test_simulate_multi_count(obs_mock):
#     ev = TreeEval()
#     board = obs_mock['board']

#     mark = obs_mock['mark']
#     result = ev.simulate_multiple_steps(
#         board, mark, 1, steps=4, min_max=False)

#     assert len(result) == 834


# def test_simulate_multi_min_max(obs_mock):
#     ev = TreeEval()
#     board = obs_mock['board']

#     mark = obs_mock['mark']
#     result = ev.simulate_multiple_steps(
#         board, mark, 1, steps=4, min_max=True)
#     result_all = ev.simulate_multiple_steps(
#         board, mark, 1, steps=4, min_max=False)

#     scores = result_all
#     scores.sort(key=lambda x: x.score)
#     print([r.to_dict() for r in result])

#     assert len(result) == 5


# # def test_eval(obs_mock):
# #     ev = TreeEval()
# #     board = [0, 0, 0, 0, 0, 0, 0,
# #              0, 0, 0, 0, 0, 0, 0,
# #              0, 0, 0, 0, 0, 0, 0,
# #              0, 0, 1, 2, 1, 0, 0,
# #              0, 0, 2, 2, 2, 1, 2,
# #              2, 1, 1, 1, 2, 1, 1]

# #     result = ev._evaluate_board(ev._reshape_board(board), mine=2, oth=1)
# #     print(result)
# #     assert result == -90


# if __name__ == '__main__':
#     ev = TreeEval()
#     ev.simulate_next_step()
#     print()
