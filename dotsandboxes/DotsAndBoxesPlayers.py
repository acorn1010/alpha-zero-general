import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.get_action_size())
        return a


# Will play at random, unless there's a chance to score a square
class GreedyRandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        previous_score = board[0, -1]
        for action in np.nonzero(valids)[0]:
            new_board, _ = self.game.get_next_state(board, 1, action)
            new_score = new_board[0, -1]
            if new_score > previous_score:
                return action
        a = np.random.randint(self.game.get_action_size())
        while valids[a]!=1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanDotsAndBoxesPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        if board[2][-1] == 1:
            # We have to pass
            return self.game.get_action_size() - 1
        valids = self.game.get_valid_moves(board, 1)
        while True:
            print("Valid moves: {}".format(np.where(valids == True)[0]))
            a = int(input())
            if valids[a]:
                return a
            print('Invalid move')
