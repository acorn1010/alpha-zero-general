import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .Connect4Logic import Board


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        Game.__init__(self)
        self._base_board = Board(height, width, win_length, np_pieces)

    def get_init_board(self):
        return self._base_board.np_pieces

    def get_board_size(self):
        return (self._base_board.height, self._base_board.width)

    def get_action_size(self):
        return self._base_board.width

    def get_next_state(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def get_valid_moves(self, board, player):
        """Any zero value in top row in a valid move"""
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def get_game_ended(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def get_canonical_form(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def get_symmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def string_representation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(' '.join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")
