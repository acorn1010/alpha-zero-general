from typing import Literal

import numpy as np


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        pass

    def get_init_board(self):  # -> np.ndarray:
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def get_board_size(self) -> (int, int):
        """
        Returns:
            (x, y): a tuple of board dimensions
        """
        pass

    def get_action_size(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def get_next_state(
            self, board: np.ndarray, player: Literal[-1, 1], action)\
            -> (np.ndarray, Literal[-1, 1]):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def get_valid_moves(self, board: np.ndarray, player: Literal[-1, 1]) -> np.array:
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def get_game_ended(self, board: np.ndarray, player: Literal[-1, 1]) -> float:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def get_canonical_form(self, board: np.ndarray, player: Literal[-1, 1]) -> np.ndarray:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def get_symmetries(self, board: np.ndarray, pi: list[float]) -> list[tuple[np.ndarray, list[float]]]:
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def string_representation(self, board: np.ndarray) -> str:
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
