from __future__ import print_function
import sys
from typing import Literal

sys.path.append('..')
from Game import Game
from .TicTacToeLogic import Board
import numpy as np

"""
Game class implementation for the game of 3D TicTacToe or Qubic.

Author: Adam Lawson, github.com/goshawk22
Date: Feb 05, 2020

Based on the TicTacToeGame by Evgeny Tyurin.
"""
class TicTacToeGame(Game):
    def __init__(self, n: int):
        self.n = n

    def get_init_board(self) -> np.ndarray:
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self) -> (int, int, int):
        # (a,b) tuple
        return self.n, self.n, self.n

    def get_action_size(self):
        # return number of actions
        return self.n * self.n * self.n + 1

    def get_next_state(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
        
        move = np.argwhere(boardvalues == action)[0]
        b.execute_move(move, player)
        return b.pieces, -player

    def get_valid_moves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.get_action_size()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legal_moves =  b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for z, x, y in legal_moves:
            boardvalues = np.arange(0,(self.n*self.n*self.n)).reshape(self.n,self.n,self.n)
            valids[boardvalues[z][x][y]] = 1
        return np.array(valids)

    def get_game_ended(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def get_canonical_form(self, board: np.ndarray, player: Literal[-1, 1]):
        # return state if player==1, else return -state if player==-1
        return player * board

    def get_symmetries(self, board, pi):
        # mirror, rotational
        pi_board = np.reshape(pi[:-1], (self.n, self.n, self.n))
        symmetries = []
        newB = np.reshape(board, (self.n*self.n, self.n))
        newPi = pi_board
        for i in range(1, 5):
            for z in [True, False]:
                for j in [True, False]:
                    if j:
                        newB = np.fliplr(newB)
                        newPi = np.fliplr(newPi)
                    if z:
                        newB = np.flipud(newB)
                        newPi = np.flipud(newPi)
                    
                    newB = np.reshape(newB, (self.n, self.n, self.n))
                    newPi = np.reshape(newPi, (self.n, self.n, self.n))
                    symmetries += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return symmetries

    def string_representation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]
        for z in range(n):
            print("   ", end="")
            for y in range(n):
                print (y,"", end="")
            print("")
            print("  ", end="")
            for _ in range(n):
                print ("-", end="-")
            print("--")
            for y in range(n):
                print(y, "|",end="")    # print the row #
                for x in range(n):
                    piece = board[z][y][x]    # get the piece to print
                    if piece == -1: print("X ",end="")
                    elif piece == 1: print("O ",end="")
                    else:
                        if x==n:
                            print("-",end="")
                        else:
                            print("- ",end="")
                print("|")

            print("  ", end="")
            for _ in range(n):
                print ("-", end="-")
            print("--")
