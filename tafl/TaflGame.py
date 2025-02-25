from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TaflLogic import Board
import numpy as np
from .GameVariants import *
from .Digits import int2base

class TaflGame(Game):

    def __init__(self, name):
        self.name = name
        self.get_init_board()

    def get_init_board(self):
        board=Board(Brandubh())
        if self.name == "Brandubh": board = Board(Brandubh())
        if self.name == "ArdRi": board = Board(ArdRi())
        if self.name == "Tablut": board = Board(Tablut())
        if self.name == "Tawlbwrdd": board = Board(Tawlbwrdd())
        if self.name == "Hnefatafl": board = Board(Hnefatafl())
        if self.name == "AleaEvangelii": board = Board(AleaEvangelii())
        self.n = board.size
        return board

    def get_board_size(self):
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions
        return self.n**4 

    def get_next_state(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = board.getCopy()
        move = int2base(action,self.n,4)
        b.execute_move(move, player)
        return b, -player

    def get_valid_moves(self, board, player):
        # return a fixed size binary vector
        # Note: Ignoring the passed in player variable since we are not inverting colors for getCanonicalForm and Arena calls with constant 1.
        valids = [0] * self.get_action_size()
        b = board.getCopy()
        legal_moves = b.get_legal_moves(board.getPlayerToMove())
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x1, y1, x2, y2 in legal_moves:
            valids[x1 + y1 * self.n + x2 * self.n ** 2 + y2 * self.n ** 3] = 1
        return np.array(valids)

    def get_game_ended(self, board, player):
        # return 0 if not ended, if player 1 won, -1 if player 1 lost
        return board.done*player

    def get_canonical_form(self, board, player):
        b = board.getCopy()
        # rules and objectives are different for the different players, so inverting board results in an invalid state.
        return b

    def get_symmetries(self, board, pi):
        return [(board, pi)]
        # mirror, rotational
        #assert(len(pi) == self.n**4)  
        #pi_board = np.reshape(pi[:-1], (self.n, self.n))
        #l = []

        #for i in range(1, 5):
        #    for j in [True, False]:
        #        newB = np.rot90(board, i)
        #        newPi = np.rot90(pi_board, i)
        #        if j:
        #            newB = np.fliplr(newB)
        #            newPi = np.fliplr(newPi)
        #        l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        #return l

    def string_representation(self, board):
        return str(board)

    def getScore(self, board, player):
        if board.done:
            return 1_000 * board.done * player
        return board.count_diff(player)



def display(board):
       render_chars = {
             "-1": "b",
              "0": " ",
              "1": "W",
              "2": "K",
             "10": "#",
             "12": "E",
             "20": "_",
             "22": "x",
       }
       print("---------------------")
       image=board.getImage()

       print("  ", " ".join(str(i) for i in range(len(image))))
       for i in range(len(image)-1,-1,-1):
           print("{:2}".format(i), end=" ")

           row=image[i]
           for col in row:
               c = render_chars[str(col)]
               sys.stdout.write(c)
           print(" ") 
       #if (board.done!=0): print("***** Done: ",board.done)  
       print("---------------------")


