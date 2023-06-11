import os
import sys
import numpy as np
from flask import Flask, request, Response

sys.path.append("..")

from MCTS import MCTS, MCTSArgs

from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.keras.NNet import NNetWrapper
from dotsandboxes.DotsAndBoxesPlayers import GreedyRandomPlayer
from utils import dotdict

app = Flask(__name__)

mcts = None
g = None


# curl -d "board=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -X POST http://localhost:8888/predict
@app.route('/predict', methods=['POST'])
def predict():
    board = np.fromstring(request.form['board'], sep=',').reshape(g.get_board_size())

    use_alpha_zero = True
    if use_alpha_zero:
        action = np.argmax(mcts.get_action_prob(board, temp=0))
    else:
        action = GreedyRandomPlayer(g).play(board)

    resp = Response(str(action))
    # https://stackoverflow.com/questions/5584923/a-cors-post-request-works-from-plain-javascript-but-why-not-with-jquery
    # https://stackoverflow.com/questions/25860304/how-do-i-set-response-headers-in-flask
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    g = DotsAndBoxesGame(n=3)
    n1 = NNetWrapper(g)
    mcts = MCTS(g, n1, MCTSArgs(simulation_count=50, cpu_count=1.0))
    n1.load_checkpoint(os.path.join('..', 'pretrained_models', 'dotsandboxes', 'keras', '3x3'), 'best.pth.tar')
    app.run(debug=False, host='0.0.0.0', port=8888)
