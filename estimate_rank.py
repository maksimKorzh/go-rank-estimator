'''
Estimate OGS/FOX rank for black and white player
'''

import argparse
import json
import subprocess
import time
import math
from threading import Thread
import sgfmill
import sgfmill.boards
import sgfmill.ascii_boards
from sgfmill import sgf
from typing import Tuple, List, Optional, Union, Literal, Any, Dict
import matplotlib
matplotlib.rcParams['toolbar'] = 'none'  # Disable toolbar before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from getpass import getpass
import re

Color = Union[Literal['b'],Literal['w']]
Move = Union[None,Literal['pass'],Tuple[int,int]]

def download_game():
  s = requests.Session()
  payload = {
      'userid': input('user: '),
      'password': getpass('password: '),
  }
  r = s.post('https://my.pandanet.co.jp/cgi-bin/cgi.exe?MH', data=payload)
  match = re.search(r'mypage\.php\?key=([A-Z0-9]+)', r.text)
  if match:
    key = match.group(1)
    print('Trying to download your last game from IGS...')
    response = requests.get('https://my.pandanet.co.jp/cgi-bin/cgi.exe?MHkey=' + key + '&pg=SearchResult')
    content = BeautifulSoup(response.text, 'lxml')
    game_url = [i for i in content.find_all('a') if 'SGF' in i.text][0]['href']
    game_sgf = requests.get(game_url).text
    with open('/home/cmk/go-rank-estimator/game.sgf', 'w') as f: f.write(game_sgf)
    print('Downloaded game')
  else:
    print('Login failed. Using local "game.sgf"')


def sgfmill_to_str(move: Move) -> str:
    if move is None:
        return 'pass'
    if move == 'pass':
        return 'pass'
    (y,x) = move
    return 'ABCDEFGHJKLMNOPQRSTUVWXYZ'[x] + str(y+1)

class KataGo:
    def __init__(self, katago_path: str, config_path: str, model_path: str, additional_args: List[str] = []):
        self.query_counter = 0
        katago = subprocess.Popen(
            [katago_path, 'analysis', '-config', config_path, '-model', model_path, *additional_args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.katago = katago
        def printforever():
            while katago.poll() is None:
                data = katago.stderr.readline()
                time.sleep(0)
                if data:
                    print('KataGo: ', data.decode(), end='')
            data = katago.stderr.read()
            if data:
                print('KataGo: ', data.decode(), end='')
        self.stderrthread = Thread(target=printforever)
        self.stderrthread.start()

    def close(self):
        self.katago.stdin.close()


    def query(self, initial_board: sgfmill.boards.Board, moves: List[Tuple[Color,Move]], komi: float, max_visits=None):
        query = {}

        query['id'] = str(self.query_counter)
        self.query_counter += 1

        query['moves'] = [(color,sgfmill_to_str(move)) for color, move in moves]
        query['initialStones'] = []
        for y in range(initial_board.side):
            for x in range(initial_board.side):
                color = initial_board.get(y,x)
                if color:
                    query['initialStones'].append((color,sgfmill_to_str((y,x))))
        query['rules'] = 'Chinese'
        query['komi'] = komi
        query['boardXSize'] = initial_board.side
        query['boardYSize'] = initial_board.side
        query['includePolicy'] = True
        if max_visits is not None:
            query['maxVisits'] = max_visits
        return self.query_raw(query)

    def query_raw(self, query: Dict[str,Any]):
        self.katago.stdin.write((json.dumps(query) + '\n').encode())
        self.katago.stdin.flush()

        # print(json.dumps(query))

        line = ''
        while line == '':
            if self.katago.poll():
                time.sleep(1)
                raise Exception('Unexpected katago exit')
            line = self.katago.stdout.readline()
            line = line.decode().strip()
            # print('Got: ' + line)
        response = json.loads(line)

        return response

def print_policy(policy):
    if policy == []: print('[]'); return
    print()
    for row in range(19):
        for col in range(19):
            sq = row*19+col
            if sq == policy.index(max(policy)): print('    BEST', end='')
            else: print('    ' + str(policy[sq])[:4] if 'e' not in str(policy[sq]) else '    -1.0', end='')
        print()
    print()

def draw_go_with_graph(stones, scores, black, white, final_label=None, board_size=19):
    fig, (ax_board, ax_scores, ax_perf) = plt.subplots(
        3, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [3, 1, 1]}
    )
    fig.canvas.manager.set_window_title('KataGo Rank Estimator')

    # -------------------- Draw Go board --------------------
    for i in range(board_size):
        ax_board.plot([0, board_size - 1], [i, i], color='black')
        ax_board.plot([i, i], [0, board_size - 1], color='black')

    # Star points
    def hoshi_points(size):
        return [3, 9, 15] if size == 19 else []

    stars = hoshi_points(board_size)
    for x in stars:
        for y in stars:
            ax_board.plot(x, y, 'ko', markersize=6)

    # Stones with move numbers
    for idx, (x, y, color) in enumerate(stones):
        if color == 'b':
            ax_board.plot(x, y, 'ko', markersize=18)
            text_color = 'white'
        elif color == 'w':
            ax_board.plot(x, y, 'o', markersize=18, markerfacecolor='white', markeredgecolor='black')
            text_color = 'black'
        else:
            continue

        try: ax_board.text(x, y, str(kifu[y, x]), color=text_color, fontsize=6, ha='center', va='center', fontweight='bold')
        except: pass

    ax_board.set_xlim(-0.5, board_size - 0.5)
    ax_board.set_ylim(-0.5, board_size - 0.5)
    ax_board.set_aspect('equal')
    ax_board.axis('off')
    ax_board.set_title(final_label, fontfamily='Monospace')

    # -------------------- Draw graph --------------------
    moves = np.arange(len(scores))
    offset = max(scores) + 10
    ax_perf.plot(np.arange(len(black)*2), [x for x in black for _ in range(2)], color='black', label='black performance (NN move number choice)')
    ax_perf.plot(np.arange(len(white)*2), [x for x in white for _ in range(2)], color='black', label='white performance (NN move number choice)', linestyle='--')
    ax_scores.plot(moves, scores, color='blue', label='score lead (territory points)')
    ax_scores.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_scores.grid(True)
    ax_scores.legend()
    ax_perf.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_perf.grid(True)
    ax_perf.legend()

    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space for bottom label
    plt.show()

def print_move(move):
    row = math.floor(move / 19)
    col = move % 19
    print('\nMove (index, row, col):', move, row+1, col+1)

def score_move(move, policy):
    if policy == []: return 0
    best_moves = sorted(policy, reverse=True)
    user_move_score = policy[move]
    return best_moves.index(user_move_score)+1

if __name__ == '__main__':
#    try:
#      if input('Download last cmk game from IGS? (y/n)') == 'y': download_game()
#    except: print('Failed. Probably you are not CMK. Using existing "game.sgf"')
    description = '''
    Example script showing how to run KataGo analysis engine and query it from python.
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-katago-path',
        help='Path to katago executable',
        required=True,
    )
    parser.add_argument(
        '-config-path',
        help='Path to KataGo analysis config (e.g. cpp/configs/analysis_example.cfg in KataGo repo)',
        required=True,
    )
    parser.add_argument(
        '-model-path',
        help='Path to neural network .bin.gz file',
        required=True,
    )
    parser.add_argument(
        '-sgf-file',
        help='Path to SGF file',
        required=True,
    )

    args = vars(parser.parse_args())
    print(args)

    with open(args['sgf_file'], 'rb') as f: game = sgf.Sgf_game.from_bytes(f.read())
    winner = game.get_winner()
    board_size = game.get_size()
    root_node = game.get_root()
    b_player = root_node.get('PB')
    w_player = root_node.get('PW')
    moves = []
    stones = []
    kifu = {}
    katago = KataGo(args['katago_path'], args['config_path'], args['model_path'])
    board = sgfmill.boards.Board(19)
    komi = 6.5
    prev_policy = []
    black_scores = []
    white_scores = []
    score_lead = []
    move_num = 1
    for node in game.get_main_sequence():
        if node.get_move()[0] is not None:
            moves.append(node.get_move())
            displayboard = board.copy()
            for color, move in moves:
                if move != 'pass':
                    try:
                        row,col = move
                        user_move = (18-row)*19+col
                        displayboard.play(row,col,color)
                    except: pass
            result = katago.query(board, moves, komi)
            score = score_move(user_move, prev_policy)
            score_lead.append(result['rootInfo']['scoreLead'])
            try:
              stones.append((node.get_move()[1][1], node.get_move()[1][0], node.get_move()[0]))
              kifu[node.get_move()[1][0], node.get_move()[1][1]] = move_num
            except: pass
            print('Move ' + str(move_num) + ' (' + node.get_move()[0] + '):\tNN #' + str(score_move(user_move, prev_policy)) + '\t\tScore: ' + str(result['rootInfo']['scoreLead']))
            move_num += 1
            if node.get_move()[0] == 'b': black_scores.append(score)
            elif node.get_move()[0] == 'w': white_scores.append(score)
            prev_policy = result['policy']
    black_performance = 0;
    white_performance = 0;
    for i in black_scores: black_performance += i
    for i in white_scores: white_performance += i
    black_performance = math.floor(black_performance / len(black_scores));
    white_performance = math.floor(white_performance / len(white_scores));
    black_rank = str((10-black_performance))+ 'd' if black_performance < 10 else str((black_performance - 9))+ 'k';
    white_rank = str((10-white_performance))+ 'd' if white_performance < 10 else str((white_performance - 9))+ 'k';
    score_lead = [v if i % 2 else -v for i, v in enumerate(score_lead)]
    final_label = b_player + ' [' + str(black_rank) + '] vs '+ w_player +' [' + str(white_rank) + '], '
    final_label += ('B+' + str(score_lead[-1])[0:4] if score_lead[-1]>0 else 'W+' + str(score_lead[-1])[1:5])
    draw_go_with_graph(stones, score_lead, black_scores, white_scores, final_label=final_label)
    katago.close()
