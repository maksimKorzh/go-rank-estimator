"""
This is a simple python program that demonstrates how to run KataGo's
analysis engine as a subprocess and send it a query. It queries the
result of playing the 4-4 point on an empty board and prints out
the json response.
"""

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

Color = Union[Literal["b"],Literal["w"]]
Move = Union[None,Literal["pass"],Tuple[int,int]]

def sgfmill_to_str(move: Move) -> str:
    if move is None:
        return "pass"
    if move == "pass":
        return "pass"
    (y,x) = move
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)

class KataGo:

    def __init__(self, katago_path: str, config_path: str, model_path: str, additional_args: List[str] = []):
        self.query_counter = 0
        katago = subprocess.Popen(
            [katago_path, "analysis", "-config", config_path, "-model", model_path, *additional_args],
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
                    print("KataGo: ", data.decode(), end="")
            data = katago.stderr.read()
            if data:
                print("KataGo: ", data.decode(), end="")
        self.stderrthread = Thread(target=printforever)
        self.stderrthread.start()

    def close(self):
        self.katago.stdin.close()


    def query(self, initial_board: sgfmill.boards.Board, moves: List[Tuple[Color,Move]], komi: float, max_visits=None):
        query = {}

        query["id"] = str(self.query_counter)
        self.query_counter += 1

        query["moves"] = [(color,sgfmill_to_str(move)) for color, move in moves]
        query["initialStones"] = []
        for y in range(initial_board.side):
            for x in range(initial_board.side):
                color = initial_board.get(y,x)
                if color:
                    query["initialStones"].append((color,sgfmill_to_str((y,x))))
        query["rules"] = "Chinese"
        query["komi"] = komi
        query["boardXSize"] = initial_board.side
        query["boardYSize"] = initial_board.side
        query["includePolicy"] = True
        if max_visits is not None:
            query["maxVisits"] = max_visits
        return self.query_raw(query)

    def query_raw(self, query: Dict[str,Any]):
        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        self.katago.stdin.flush()

        # print(json.dumps(query))

        line = ""
        while line == "":
            if self.katago.poll():
                time.sleep(1)
                raise Exception("Unexpected katago exit")
            line = self.katago.stdout.readline()
            line = line.decode().strip()
            # print("Got: " + line)
        response = json.loads(line)

        # print(response)
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

def print_board_array(board):
    print()
    for row in range(19):
        for col in range(19):
            sq = (18-row)*19+col
            if board[sq] == None: print(' .', end='')
            else: print(' ' + board[sq], end='')
        print()
    print()

def print_move(move):
    row = math.floor(move / 19)
    col = move % 19
    print('\nMove (index, row, col):', move, row+1, col+1)

def score_move(move, policy):
    if policy == []: return 0
    best_moves = sorted(policy, reverse=True)
    user_move_score = policy[move]
    return best_moves.index(user_move_score)+1

if __name__ == "__main__":
    description = """
    Example script showing how to run KataGo analysis engine and query it from python.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-katago-path",
        help="Path to katago executable",
        required=True,
    )
    parser.add_argument(
        "-config-path",
        help="Path to KataGo analysis config (e.g. cpp/configs/analysis_example.cfg in KataGo repo)",
        required=True,
    )
    parser.add_argument(
        "-model-path",
        help="Path to neural network .bin.gz file",
        required=True,
    )
    parser.add_argument(
        "-sgf-file",
        help="Path to SGF file",
        required=True,
    )

    args = vars(parser.parse_args())
    print(args)

    with open(args["sgf_file"], "rb") as f: game = sgf.Sgf_game.from_bytes(f.read())
    winner = game.get_winner()
    board_size = game.get_size()
    root_node = game.get_root()
    b_player = root_node.get("PB")
    w_player = root_node.get("PW")
    moves = []
    katago = KataGo(args["katago_path"], args["config_path"], args["model_path"])
    board = sgfmill.boards.Board(19)
    komi = 6.5
    prev_policy = []
    black_scores = []
    white_scores = []
    for node in game.get_main_sequence()[:-2]:
        if node.get_move()[0] is not None:
            moves.append(node.get_move())
            displayboard = board.copy()
            for color, move in moves:
                if move != "pass":
                    row,col = move
                    user_move = (18-row)*19+col
                    displayboard.play(row,col,color)
            result = katago.query(board, moves, komi)
            score = score_move(user_move, prev_policy)
            #print_move(user_move)
            
            #print('NN choice:', score_move(user_move, prev_policy))
            if node.get_move()[0] == 'b': black_scores.append(score)
            elif node.get_move()[0] == 'w': white_scores.append(score)




            #board_array = []
            #[[board_array.append(y) for y in i] for i in displayboard.__dict__["board"]]
            #print_board_array(board_array)
            #print(sgfmill.ascii_boards.render_board(displayboard))
            #print_policy(prev_policy)
            prev_policy = result["policy"]
    katago.close()

'''
We don't care about displayed coords, we care about actual indices.
So for the first move coords are row 15, col 16, starting from top left.
'''
