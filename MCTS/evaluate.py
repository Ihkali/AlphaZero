"""
MCTS/evaluate.py — Arena evaluation: pit two models against each other.
"""

import chess
import torch
import numpy as np

from MCTS.config import Config
from MCTS.encode import encode_board, index_to_move
from MCTS.mcts import mcts_search, select_action


def evaluate_models(new_net, best_net, num_games=Config.eval_games,
                    num_sims=Config.eval_mcts_sims, device="cpu", verbose=True):
    new_net.eval()
    best_net.eval()
    new_wins = 0
    best_wins = 0
    draws = 0

    for g in range(num_games):
        new_is_white = (g % 2 == 0)
        result = _play_eval_game(
            white_net=new_net if new_is_white else best_net,
            black_net=best_net if new_is_white else new_net,
            num_sims=num_sims, device=device,
        )
        if result == 0:
            draws += 1
            result_str = "Draw"
        elif (result == 1 and new_is_white) or (result == -1 and not new_is_white):
            new_wins += 1
            result_str = "New wins"
        else:
            best_wins += 1
            result_str = "Best wins"

        if verbose:
            color = "W" if new_is_white else "B"
            print(f"  Eval game {g+1}/{num_games} (new={color}): {result_str}")

    total = num_games
    win_rate = (new_wins + 0.5 * draws) / total
    accepted = win_rate > Config.accept_threshold

    result = {
        "new_wins": new_wins, "best_wins": best_wins, "draws": draws,
        "win_rate": win_rate, "accepted": accepted,
    }
    if verbose:
        print(f"\n  Evaluation: new={new_wins} best={best_wins} draws={draws}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
    return result


def _play_eval_game(white_net, black_net, num_sims, device,
                    max_moves=Config.max_game_moves):
    board = chess.Board()
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        net = white_net if board.turn == chess.WHITE else black_net
        policy, _ = mcts_search(
            board, net, num_simulations=num_sims,
            temperature=0.0, dirichlet_epsilon=0.0, device=device,
        )
        action = select_action(policy, temperature=0)
        move = index_to_move(action, board)
        if move not in board.legal_moves:
            move = list(board.legal_moves)[0]
        board.push(move)
        move_count += 1

    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1
    return 0
