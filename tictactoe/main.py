#!/usr/bin/env python3
"""
tictactoe/main.py — Train & test MCTS on Tic-Tac-Toe.

Three modes:
  1. Self-play + training loop  (default)
  2. Watch a trained model play itself  (--watch)
  3. Play against the trained AI         (--play)

Usage:
    python -m tictactoe.main                         # train from scratch
    python -m tictactoe.main --iterations 50         # more training
    python -m tictactoe.main --watch                 # watch AI vs AI
    python -m tictactoe.main --play                  # you vs AI
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tictactoe.game import TicTacToe
from tictactoe.model import TicTacToeNet
from tictactoe.mcts import mcts_search, select_action


CHECKPOINT_PATH = "tictactoe/model.pt"


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values directly
# ═══════════════════════════════════════════════════════════════════════════

MODE            = "train"     # "train", "watch", or "play"
ITERATIONS      = 100          # training iterations
GAMES_PER_ITER  = 50          # self-play games per iteration
SIMULATIONS     = 100          # MCTS simulations per move (self-play)
EVAL_SIMS       = 200          # MCTS simulations for evaluation
EVAL_GAMES      = 100          # evaluation games vs random
LEARNING_RATE   = 1e-3
EPOCHS          = 10           # training epochs per iteration
REPLAY_WINDOW   = 5            # keep examples from last N iterations
SHOW_GAME_EVERY = 10           # show a demo AI-vs-AI game every N iters (0=off)
DEVICE          = "cpu"


# ═══════════════════════════════════════════════════════════════════════════
#  SELF-PLAY: generate training data
# ═══════════════════════════════════════════════════════════════════════════

# ── Symmetry augmentation (8-fold: 4 rotations x 2 reflections) ───────
_SYMMETRIES = []
for _k in range(4):                       # 0, 90, 180, 270
    for _flip in (False, True):            # identity, horizontal flip
        _SYMMETRIES.append((_k, _flip))

def _apply_sym_board(planes: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply rotation + flip to a (3,3,3) encoded board."""
    p = np.rot90(planes, k=k, axes=(1, 2)).copy()
    if flip:
        p = np.flip(p, axis=2).copy()
    return p

def _apply_sym_policy(policy: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply the same rotation + flip to a flat policy[9]."""
    p = policy.reshape(3, 3)
    p = np.rot90(p, k=k)
    if flip:
        p = np.flip(p, axis=1)
    return p.flatten().copy()

def _augment_examples(examples):
    """Apply all 8 symmetries to a list of (state, policy, value)."""
    augmented = []
    for state, policy, value in examples:
        for k, flip in _SYMMETRIES:
            s = _apply_sym_board(state, k, flip)
            p = _apply_sym_policy(policy, k, flip)
            augmented.append((s, p, value))
    return augmented


def self_play_game(net, num_sims=100, temperature=1.0, temp_threshold=6,
                   device="cpu"):
    """Play one game with MCTS. Returns list of (state, policy, value)."""
    game = TicTacToe()
    history = []  # (encoded_state, mcts_policy, current_player)

    while not game.is_game_over():
        temp = temperature if len(game.move_stack) < temp_threshold else 0.0
        policy, _ = mcts_search(
            game, net,
            num_simulations=num_sims,
            temperature=temp,
            device=device,
        )
        history.append((game.encode(), policy.copy(), game.current_player))

        action = select_action(policy, temp)
        game.push(action)

    # Assign values based on game result
    result = game.result()  # +1 X wins, -1 O wins, 0 draw
    examples = []
    for state, policy, player in history:
        if result == 0:
            value = 0.0
        elif result == player:
            value = 1.0
        else:
            value = -1.0
        examples.append((state, policy, value))

    return examples, result


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_on_examples(net, optimizer, examples, epochs=5, device="cpu",
                      batch_size=256):
    """Train the network on collected self-play examples."""
    if not examples:
        return 0.0

    states = torch.tensor(np.array([e[0] for e in examples]),
                          dtype=torch.float32).to(device)
    target_policies = torch.tensor(np.array([e[1] for e in examples]),
                                   dtype=torch.float32).to(device)
    target_values = torch.tensor(np.array([e[2] for e in examples]),
                                 dtype=torch.float32).unsqueeze(1).to(device)

    net.train()
    total_loss = 0.0
    n_batches = 0
    n = len(examples)

    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            s_b = states[idx]
            p_b = target_policies[idx]
            v_b = target_values[idx]

            p_logits, v = net(s_b)

            # Policy loss: cross-entropy with MCTS policy
            log_probs = F.log_softmax(p_logits, dim=1)
            policy_loss = -torch.sum(p_b * log_probs) / len(idx)

            # Value loss: MSE
            value_loss = F.mse_loss(v, v_b)

            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION: model plays itself (AI vs AI self-play)
# ═══════════════════════════════════════════════════════════════════════════

def play_self_eval(net, num_games=100, num_sims=200, device="cpu"):
    """Model plays itself. Returns (x_wins, o_wins, draws)."""
    x_wins, o_wins, draws = 0, 0, 0

    for g in range(num_games):
        game = TicTacToe()

        while not game.is_game_over():
            policy, _ = mcts_search(
                game, net, num_simulations=num_sims,
                temperature=0.0, device=device,
                dirichlet_epsilon=0.0,
            )
            action = select_action(policy, 0.0)
            game.push(action)

        result = game.result()
        if result == 1:
            x_wins += 1
        elif result == -1:
            o_wins += 1
        else:
            draws += 1

    return x_wins, o_wins, draws


# ═══════════════════════════════════════════════════════════════════════════
#  INTERACTIVE: play against AI or watch AI vs AI
# ═══════════════════════════════════════════════════════════════════════════

def watch_game(net, num_sims=200, device="cpu"):
    """Watch the AI play itself with MCTS."""
    game = TicTacToe()
    move_num = 0
    print("\n  ╔═══════════════════════════╗")
    print("  ║   AI vs AI  (MCTS Watch)  ║")
    print("  ╚═══════════════════════════╝\n")

    while not game.is_game_over():
        player = "X" if game.current_player == 1 else "O"
        policy, value = mcts_search(
            game, net, num_simulations=num_sims,
            temperature=0.1, device=device,
            dirichlet_epsilon=0.0,
        )
        action = select_action(policy, 0.0)

        print(f"  Move {move_num + 1} — {player} plays cell {action} "
              f"(value: {value:+.3f})")
        print(f"  Policy: {_fmt_policy(policy)}")
        game.push(action)
        print(f"\n{_board_display(game)}\n")
        move_num += 1

    result = game.result()
    if result == 1:
        print("  Result: X wins!")
    elif result == -1:
        print("  Result: O wins!")
    else:
        print("  Result: Draw")


def show_training_game(net, num_sims=200, device="cpu", iteration=0):
    """Compact AI-vs-AI game display for mid-training progress."""
    game = TicTacToe()
    moves = []
    while not game.is_game_over():
        policy, value = mcts_search(
            game, net, num_simulations=num_sims,
            temperature=0.1, device=device,
            dirichlet_epsilon=0.0,
        )
        action = select_action(policy, 0.0)
        player = "X" if game.current_player == 1 else "O"
        moves.append(f"{player}:{action}")
        game.push(action)

    result = game.result()
    tag = {1: "X wins", -1: "O wins", 0: "Draw"}[result]
    symbols = {0: "·", 1: "X", -1: "O"}
    b = game.board
    board_line = (f"  {symbols[b[0]]} {symbols[b[1]]} {symbols[b[2]]}  "
                 f"{symbols[b[3]]} {symbols[b[4]]} {symbols[b[5]]}  "
                 f"{symbols[b[6]]} {symbols[b[7]]} {symbols[b[8]]}")
    print(f"\n  ── Demo game (iter {iteration}) ──")
    print(f"  Moves: {' '.join(moves)}")
    print(f"{board_line}  → {tag}\n")


def play_human(net, num_sims=200, device="cpu"):
    """Play against the MCTS AI."""
    print("\n  ╔═══════════════════════════╗")
    print("  ║   You vs AI  (MCTS)       ║")
    print("  ╚═══════════════════════════╝")
    print("  You are X (first). Enter cell 0-8:")
    print("  Board layout:")
    print("    0 | 1 | 2")
    print("    ---------")
    print("    3 | 4 | 5")
    print("    ---------")
    print("    6 | 7 | 8\n")

    game = TicTacToe()
    human_player = 1       # X

    while not game.is_game_over():
        print(f"{_board_display(game)}\n")

        if game.current_player == human_player:
            while True:
                try:
                    action = int(input("  Your move (0-8): "))
                    if action in game.legal_moves:
                        break
                    print("  Illegal move, try again.")
                except (ValueError, EOFError):
                    print("  Enter a number 0-8.")
        else:
            policy, value = mcts_search(
                game, net, num_simulations=num_sims,
                temperature=0.0, device=device,
                dirichlet_epsilon=0.0,
            )
            action = select_action(policy, 0.0)
            print(f"  AI plays cell {action} (value: {value:+.3f})")
            print(f"  Policy: {_fmt_policy(policy)}")

        game.push(action)

    print(f"\n{_board_display(game)}")
    result = game.result()
    if result == human_player:
        print("\n  You win!")
    elif result == -human_player:
        print("\n  AI wins!")
    else:
        print("\n  It's a draw!")


def _board_display(game):
    symbols = {0: "·", 1: "X", -1: "O"}
    lines = ["  ┌───┬───┬───┐"]
    for r in range(3):
        cells = " │ ".join(symbols[game.board[r * 3 + c]] for c in range(3))
        lines.append(f"  │ {cells} │")
        if r < 2:
            lines.append("  ├───┼───┼───┤")
    lines.append("  └───┴───┴───┘")
    return "\n".join(lines)


def _fmt_policy(policy):
    return "[" + " ".join(f"{p:.2f}" for p in policy) + "]"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = DEVICE
    net = TicTacToeNet().to(device)

    # Load existing checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        data = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        net.load_state_dict(data["model_state"])
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # ── Watch mode ────────────────────────────────────────────────────
    if MODE == "watch":
        watch_game(net, num_sims=SIMULATIONS, device=device)
        return

    # ── Play mode ─────────────────────────────────────────────────────
    if MODE == "play":
        play_human(net, num_sims=SIMULATIONS, device=device)
        return

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n{'═' * 56}")
    print(f"  MCTS TIC-TAC-TOE TRAINING")
    print(f"  {ITERATIONS} iters × {GAMES_PER_ITER} games × "
          f"{SIMULATIONS} sims/move")
    print(f"{'═' * 56}\n")

    best_winrate = 0.0
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    replay_buffer = []          # list of augmented example batches

    for iteration in range(1, ITERATIONS + 1):
        t0 = time.time()

        # ── Self-play ─────────────────────────────────────────────────
        iter_examples = []
        results = {1: 0, -1: 0, 0: 0}

        for g in range(GAMES_PER_ITER):
            examples, result = self_play_game(
                net, num_sims=SIMULATIONS,
                temperature=1.0, temp_threshold=6, device=device,
            )
            iter_examples.extend(examples)
            results[result] += 1

        # Augment with 8-fold symmetries
        augmented = _augment_examples(iter_examples)

        # Replay buffer: keep last REPLAY_WINDOW iterations
        replay_buffer.append(augmented)
        if len(replay_buffer) > REPLAY_WINDOW:
            replay_buffer.pop(0)
        all_examples = [ex for batch in replay_buffer for ex in batch]

        sp_time = time.time() - t0

        # ── Train ─────────────────────────────────────────────────────
        t1 = time.time()
        avg_loss = train_on_examples(
            net, optimizer, all_examples, epochs=EPOCHS, device=device,
        )
        train_time = time.time() - t1

        # ── Evaluate: AI vs AI (self-play) ─────────────────────────
        t2 = time.time()
        x_wins, o_wins, draws = play_self_eval(
            net, num_games=EVAL_GAMES,
            num_sims=EVAL_SIMS, device=device,
        )
        eval_time = time.time() - t2
        draw_rate = draws / EVAL_GAMES

        # ── Report ────────────────────────────────────────────────────
        print(f"  Iter {iteration:3d}/{ITERATIONS} │ "
              f"SP: X={results[1]} O={results[-1]} D={results[0]} │ "
              f"{len(all_examples)} ex │ loss={avg_loss:.4f} │ "
              f"AI vs AI: X={x_wins} O={o_wins} D={draws} "
              f"(draw {draw_rate:.0%}) │ "
              f"{sp_time:.1f}s+{train_time:.1f}s+{eval_time:.1f}s")

        # ── Show demo game periodically ───────────────────────────
        if SHOW_GAME_EVERY > 0 and iteration % SHOW_GAME_EVERY == 0:
            show_training_game(net, num_sims=EVAL_SIMS, device=device,
                               iteration=iteration)

        # ── Save if draw rate improved ──────────────────────────────
        if draw_rate >= best_winrate:
            best_winrate = draw_rate
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({"model_state": net.state_dict(),
                         "iteration": iteration,
                         "draw_rate": draw_rate},
                       CHECKPOINT_PATH)

        # Early stop if nearly all draws (perfect play)
        if draw_rate >= 0.95 and iteration >= 10:
            print(f"\n  {draw_rate:.0%} draws in self-play — near-perfect play! Stopping.")
            break

    # ── Final evaluation ──────────────────────────────────────────────
    print(f"\n{'═' * 56}")
    print(f"  FINAL EVALUATION ({EVAL_GAMES} games, AI vs AI)")
    print(f"{'═' * 56}")
    x_wins, o_wins, draws = play_self_eval(
        net, num_games=EVAL_GAMES,
        num_sims=EVAL_SIMS, device=device,
    )
    total = EVAL_GAMES
    print(f"  X wins   : {x_wins:4d}  ({x_wins/total*100:.1f}%)")
    print(f"  O wins   : {o_wins:4d}  ({o_wins/total*100:.1f}%)")
    print(f"  Draws    : {draws:4d}  ({draws/total*100:.1f}%)")
    print(f"{'═' * 56}")

    if draws >= total * 0.9:
        print(f"\n  {draws} draws out of {total} — AI plays near-optimally (draws = perfect play)!")
    elif abs(x_wins - o_wins) <= total * 0.1:
        print(f"\n  Balanced play with {draws} draws — model is learning well.")
    else:
        side = "X" if x_wins > o_wins else "O"
        print(f"\n  {side} is dominant — model has a side bias, more training needed.")

    # Show one demo game
    print(f"\n  --- Demo game ---")
    watch_game(net, num_sims=EVAL_SIMS, device=device)


if __name__ == "__main__":
    main()
