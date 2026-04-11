"""
MCTS/self_play.py — Self-play game generation + Disk-streaming data.

Each game's examples are appended to disk immediately after completion,
keeping RAM usage near-zero during self-play.

Training reads batches directly from memory-mapped .npy files on disk.
"""

import os
import glob
import time
import struct
import numpy as np
import chess
import torch
import torch.multiprocessing as mp

from MCTS.config import Config
from MCTS.encode import encode_board, index_to_move
from MCTS.mcts import mcts_search, select_action, get_subtree_for_action


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMING DATA WRITER  — appends game examples to .npy files on disk
# ═══════════════════════════════════════════════════════════════════════════

class StreamingDataWriter:
    """
    Appends examples to three raw .npy files (states, policies, values)
    one game at a time.  Keeps zero examples in RAM.
    """

    def __init__(self, data_dir: str, iteration: int):
        os.makedirs(data_dir, exist_ok=True)
        prefix = os.path.join(data_dir, f"iter_{iteration:04d}")
        self._s_path = prefix + "_states.npy"
        self._p_path = prefix + "_policies.npy"
        self._v_path = prefix + "_values.npy"
        self._count = 0
        # Create empty files with proper npy headers (we'll fix count at end)
        self._s_fp = self._open_npy(self._s_path, (0, Config.input_planes, 8, 8))
        self._p_fp = self._open_npy(self._p_path, (0, Config.policy_size))
        self._v_fp = self._open_npy(self._v_path, (0,))

    @staticmethod
    def _open_npy(path, shape, dtype=np.float32):
        """Create an npy v1.0 file with a header that we can patch later."""
        fp = open(path, "wb")
        # Write a padded npy header so we can update the count in-place
        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        }
        np.lib.format.write_array_header_1_0(fp, header)
        return fp

    def append(self, examples: list):
        """Append a list of (state, policy, value) tuples to disk."""
        if not examples:
            return
        n = len(examples)
        states = np.array([e[0] for e in examples], dtype=np.float32)
        policies = np.array([e[1] for e in examples], dtype=np.float32)
        values = np.array([e[2] for e in examples], dtype=np.float32)
        self._s_fp.write(states.tobytes())
        self._p_fp.write(policies.tobytes())
        self._v_fp.write(values.tobytes())
        self._count += n

    def finalize(self) -> tuple[str, str, str, int]:
        """Close files and patch the npy headers with the correct count."""
        for fp, path, extra_shape in [
            (self._s_fp, self._s_path, (Config.input_planes, 8, 8)),
            (self._p_fp, self._p_path, (Config.policy_size,)),
            (self._v_fp, self._v_path, ()),
        ]:
            fp.close()
            self._patch_npy_count(path, self._count, extra_shape)
        return self._s_path, self._p_path, self._v_path, self._count

    @staticmethod
    def _patch_npy_count(path, count, extra_shape):
        """Rewrite the npy header with the final row count."""
        shape = (count,) + extra_shape
        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": shape,
        }
        with open(path, "r+b") as fp:
            np.lib.format.write_array_header_1_0(fp, header)


# ═══════════════════════════════════════════════════════════════════════════
#  DISK REPLAY BUFFER  — memory-mapped, zero-copy sampling
# ═══════════════════════════════════════════════════════════════════════════

class DiskReplayBuffer:
    """
    Samples training batches directly from memory-mapped .npy files.
    Only the requested batch is loaded into RAM (a few MB at most).
    """

    def __init__(self, data_dir: str = Config.data_dir,
                 window: int = Config.data_window):
        self.data_dir = data_dir
        self.window = window
        self._files = []          # list of (s_path, p_path, v_path)
        self._counts = []         # examples per file
        self._cum = np.array([], dtype=np.int64)
        self._total = 0
        self._loaded = False

    def refresh(self):
        """Scan data_dir for iteration files and build an index."""
        # Find all complete iterations (those with _states.npy)
        pattern = os.path.join(self.data_dir, "iter_*_states.npy")
        s_files = sorted(glob.glob(pattern))
        if not s_files:
            self._files, self._counts = [], []
            self._cum = np.array([], dtype=np.int64)
            self._total = 0
            self._loaded = True
            return

        # Take the last `window` iterations
        s_files = s_files[-self.window:]

        self._files = []
        self._counts = []
        for sf in s_files:
            prefix = sf.replace("_states.npy", "")
            pf = prefix + "_policies.npy"
            vf = prefix + "_values.npy"
            if not (os.path.exists(pf) and os.path.exists(vf)):
                continue
            # Peek at the mmap to get count without loading
            mm = np.load(sf, mmap_mode="r")
            self._files.append((sf, pf, vf))
            self._counts.append(len(mm))
            del mm

        self._cum = np.cumsum(self._counts)
        self._total = int(self._cum[-1]) if len(self._cum) > 0 else 0
        self._loaded = True
        print(f"  DiskReplayBuffer: {self._total} examples indexed "
              f"from {len(self._files)} files (window={self.window})")

    def sample(self, batch_size: int = Config.batch_size):
        """Sample a batch by reading only the needed rows via mmap."""
        if not self._loaded:
            self.refresh()
        if self._total == 0:
            return (np.zeros((0,), dtype=np.float32),) * 3
        bs = min(batch_size, self._total)
        indices = np.random.choice(self._total, size=bs, replace=False)
        indices.sort()  # sequential access is faster on mmap

        # Group indices by file
        file_idx = np.searchsorted(self._cum, indices, side="right")
        offsets = np.zeros_like(indices)
        offsets[file_idx > 0] = self._cum[file_idx[file_idx > 0] - 1]
        local_idx = indices - offsets

        states_list, policies_list, values_list = [], [], []
        # Open mmaps lazily per unique file, read rows, close
        for fi in np.unique(file_idx):
            mask = file_idx == fi
            rows = local_idx[mask]
            sf, pf, vf = self._files[fi]
            s_mm = np.load(sf, mmap_mode="r")
            p_mm = np.load(pf, mmap_mode="r")
            v_mm = np.load(vf, mmap_mode="r")
            states_list.append(np.array(s_mm[rows]))
            policies_list.append(np.array(p_mm[rows]))
            values_list.append(np.array(v_mm[rows]))
            del s_mm, p_mm, v_mm

        return (np.concatenate(states_list),
                np.concatenate(policies_list),
                np.concatenate(values_list))

    def __len__(self):
        if not self._loaded:
            self.refresh()
        return self._total


# ═══════════════════════════════════════════════════════════════════════════
#  SELF-PLAY GAME
# ═══════════════════════════════════════════════════════════════════════════

def self_play_game(
    neural_net, num_sims=Config.num_mcts_sims,
    temp_threshold=Config.temp_threshold_move,
    max_moves=Config.max_game_moves,
    device="cpu", resign_threshold=-1.0,
    value_net=None, rollout_net=None,
    lambda_mix=Config.lambda_mix,
    force_full_game=False,
):
    """Play one self-play game with MCTS.

    When `force_full_game` is True, the game ignores the resign
    threshold and plays to completion — used for resign verification
    (paper recommends checking that the resign threshold is calibrated).
    """
    board = chess.Board()
    game_history = []
    move_count = 0
    resign_count = 0
    resigned = False
    reuse_root = None  # subtree reuse: persist tree across moves

    while not board.is_game_over() and move_count < max_moves:
        temperature = 1.0 if move_count < temp_threshold else 0.0
        encoded = encode_board(board)

        policy, root_value, root_node = mcts_search(
            board, policy_net=neural_net,
            value_net=value_net,
            rollout_net=rollout_net,
            num_simulations=num_sims,
            lambda_mix=lambda_mix,
            temperature=temperature,
            device=device,
            reuse_root=reuse_root,
        )
        game_history.append((encoded, policy, board.turn))

        action = select_action(policy, temperature)
        move = index_to_move(action, board)
        if move not in board.legal_moves:
            move = list(board.legal_moves)[0]

        # Extract the subtree for the chosen action → reuse next move
        reuse_root = get_subtree_for_action(root_node, action)

        board.push(move)
        move_count += 1

        # Resign logic (skip if force_full_game for verification)
        if resign_threshold > -1.0 and root_value < resign_threshold:
            resign_count += 1
            if resign_count >= Config.resign_consecutive and not force_full_game:
                resigned = True
                break
        else:
            resign_count = 0

    outcome = _get_game_result(board, resigned)
    examples = []
    for encoded, policy, player in game_history:
        if outcome == 0:
            value = 0.0
        elif outcome == 1:
            value = 1.0 if player == chess.WHITE else -1.0
        else:
            value = -1.0 if player == chess.WHITE else 1.0
        examples.append((encoded, policy, value))

    return examples, outcome, move_count


def _get_game_result(board, resigned=False):
    if resigned:
        # The side to move resigned → the other side wins
        return -1 if board.turn == chess.WHITE else 1
    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1
    return 0


def _worker_play_games(worker_id, model_state_dict, num_games, num_sims,
                       max_moves, temp_threshold, resign_threshold,
                       result_queue,
                       resign_check_fraction=Config.resign_check_fraction):
    from MCTS.model import AlphaZeroNet
    net = AlphaZeroNet()
    net.load_state_dict(model_state_dict)
    net.eval()
    device = "cpu"
    for g in range(num_games):
        # Resign verification: a fraction of games play out fully
        # to check that the resign threshold is properly calibrated
        force_full = (np.random.random() < resign_check_fraction
                      and resign_threshold > -1.0)
        examples, outcome, move_count = self_play_game(
            net, num_sims=num_sims, device=device,
            max_moves=max_moves, temp_threshold=temp_threshold,
            resign_threshold=resign_threshold,
            force_full_game=force_full,
        )
        outcome_str = {1: "White wins", -1: "Black wins", 0: "Draw"}[outcome]
        result_queue.put({
            "worker": worker_id, "game": g, "examples": examples,
            "outcome": outcome, "outcome_str": outcome_str,
            "move_count": move_count,
            "force_full": force_full,
        })


def run_self_play(neural_net, num_games=Config.self_play_games,
                  num_sims=Config.num_mcts_sims, device="cpu",
                  num_workers=Config.num_workers, verbose=True,
                  resign_threshold=-1.0, iteration=0,
                  data_dir=Config.data_dir):
    """
    Run parallel self-play.  Each completed game is flushed to disk
    immediately via StreamingDataWriter — RAM holds zero past games.
    """
    model_state_dict = {k: v.cpu() for k, v in neural_net.state_dict().items()}
    games_per_worker = _distribute(num_games, num_workers)
    active_workers = sum(1 for g in games_per_worker if g > 0)

    if verbose:
        print(f"  Launching {active_workers} workers "
              f"({num_games} games, {num_sims} sims/move)...")

    result_queue = mp.Queue()
    processes = []
    for w_id, n_games in enumerate(games_per_worker):
        if n_games == 0:
            continue
        p = mp.Process(
            target=_worker_play_games,
            args=(w_id, model_state_dict, n_games, num_sims,
                  Config.max_game_moves, Config.temp_threshold_move,
                  resign_threshold, result_queue),
        )
        p.start()
        processes.append(p)

    # Stream results to disk game-by-game
    writer = StreamingDataWriter(data_dir, iteration)
    results = {"White wins": 0, "Black wins": 0, "Draw": 0}
    game_lengths = []
    games_done = 0
    total_examples = 0
    sp_start_time = time.time()

    while games_done < num_games:
        msg = result_queue.get()
        games_done += 1
        n_ex = len(msg["examples"])
        total_examples += n_ex

        # Flush this game's data to disk immediately, then discard
        writer.append(msg["examples"])
        game_lengths.append(msg["move_count"])

        if msg["outcome"] == 1:
            results["White wins"] += 1
        elif msg["outcome"] == -1:
            results["Black wins"] += 1
        else:
            results["Draw"] += 1

        if verbose:
            elapsed = time.time() - sp_start_time
            gpm = games_done / elapsed * 60 if elapsed > 0 else 0
            pct = games_done / num_games * 100
            bar_len = 20
            filled = int(pct / 100 * bar_len)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"  [{bar}] {games_done:3d}/{num_games} "
                  f"(w{msg['worker']}) {msg['outcome_str']:11s} | "
                  f"{msg['move_count']:3d} moves | "
                  f"{n_ex} ex → disk | "
                  f"{gpm:.1f} games/min")

    for p in processes:
        p.join()

    # Finalize the npy files (patch headers with final count)
    s_path, p_path, v_path, count = writer.finalize()
    size_mb = sum(os.path.getsize(f) for f in (s_path, p_path, v_path)) / (1024**2)

    stats = {
        **results,
        "total_games": num_games,
        "total_examples": total_examples,
        "avg_length": sum(game_lengths) / max(len(game_lengths), 1),
        "min_length": min(game_lengths) if game_lengths else 0,
        "max_length": max(game_lengths) if game_lengths else 0,
    }

    if verbose:
        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  SELF-PLAY RESULTS ({active_workers} workers)      │")
        print(f"  ├─────────────────────────────────────┤")
        print(f"  │  White wins : {results['White wins']:4d}                   │")
        print(f"  │  Black wins : {results['Black wins']:4d}                   │")
        print(f"  │  Draws      : {results['Draw']:4d}                   │")
        print(f"  │  Total games: {num_games:4d}                   │")
        print(f"  ├─────────────────────────────────────┤")
        print(f"  │  Avg length : {stats['avg_length']:6.1f} moves          │")
        print(f"  │  Min / Max  : {stats['min_length']:4d} / {stats['max_length']:4d} moves     │")
        print(f"  │  Examples   : {total_examples:6d}                │")
        print(f"  │  Disk       : {size_mb:6.1f} MB               │")
        print(f"  └─────────────────────────────────────┘")

    return stats


def _distribute(total, workers):
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]
