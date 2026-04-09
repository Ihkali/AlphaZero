"""
SL/dataset.py — Streaming Dataset that reads chess_games.csv in chunks.

Each CSV row has an 'AN' column with the full game in algebraic notation and
a 'Result' column (1-0, 0-1, 1/2-1/2).  We replay each game, encoding every
position as a (state, move_index, value) training sample.

Because 6M+ games would explode memory if materialised, we:
  1. Pre-process the CSV into compact raw binary files on disk the first time
     (SL/data/sl_{states,moves,values}.bin  +  sl_meta.json).
  2. On subsequent runs, memory-map those files for near-zero RAM usage.
     No np.save/np.load — avoids pulling entire arrays into RAM.
"""

import os
import csv
import json
import random
import logging
import numpy as np
import chess
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from SL.config import Config
from SL.encode import encode_board, move_to_index

logger = logging.getLogger("sl")

CACHE_DIR = "SL/data"
META_FILE = os.path.join(CACHE_DIR, "sl_meta.json")


# ═══════════════════════════════════════════════════════════════════════════
#  CSV → numpy cache
# ═══════════════════════════════════════════════════════════════════════════

def _result_value(result_str: str, perspective_is_white: bool) -> float:
    """Return value in [-1, +1] from the perspective of the side to move."""
    if result_str == "1-0":
        return 1.0 if perspective_is_white else -1.0
    elif result_str == "0-1":
        return -1.0 if perspective_is_white else 1.0
    else:
        return 0.0


def _parse_game(an_str: str, result_str: str):
    """Yield (board_before_move, move, value) for each ply in a game."""
    board = chess.Board()
    # Strip result token at the end if present
    an = an_str.strip()
    for token in ("1-0", "0-1", "1/2-1/2", "*"):
        an = an.replace(token, "")
    an = an.strip()
    if not an:
        return

    try:
        # Parse moves via python-chess (handles SAN)
        moves = []
        temp_board = chess.Board()
        for san_tok in _split_san(an):
            move = temp_board.parse_san(san_tok)
            moves.append(move)
            temp_board.push(move)
    except Exception:
        return  # skip unparseable games

    board = chess.Board()
    for move in moves:
        is_white = board.turn == chess.WHITE
        value = _result_value(result_str, is_white)
        yield board.copy(), move, value
        board.push(move)


def _split_san(an: str) -> list[str]:
    """Split AN string into individual SAN move tokens.

    Handles formats like: '1. e4 e5 2. Nf3 ...'  or  '1.e4 e5 2.Nf3 ...'
    """
    tokens = an.split()
    moves = []
    for tok in tokens:
        # Skip move numbers like "1.", "2.", "12."
        if tok.endswith("."):
            continue
        # Handle "1.e4" → strip the "1." prefix
        if "." in tok:
            parts = tok.split(".")
            # Last part is the move
            candidate = parts[-1].strip()
            if candidate:
                moves.append(candidate)
        else:
            moves.append(tok)
    return moves


def _count_csv_rows(csv_path: str) -> int:
    """Fast line count (subtract header)."""
    n = 0
    with open(csv_path, "rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def build_cache(csv_path: str, max_games: int = 0, min_elo: int = 0):
    """Read CSV in streaming fashion → write raw binary cache directly.

    Writes directly to the final .bin files (no intermediate temp files,
    no np.save conversion) so RAM stays minimal throughout.
    Returns (num_positions,).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    states_path = os.path.join(CACHE_DIR, "sl_states.bin")
    moves_path  = os.path.join(CACHE_DIR, "sl_moves.bin")
    values_path = os.path.join(CACHE_DIR, "sl_values.bin")

    # Check if cache already exists via metadata
    if os.path.isfile(META_FILE):
        with open(META_FILE, "r") as mf:
            meta = json.load(mf)
        n = meta["num_positions"]
        logger.info(f"Cache already exists with {n:,} positions — skipping rebuild.")
        return n

    logger.info(f"Building SL cache from {csv_path} ...")
    logger.info(f"  min_elo={min_elo}, max_games={max_games or 'all'}")

    total_rows = _count_csv_rows(csv_path)
    if max_games:
        total_rows = min(total_rows, max_games)

    # ── Stream CSV → flush chunks straight into final .bin files ──────
    f_states = open(states_path, "wb")
    f_moves  = open(moves_path,  "wb")
    f_values = open(values_path, "wb")

    FLUSH_EVERY = 5000  # flush batch to disk every N games
    buf_states: list = []
    buf_moves:  list = []
    buf_values: list = []

    game_count = 0
    pos_count  = 0
    skipped    = 0

    def _flush():
        nonlocal buf_states, buf_moves, buf_values
        if not buf_states:
            return
        s = np.array(buf_states, dtype=np.float32)
        m = np.array(buf_moves,  dtype=np.int64)
        v = np.array(buf_values, dtype=np.float32)
        f_states.write(s.tobytes())
        f_moves.write(m.tobytes())
        f_values.write(v.tobytes())
        buf_states.clear()
        buf_moves.clear()
        buf_values.clear()

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        pbar = tqdm(reader, total=total_rows, unit="game",
                    desc="Building cache", dynamic_ncols=True)

        for row in pbar:
            # Elo filter
            try:
                w_elo = int(float(row.get("WhiteElo", 0)))
                b_elo = int(float(row.get("BlackElo", 0)))
            except (ValueError, TypeError):
                skipped += 1
                continue

            if min_elo and (w_elo < min_elo or b_elo < min_elo):
                skipped += 1
                continue

            an = row.get("AN", "")
            result = row.get("Result", "")
            if not an or result not in ("1-0", "0-1", "1/2-1/2"):
                skipped += 1
                continue

            game_positions = list(_parse_game(an, result))
            if not game_positions:
                skipped += 1
                continue

            if Config.positions_per_game > 0 and len(game_positions) > Config.positions_per_game:
                game_positions = random.sample(game_positions, Config.positions_per_game)

            for board, move, value in game_positions:
                state = encode_board(board)
                move_idx = move_to_index(move, board)
                buf_states.append(state)
                buf_moves.append(move_idx)
                buf_values.append(value)
                pos_count += 1

            game_count += 1
            if game_count % FLUSH_EVERY == 0:
                _flush()
                pbar.set_postfix(games=f"{game_count:,}",
                                 pos=f"{pos_count:,}", skipped=f"{skipped:,}")

            if max_games and game_count >= max_games:
                break

        pbar.close()

    _flush()
    f_states.close()
    f_moves.close()
    f_values.close()

    logger.info(f"  Done: {game_count:,} games → {pos_count:,} positions  "
                f"(skipped {skipped:,})")

    if pos_count == 0:
        for p in (states_path, moves_path, values_path):
            os.unlink(p)
        raise ValueError("No valid positions found! Check CSV path and elo filter.")

    # ── Save metadata (tiny JSON, no heavy conversion) ────────────────
    meta = {
        "num_positions": pos_count,
        "num_games": game_count,
        "state_shape": [pos_count, Config.input_planes, 8, 8],
        "state_dtype": "float32",
        "move_dtype": "int64",
        "value_dtype": "float32",
    }
    with open(META_FILE, "w") as mf:
        json.dump(meta, mf, indent=2)

    logger.info(f"  Cache saved to {CACHE_DIR}/  ({pos_count:,} positions)")
    return pos_count


# ═══════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset (memory-mapped raw binary — zero RAM overhead)
# ═══════════════════════════════════════════════════════════════════════════

def _load_meta():
    with open(META_FILE, "r") as f:
        return json.load(f)


class ChessSLDataset(Dataset):
    """Memory-mapped dataset over the raw binary cache.

    np.memmap reads data directly from disk page-by-page,
    so only the current batch lives in RAM at any time.
    """

    def __init__(self, indices: np.ndarray | None = None):
        meta = _load_meta()
        n = meta["num_positions"]
        state_shape = tuple(meta["state_shape"])  # (N, 119, 8, 8)

        self.states = np.memmap(
            os.path.join(CACHE_DIR, "sl_states.bin"),
            dtype=np.float32, mode="r", shape=state_shape,
        )
        self.moves = np.memmap(
            os.path.join(CACHE_DIR, "sl_moves.bin"),
            dtype=np.int64, mode="r", shape=(n,),
        )
        self.values = np.memmap(
            os.path.join(CACHE_DIR, "sl_values.bin"),
            dtype=np.float32, mode="r", shape=(n,),
        )

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # .copy() detaches from mmap so the page can be freed
        state = torch.from_numpy(self.states[i].copy()).float()
        move  = torch.tensor(self.moves[i], dtype=torch.long)
        value = torch.tensor(self.values[i], dtype=torch.float32)
        return state, move, value


def make_datasets():
    """Return (train_dataset, val_dataset) from the cache."""
    meta = _load_meta()
    n = meta["num_positions"]
    indices = np.arange(n)
    np.random.shuffle(indices)

    val_size = max(1, int(n * Config.val_split))
    val_idx  = indices[:val_size]
    train_idx = indices[val_size:]

    logger.info(f"Dataset split: train={len(train_idx):,}  val={len(val_idx):,}")
    return ChessSLDataset(train_idx), ChessSLDataset(val_idx)


# ═══════════════════════════════════════════════════════════════════════════
#  In-memory chunk dataset — loads a slice of the cache into RAM
# ═══════════════════════════════════════════════════════════════════════════

class InMemoryChunkDataset(Dataset):
    """Holds a contiguous slice of the binary cache fully in RAM.

    Much faster than random memmap access because all data lives in
    physical memory and there is no page-fault overhead.
    """

    def __init__(self, states: np.ndarray, moves: np.ndarray, values: np.ndarray):
        self.states = states   # (C, 119, 8, 8)  float32, in RAM
        self.moves  = moves    # (C,)             int64
        self.values = values   # (C,)             float32

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        state = torch.from_numpy(self.states[idx]).float()
        move  = torch.tensor(self.moves[idx], dtype=torch.long)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, move, value


def chunk_iterator(chunk_size: int = 0):
    """Yield (train_dataset, val_dataset) one chunk at a time.

    Each chunk reads `chunk_size` positions sequentially from the
    memmap files into plain numpy arrays (→ RAM), shuffles them,
    and splits off a small validation set.  When the generator
    advances to the next chunk the previous arrays are freed.

    Args:
        chunk_size: positions per chunk.  0 = use Config.chunk_positions.
    Yields:
        (train_ds, val_ds, chunk_idx, num_chunks)
    """
    if chunk_size <= 0:
        chunk_size = Config.chunk_positions

    meta = _load_meta()
    n = meta["num_positions"]
    state_shape = tuple(meta["state_shape"])  # (N, 119, 8, 8)

    # Open memmap handles (no RAM used yet)
    mm_states = np.memmap(
        os.path.join(CACHE_DIR, "sl_states.bin"),
        dtype=np.float32, mode="r", shape=state_shape,
    )
    mm_moves = np.memmap(
        os.path.join(CACHE_DIR, "sl_moves.bin"),
        dtype=np.int64, mode="r", shape=(n,),
    )
    mm_values = np.memmap(
        os.path.join(CACHE_DIR, "sl_values.bin"),
        dtype=np.float32, mode="r", shape=(n,),
    )

    num_chunks = max(1, (n + chunk_size - 1) // chunk_size)

    # Shuffle chunk order each epoch (caller re-creates the generator)
    chunk_order = np.arange(num_chunks)
    np.random.shuffle(chunk_order)

    for ci, chunk_idx in enumerate(chunk_order):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, n)
        csize = end - start

        # Sequential read from memmap → copy into contiguous RAM arrays
        s = np.array(mm_states[start:end])    # (csize, 119, 8, 8)
        m = np.array(mm_moves[start:end])     # (csize,)
        v = np.array(mm_values[start:end])    # (csize,)

        # Shuffle within chunk
        perm = np.random.permutation(csize)
        s = s[perm]
        m = m[perm]
        v = v[perm]

        val_size = max(1, int(csize * Config.val_split))
        train_ds = InMemoryChunkDataset(s[val_size:], m[val_size:], v[val_size:])
        val_ds   = InMemoryChunkDataset(s[:val_size], m[:val_size], v[:val_size])

        logger.info(f"  Chunk {ci+1}/{num_chunks}  "
                    f"[{start:,}–{end:,}]  "
                    f"train={len(train_ds):,}  val={len(val_ds):,}")

        yield train_ds, val_ds, ci, num_chunks

        # Explicitly free RAM before next chunk
        del s, m, v, perm, train_ds, val_ds
