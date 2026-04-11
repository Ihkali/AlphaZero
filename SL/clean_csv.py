#!/usr/bin/env python3
"""
SL/clean_csv.py — Validate chess games in CSV by replaying every move.

Uses multiprocessing to validate games in parallel for speed.
Games that fail (illegal moves, parse errors) are removed.

Usage:
    python SL/clean_csv.py                              # writes chess_games_clean.csv
    python SL/clean_csv.py --inplace                    # overwrites chess_games.csv
    python SL/clean_csv.py --csv path/to/games.csv
    python SL/clean_csv.py --min-elo 2000 --output chess_games_2000.csv
    python SL/clean_csv.py --min-elo 2000 --event classical  # only classical games
    python SL/clean_csv.py --workers 8                  # parallel workers
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import multiprocessing as mp
from functools import partial
import chess
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these instead of using CLI flags
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH         = "chess_games.csv"
OUTPUT_PATH      = None          # None = auto-generate <name>_clean.csv
INPLACE          = False         # True = overwrite input CSV
MIN_ELO          = 0             # minimum Elo for both players
MIN_MOVES        = 5             # minimum plies per game
EVENT_FILTER     = None          # e.g. "classical" or "classical,correspondence"
WORKERS          = None          # None = CPU count


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n - 1


def split_san(an: str) -> list[str]:
    tokens = an.split()
    moves = []
    for tok in tokens:
        if tok.endswith("."):
            continue
        if "." in tok:
            parts = tok.split(".")
            candidate = parts[-1].strip()
            if candidate:
                moves.append(candidate)
        else:
            moves.append(tok)
    return moves


def validate_game(an_str: str) -> tuple[bool, str]:
    """Replay a game move-by-move. Returns (is_valid, reason)."""
    an = an_str.strip()
    for token in ("1-0", "0-1", "1/2-1/2", "*"):
        an = an.replace(token, "")
    an = an.strip()
    if not an:
        return False, "empty AN"
    try:
        san_tokens = split_san(an)
    except Exception as e:
        return False, f"split error: {e}"
    if len(san_tokens) < 2:
        return False, "too few moves"
    board = chess.Board()
    for i, san_tok in enumerate(san_tokens):
        try:
            move = board.parse_san(san_tok)
        except (chess.InvalidMoveError, chess.IllegalMoveError,
                chess.AmbiguousMoveError) as e:
            return False, f"move {i+1} '{san_tok}': {e}"
        except Exception as e:
            return False, f"move {i+1} '{san_tok}': unexpected {e}"
        board.push(move)
    return True, "ok"


def _validate_row(row_tuple, min_elo=0, min_moves=5, event_filter=None):
    """Worker function: filter + validate one row. Returns (row_dict, keep, reason)."""
    row = row_tuple

    # Event filter (cheap — done first)
    if event_filter:
        event = row.get("Event", "").strip().lower()
        if not any(ef in event for ef in event_filter):
            return row, False, "event filter"

    # Elo filter (cheap)
    if min_elo:
        try:
            w_elo = int(float(row.get("WhiteElo", 0)))
            b_elo = int(float(row.get("BlackElo", 0)))
        except (ValueError, TypeError):
            return row, False, "bad elo"
        if w_elo < min_elo or b_elo < min_elo:
            return row, False, "elo filter"

    an = row.get("AN", "")
    result = row.get("Result", "")
    if not an or result not in ("1-0", "0-1", "1/2-1/2"):
        return row, False, "missing AN" if not an else f"bad result"

    # Move count check (cheap)
    san_tokens = split_san(an.strip())
    for tok in ("1-0", "0-1", "1/2-1/2", "*"):
        while tok in san_tokens:
            san_tokens.remove(tok)
    if len(san_tokens) < min_moves:
        return row, False, "too few moves"

    # Full validation (expensive)
    valid, reason = validate_game(an)
    return row, valid, reason


def main():
    n_workers = WORKERS or mp.cpu_count()
    csv_path = CSV_PATH
    if INPLACE:
        out_path = csv_path + ".tmp"
    elif OUTPUT_PATH:
        out_path = OUTPUT_PATH
    else:
        base, ext = os.path.splitext(csv_path)
        out_path = f"{base}_clean{ext}"

    print(f"Input:   {csv_path}")
    print(f"Output:  {out_path if not INPLACE else csv_path}")
    event_filter = None
    if EVENT_FILTER:
        event_filter = [e.strip().lower() for e in EVENT_FILTER.split(",")]

    print(f"Workers: {n_workers}")
    if MIN_ELO:
        print(f"Min Elo: {MIN_ELO}")
    if event_filter:
        print(f"Event:   {event_filter}")
    print()

    print("Counting games...")
    total = count_lines(csv_path)
    print(f"Total rows: {total:,}\n")

    CHUNK = 4096  # rows per batch sent to pool
    kept = 0
    removed = 0
    error_reasons: dict[str, int] = {}

    worker_fn = partial(_validate_row,
                        min_elo=MIN_ELO,
                        min_moves=MIN_MOVES,
                        event_filter=event_filter)

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout, \
         mp.Pool(n_workers) as pool:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(total=total, unit="game", desc="Validating",
                    dynamic_ncols=True)

        # Feed rows in chunks via imap for ordered output
        for row, valid, reason in pool.imap(worker_fn, reader, chunksize=CHUNK):
            if valid:
                writer.writerow(row)
                kept += 1
            else:
                removed += 1
                key = reason.split(":")[0] if ":" in reason else reason
                error_reasons[key] = error_reasons.get(key, 0) + 1

            pbar.update(1)
            if (kept + removed) % 50_000 == 0:
                pbar.set_postfix(kept=f"{kept:,}", bad=f"{removed:,}")

        pbar.set_postfix(kept=f"{kept:,}", bad=f"{removed:,}")
        pbar.close()

    if INPLACE:
        os.replace(out_path, csv_path)
        print(f"\nOverwrote {csv_path}")

    total_scanned = kept + removed
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total games scanned:  {total_scanned:,}")
    print(f"  Kept (valid):         {kept:,}")
    print(f"  Removed (invalid):    {removed:,}")
    print(f"  Removal rate:         {removed/max(1,total_scanned)*100:.2f}%")
    if error_reasons:
        print(f"\n  Error breakdown:")
        for reason, count in sorted(error_reasons.items(),
                                     key=lambda x: -x[1])[:15]:
            print(f"    {reason:<30s} {count:>10,}")
    print(f"{'='*60}")
    print(f"Clean CSV: {csv_path if INPLACE else out_path}")


if __name__ == "__main__":
    main()
