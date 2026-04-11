#!/usr/bin/env python3
"""
SL/strict_clean.py — Pick the best chess games by full simulation.

Every game is replayed move-by-move. Only games that reach a *natural*
conclusion on the board are kept:

  KEPT (decisive / drawn by rule):
    • Checkmate  (result must match: 1-0 or 0-1)
    • Stalemate  (result must be 1/2-1/2)
    • Threefold repetition
    • Fifty-move rule
    • Insufficient material
    • Fivefold repetition / 75-move rule (auto-draw)

  REJECTED:
    • Resignation, time forfeit, abandonment, rules infraction
    • Games where the Termination field is NOT "Normal"
    • Any game with an illegal or unparsable move
    • Games that end mid-play without board-level termination
    • Games with < min_moves plies

After filtering, games are ranked by average Elo (WhiteElo + BlackElo) / 2
and the top --top-n (default 10 000) are written to the output CSV.

Usage:
    python SL/strict_clean.py --workers 8
    python SL/strict_clean.py --top-n 10000 --output chess_games_top10k.csv
    python SL/strict_clean.py --csv chess_games.csv --min-moves 10 --workers 8
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import heapq
import multiprocessing as mp
from functools import partial
import chess
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these instead of using CLI flags
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH         = "chess_games.csv"
OUTPUT_PATH      = "chess_games_top10k.csv"
TOP_N            = 10_000        # keep top N games by average Elo
MIN_MOVES        = 10            # minimum plies (half-moves) per game
WORKERS          = None          # None = CPU count


# ─── Helpers ──────────────────────────────────────────────────────────────

def _split_san(an: str) -> list[str]:
    """Split algebraic notation string into SAN tokens."""
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


def _simulate_game(an_str: str, result_str: str, min_moves: int = 10):
    """Replay game, verify every move, and check natural ending.

    Returns:
        (valid: bool, reason: str)
    """
    an = an_str.strip()
    # Strip trailing result tokens from the move text
    for token in ("1-0", "0-1", "1/2-1/2", "*"):
        an = an.replace(token, "")
    an = an.strip()

    if not an:
        return False, "empty AN"

    try:
        san_tokens = _split_san(an)
    except Exception as e:
        return False, f"split error: {e}"

    if len(san_tokens) < min_moves:
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

    # ── Check that the board position matches the claimed result ──────

    # Checkmate
    if board.is_checkmate():
        # Side to move is checkmated → the other side won
        if board.turn == chess.WHITE:
            expected = "0-1"   # Black won
        else:
            expected = "1-0"   # White won
        if result_str == expected:
            return True, "checkmate"
        else:
            return False, f"checkmate but result={result_str} expected={expected}"

    # Stalemate
    if board.is_stalemate():
        if result_str == "1/2-1/2":
            return True, "stalemate"
        return False, f"stalemate but result={result_str}"

    # Insufficient material
    if board.is_insufficient_material():
        if result_str == "1/2-1/2":
            return True, "insufficient material"
        return False, f"insufficient material but result={result_str}"

    # Threefold repetition (or fivefold)
    if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
        if result_str == "1/2-1/2":
            return True, "repetition"
        return False, f"repetition but result={result_str}"

    # Fifty-move rule (or 75-move auto-draw)
    if board.is_seventyfive_moves() or board.can_claim_fifty_moves():
        if result_str == "1/2-1/2":
            return True, "fifty-move rule"
        return False, f"fifty-move but result={result_str}"

    # If none of the above → game ended without a natural conclusion
    # (resignation, timeout, disconnect, etc.)
    return False, "no natural ending"


# ─── Worker function ─────────────────────────────────────────────────────

def _validate_row(row, min_moves=10):
    """Validate one CSV row. Returns (row, valid, reason, avg_elo)."""
    # Quick checks first
    termination = row.get("Termination", "").strip()
    if termination != "Normal":
        return row, False, f"termination={termination}", 0

    result = row.get("Result", "")
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return row, False, "bad result", 0

    an = row.get("AN", "")
    if not an:
        return row, False, "missing AN", 0

    try:
        w_elo = int(float(row.get("WhiteElo", 0)))
        b_elo = int(float(row.get("BlackElo", 0)))
    except (ValueError, TypeError):
        return row, False, "bad elo", 0

    avg_elo = (w_elo + b_elo) / 2.0

    # Full simulation (expensive)
    valid, reason = _simulate_game(an, result, min_moves=min_moves)
    return row, valid, reason, avg_elo


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    n_workers = WORKERS or mp.cpu_count()
    csv_path = CSV_PATH

    print(f"Input:      {csv_path}")
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Top-N:      {TOP_N:,}")
    print(f"Min moves:  {MIN_MOVES}")
    print(f"Workers:    {n_workers}")
    print()

    # Count rows
    n = 0
    with open(csv_path, "rb") as f:
        for _ in f:
            n += 1
    total = max(0, n - 1)
    print(f"Total rows: {total:,}\n")

    # ── Pass 1: validate all games, collect valid ones in a min-heap ──
    #    Heap keeps only the top-N by avg Elo (O(N log top_n) memory).
    CHUNK = 4096
    worker_fn = partial(_validate_row, min_moves=MIN_MOVES)

    # heap of (avg_elo, counter, row_dict)  — min-heap, so lowest elo pops first
    heap: list[tuple[float, int, dict]] = []
    counter = 0
    passed = 0
    rejected = 0
    reason_counts: dict[str, int] = {}
    ending_counts: dict[str, int] = {}

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fin, \
         mp.Pool(n_workers) as pool:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames

        pbar = tqdm(total=total, unit="game", desc="Simulating",
                    dynamic_ncols=True)

        for row, valid, reason, avg_elo in pool.imap(worker_fn, reader, chunksize=CHUNK):
            if valid:
                passed += 1
                ending_counts[reason] = ending_counts.get(reason, 0) + 1
                counter += 1
                if len(heap) < TOP_N:
                    heapq.heappush(heap, (avg_elo, counter, row))
                elif avg_elo > heap[0][0]:
                    heapq.heapreplace(heap, (avg_elo, counter, row))
            else:
                rejected += 1
                key = reason.split(":")[0] if ":" in reason else reason
                reason_counts[key] = reason_counts.get(key, 0) + 1

            pbar.update(1)
            if (passed + rejected) % 50_000 == 0:
                min_elo_in_heap = f"{heap[0][0]:.0f}" if heap else "–"
                pbar.set_postfix(
                    ok=f"{passed:,}", bad=f"{rejected:,}",
                    heap=f"{len(heap):,}", minElo=min_elo_in_heap)

        pbar.close()

    # Sort heap by Elo descending for output
    heap.sort(key=lambda x: -x[0])

    # ── Write output CSV ──────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for avg_elo, _, row in heap:
            writer.writerow(row)

    # ── Summary ───────────────────────────────────────────────────────
    total_scanned = passed + rejected
    elos = [e for e, _, _ in heap]
    print(f"\n{'='*64}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Total scanned:   {total_scanned:,}")
    print(f"  Passed (natural ending + valid moves): {passed:,}")
    print(f"  Rejected:        {rejected:,}")
    print(f"  Pass rate:       {passed/max(1,total_scanned)*100:.2f}%")
    print()
    print(f"  Natural endings found:")
    for ending, cnt in sorted(ending_counts.items(), key=lambda x: -x[1]):
        print(f"    {ending:<28s} {cnt:>10,}")
    print()
    print(f"  Top {len(heap):,} games written to: {OUTPUT_PATH}")
    if elos:
        print(f"  Elo range: {elos[-1]:.0f} – {elos[0]:.0f}  "
              f"(avg {sum(elos)/len(elos):.0f})")
    print()
    print(f"  Rejection breakdown (top 15):")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {reason:<35s} {count:>10,}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
