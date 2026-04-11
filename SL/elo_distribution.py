#!/usr/bin/env python3
"""
SL/elo_distribution.py — Show Elo distribution of chess_games.csv.

Usage:
    python SL/elo_distribution.py
    python SL/elo_distribution.py --csv path/to/games.csv
    python SL/elo_distribution.py --buckets 20
"""

import os
import sys
import csv
from collections import Counter
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these instead of using CLI flags
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH         = "chess_games.csv"
NUM_BUCKETS      = 15            # histogram buckets


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def main():
    total = count_lines(CSV_PATH)
    print(f"Scanning {CSV_PATH} ({total:,} rows)...\n")

    white_elos = []
    black_elos = []
    avg_elos = []
    bad = 0

    with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total, unit="game", desc="Reading",
                        dynamic_ncols=True):
            try:
                w = int(float(row.get("WhiteElo", 0)))
                b = int(float(row.get("BlackElo", 0)))
            except (ValueError, TypeError):
                bad += 1
                continue
            if w < 100 or b < 100:
                bad += 1
                continue
            white_elos.append(w)
            black_elos.append(b)
            avg_elos.append((w + b) // 2)

    n = len(avg_elos)
    if n == 0:
        print("No valid Elo data found!")
        return

    all_elos = white_elos + black_elos
    min_elo = min(all_elos)
    max_elo = max(all_elos)

    print(f"\n{'='*60}")
    print(f"  Games with valid Elo: {n:,}  (skipped {bad:,})")
    print(f"{'='*60}")
    print(f"\n  White Elo  — min: {min(white_elos):,}  max: {max(white_elos):,}  "
          f"avg: {sum(white_elos)//len(white_elos):,}")
    print(f"  Black Elo  — min: {min(black_elos):,}  max: {max(black_elos):,}  "
          f"avg: {sum(black_elos)//len(black_elos):,}")
    print(f"  Avg Elo    — min: {min(avg_elos):,}  max: {max(avg_elos):,}  "
          f"avg: {sum(avg_elos)//len(avg_elos):,}")

    # Histogram of average Elo
    bucket_size = max(1, (max_elo - min_elo + 1) // NUM_BUCKETS)
    buckets = Counter()
    for e in avg_elos:
        b = ((e - min_elo) // bucket_size) * bucket_size + min_elo
        buckets[b] += 1

    max_count = max(buckets.values())
    bar_width = 40

    print(f"\n  Average Elo distribution ({NUM_BUCKETS} buckets):\n")
    print(f"  {'Elo range':>15s}  {'Count':>10s}  {'%':>6s}  Bar")
    print(f"  {'-'*15}  {'-'*10}  {'-'*6}  {'-'*bar_width}")

    for b in sorted(buckets.keys()):
        lo = b
        hi = b + bucket_size - 1
        count = buckets[b]
        pct = count / n * 100
        bar_len = int(count / max_count * bar_width)
        bar = "█" * bar_len
        print(f"  {lo:>7,}-{hi:<7,}  {count:>10,}  {pct:>5.1f}%  {bar}")

    # Useful thresholds
    print(f"\n  Games by minimum Elo (both players ≥ threshold):\n")
    print(f"  {'Threshold':>10s}  {'Games':>12s}  {'%':>6s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*6}")
    for thresh in [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]:
        count = sum(1 for w, b in zip(white_elos, black_elos)
                    if w >= thresh and b >= thresh)
        pct = count / n * 100
        print(f"  {thresh:>10,}  {count:>12,}  {pct:>5.1f}%")

    print()


if __name__ == "__main__":
    main()
