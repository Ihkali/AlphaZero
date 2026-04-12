#!/usr/bin/env bash
#
# pipeline.sh — Full training pipeline
#
#   Step 1: Simulate every game in chess_games.csv, keep only games that
#           end naturally (checkmate, stalemate, draw by rule).
#           Pick top 10,000 by average Elo.
#
#   Step 2: Build data cache and train for 5 epochs.
#
# Usage:
#   chmod +x pipeline.sh
#   ./pipeline.sh
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
VENV="${VENV:-/Users/ihkali/manimations/bin/activate}"
WORKERS=8
RAW_CSV="chess_games.csv"
CLEAN_CSV="chess_games_top10k.csv"
TOP_N=10000
MIN_MOVES=10

# ─── Activate virtual-env ─────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "  Activating virtual environment"
echo "════════════════════════════════════════════════════════════════════"
source "$VENV"

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Strict simulation + top-N by Elo
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Step 1: Simulating every game in ${RAW_CSV}"
echo "    • Replay all moves, reject any illegal move"
echo "    • Keep ONLY games ending in checkmate / stalemate / draw by rule"
echo "    • Reject resignations, time forfeits, abandonments"
echo "    • Pick top ${TOP_N} games by average Elo"
echo "    • Min ${MIN_MOVES} plies per game"
echo "════════════════════════════════════════════════════════════════════"

rm -f "$CLEAN_CSV"
python SL/strict_clean.py \
    --csv "$RAW_CSV" \
    --output "$CLEAN_CSV" \
    --top-n "$TOP_N" \
    --min-moves "$MIN_MOVES" \
    --workers "$WORKERS"

echo ""
echo "  ✓ ${CLEAN_CSV} ready"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Build cache & train 5 epochs
# ═══════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════"
echo "  Step 2: Building data cache & training 5 epochs"
echo "════════════════════════════════════════════════════════════════════"

rm -rf SL/data
rm -f SL/checkpoints/*.pt

python SL/main.py \
    --csv "$CLEAN_CSV" \
    --epochs 5 \
    --min-elo 0 \
    --num-workers 0 \
    --rebuild-cache

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✓ Pipeline complete!"
echo ""
echo "  Data CSV    : ${CLEAN_CSV}"
echo "  Final model : SL/checkpoints/final_model.pt"
echo "  Best model  : SL/checkpoints/sl_best.pt"
echo "  Latest      : SL/checkpoints/latest.pt"
echo "════════════════════════════════════════════════════════════════════"
