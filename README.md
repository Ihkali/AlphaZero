# AlphaZero Chess

Three independent chess AI training approaches — MCTS, GRPO, and PPO — each in its own self-contained folder.

## Structure

```
AlphaZero/
├── MCTS/    AlphaZero-style MCTS self-play + replay buffer
├── GRPO/    Group Relative Policy Optimization (game-outcome rewards)
└── PPO/     Proximal Policy Optimization (partial step-level rewards + GAE)
```

All three share the same **AlphaZeroNet** architecture (128 filters × 12 residual blocks, 4.3M parameters) but are completely independent — no cross-imports.

## Requirements

- Python 3.10+
- PyTorch (with MPS/CUDA support)
- python-chess
- tqdm
- pygame (for GUI play/watch)
- numpy

```bash
pip install torch python-chess tqdm pygame numpy
```

## Training

Run from the project root (`AlphaZero/`):

```bash
# ── MCTS (AlphaZero self-play) ────────────────────────
python -m MCTS.main              # full training
python -m MCTS.main --quick      # smoke test

# ── GRPO (game-outcome rewards) ───────────────────────
python -m GRPO.main              # full training
python -m GRPO.main --quick      # smoke test
python -m GRPO.main --updates 500 --games 100

# ── PPO (partial rewards + GAE) ───────────────────────
python -m PPO.main               # full training
python -m PPO.main --quick       # smoke test
python -m PPO.main --updates 500 --gamma 0.99
```

### Common CLI flags

| Flag | Description |
|------|-------------|
| `--quick` | Smoke test (few updates, small games) |
| `--updates N` | Total training updates |
| `--games N` | Games per update |
| `--envs N` | Parallel environments |
| `--lr F` | Learning rate |
| `--device DEV` | `mps`, `cuda`, or `cpu` |
| `--lr-schedule` | `cosine`, `linear`, or `constant` |

## Play against the AI

```bash
# Play vs GRPO model
python -m GRPO.play_gui
python -m GRPO.play_gui --color black --temp 0.3

# Play vs PPO model
python -m PPO.play_gui
python -m PPO.play_gui --checkpoint PPO/checkpoints/ppo_final.pt
```

## Watch AI vs AI

```bash
python -m GRPO.watch_ai
python -m GRPO.watch_ai --delay 0.5 --temp 0.1

python -m PPO.watch_ai
python -m PPO.watch_ai --white PPO/checkpoints/ppo_final.pt --delay 0.3
```

## Key differences

| | MCTS | GRPO | PPO |
|---|------|------|-----|
| **Search** | Monte Carlo Tree Search | None (raw policy) | None (raw policy) |
| **Rewards** | Game outcome via MCTS value targets | Game outcome only (W/L/D) | Per-move partial rewards + game outcome |
| **Advantage** | N/A (supervised from MCTS π) | Episode reward = advantage | GAE (γ=0.99, λ=0.95) |
| **Value training** | MSE vs MCTS returns | Auxiliary MSE vs game outcome | MSE vs discounted returns |
| **Partial rewards** | No | No | Material, mobility, centre, check, capture |

## Device

Defaults to `mps` (Apple Silicon). Change via `--device cuda` or `--device cpu`, or edit `config.py` in any folder.
