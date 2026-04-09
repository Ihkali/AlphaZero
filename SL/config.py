"""
SL/config.py — Hyperparameters for Supervised Learning on chess games.
"""


class Config:
    # ── State encoding (same as MCTS) ─────────────────────────────────
    history_length: int = 8
    input_planes: int = 119          # T*14 + 7 = 8*14 + 7
    board_size: int = 8

    # ── Action encoding ───────────────────────────────────────────────
    action_planes: int = 73
    policy_size: int = board_size * board_size * action_planes  # 4672

    # ── Neural network ────────────────────────────────────────────────
    num_filters: int = 256
    num_res_blocks: int = 15
    se_reduction: int = 8
    value_head_hidden: int = 512

    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 256
    learning_rate: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    num_epochs: int = 2              # passes over the full dataset
    policy_weight: float = 1.0       # weight for policy (CE) loss
    value_weight: float = 1.0        # weight for value (MSE) loss

    # ── Data ──────────────────────────────────────────────────────────
    csv_path: str = "chess_games_2000.csv"
    max_games: int = 0               # 0 = use all games
    min_elo: int = 1800              # only learn from higher-rated games
    val_split: float = 0.02          # fraction held out for validation
    chunk_size: int = 50_000         # CSV rows to read per chunk
    positions_per_game: int = 0      # 0 = all positions, N = sample N per game
    num_data_workers: int = 2        # DataLoader workers (low = less RAM)
    chunk_positions: int = 100_000   # positions loaded into RAM at once during training

    # ── Checkpointing ─────────────────────────────────────────────────
    save_every_steps: int = 5000     # save checkpoint every N gradient steps
    log_every_steps: int = 100       # print metrics every N steps
    eval_every_steps: int = 1000     # run validation every N steps

    # ── Device ────────────────────────────────────────────────────────
    device: str = "mps"

    # ── Paths ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "SL/checkpoints"
    log_dir: str = "SL/logs"
