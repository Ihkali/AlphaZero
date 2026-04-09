"""
MCTS/config.py — Hyperparameters for AlphaZero MCTS Chess training.
"""


class Config:
    # ── State encoding ────────────────────────────────────────────────
    history_length: int = 8
    input_planes: int = 119          # T*14 + 7 = 8*14 + 7
    board_size: int = 8

    # ── Action encoding ───────────────────────────────────────────────
    action_planes: int = 73
    policy_size: int = board_size * board_size * action_planes  # 4672

    # ── Neural network ────────────────────────────────────────────────
    num_filters: int = 256           # match SL pretrained model
    num_res_blocks: int = 15         # 15 residual blocks (SE-Net)
    se_reduction: int = 8            # SE-block squeeze ratio
    value_head_hidden: int = 512     # match SL pretrained model

    # ── MCTS ──────────────────────────────────────────────────────────
    num_mcts_sims: int = 800         # 800 simulations per move
    mcts_leaf_batch: int = 8         # leaves to batch-evaluate per MCTS step
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temp_threshold_move: int = 30
    fpu_reduction: float = 0.25      # first-play urgency

    # ── Self-play ─────────────────────────────────────────────────────
    self_play_games: int = 500       # 500 episodes per iteration
    max_game_moves: int = 200        # max 200 moves per game
    replay_buffer_size: int = 500_000  # large in-memory cap (disk is primary)

    # ── Disk-based data management ────────────────────────────────────
    data_window: int = 10            # use last N iterations' data for training
    save_data_every_iter: bool = True  # always persist self-play data
    # ── Training ──────────────────────────────────────────────────────
    batch_size: int = 256
    train_steps: int = 500
    learning_rate: float = 2e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 200          # linear warmup before cosine decay

    # ── Evaluation ────────────────────────────────────────────────────
    eval_games: int = 10
    accept_threshold: float = 0.55
    eval_mcts_sims: int = 100
    eval_start_iter: int = 3         # skip eval for first few iters

    # ── Main loop ─────────────────────────────────────────────────────
    num_iterations: int = 3          # 3 iterations total
    resign_threshold: float = -0.9
    resign_consecutive: int = 10
    resign_enabled_after_iter: int = 10  # don't resign early on

    # ── Device ────────────────────────────────────────────────────────
    device: str = "mps"

    # ── Parallel ──────────────────────────────────────────────────────
    num_workers: int = 4             # max 4 processes (RAM-limited)

    # ── Paths ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "MCTS/checkpoints"
    log_dir: str = "MCTS/logs"
    data_dir: str = "MCTS/data"
    sl_model_path: str = "data/sl_model.pt"   # SL pretrained starting weights
