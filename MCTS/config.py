"""
MCTS/config.py — Hyperparameters for AlphaZero MCTS Chess training.

Based on "Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm" (Silver et al., 2017).

Pipeline:
  1. Load SL pre-trained model f_θ(s) → (p, v)      (trained in SL/)
  2. Self-play with MCTS → collect (s, π, z) data
  3. Train f_θ on self-play data:  L = (z−v)² − π⊤log p + c‖θ‖²
  4. Repeat from step 2
"""


class Config:
    # ── State encoding (paper: 8×8×119 input) ─────────────────────────
    history_length: int = 8
    input_planes: int = 119          # T*14 + 7 = 8*14 + 7
    board_size: int = 8

    # ── Action encoding (paper: 8×8×73 output planes) ────────────────
    action_planes: int = 73
    policy_size: int = board_size * board_size * action_planes  # 4672

    # ── Neural network f_θ(s) → (p, v) ───────────────────────────────
    num_filters: int = 256           # match SL pretrained model
    num_res_blocks: int = 15         # 15 residual blocks (SE-Net)
    se_reduction: int = 8            # SE-block squeeze ratio
    value_head_hidden: int = 512     # match SL pretrained model

    # ── MCTS (paper: "each search consists of 800 simulations") ───────
    num_mcts_sims: int = 800         # simulations per move
    mcts_leaf_batch: int = 8         # leaves to batch-evaluate per step

    # Exploration — dynamic cPUCT (AlphaZero / MuZero style)
    # c(s) = ln((1 + N(s) + cpuct_base) / cpuct_base) + cpuct_init
    c_puct: float = 2.5             # static fallback when dynamic disabled
    cpuct_base: float = 19652       # dynamic cPUCT log-base
    cpuct_init: float = 2.5         # dynamic cPUCT initial value
    use_dynamic_cpuct: bool = True   # enable dynamic exploration scaling

    # First Play Urgency — unvisited children use Q = parent_Q − reduction
    # (prevents wasting simulations on clearly bad unexplored moves)
    fpu_reduction: float = 0.25

    # Dirichlet noise at root (paper: α=0.3, ε=0.25 for chess)
    # "scaled in proportion to the typical number of legal moves"
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Temperature: τ=1 for first N moves, then τ→0  (paper: N=30)
    temperature: float = 1.0
    temp_threshold_move: int = 30

    # ── Legacy (backward-compat, not used in AlphaZero mode) ─────────
    lambda_mix: float = 0.0          # V = (1-λ)·vθ + λ·zL  (0 = pure value)
    rollout_max_depth: int = 200     # unused when λ=0
    rollout_filters: int = 32        # RolloutPolicy backward compat
    rollout_blocks: int = 2          # RolloutPolicy backward compat

    # ── Self-play ─────────────────────────────────────────────────────
    self_play_games: int = 500       # games per iteration
    max_game_moves: int = 200        # max moves per game (resign/adjudicate)

    # ── Resign ────────────────────────────────────────────────────────
    resign_threshold: float = -0.9   # resign when V < threshold
    resign_consecutive: int = 10     # must be below threshold N times
    resign_enabled_after_iter: int = 10  # no resign in early iterations
    resign_check_fraction: float = 0.1   # 10% of games play out despite resign

    # ── Disk-based data management ────────────────────────────────────
    data_window: int = 20            # use last N iterations' data
    save_data_every_iter: bool = True
    replay_buffer_size: int = 500_000

    # ── Training (paper: L = (z−v)² − π⊤log p + c‖θ‖²) ──────────────
    optimizer: str = "sgd"           # "sgd" (paper) or "adam"
    batch_size: int = 256
    train_steps: int = 1000          # gradient steps per iteration
    learning_rate: float = 0.01      # SGD initial LR (scaled for bs=256)
    lr_min: float = 1e-5             # minimum LR floor
    lr_milestones: tuple = (30, 60, 80)   # iteration milestones for LR drop
    lr_gamma: float = 0.1            # LR multiplier at each milestone
    momentum: float = 0.9            # SGD momentum (paper: 0.9)
    nesterov: bool = True            # Nesterov accelerated gradient
    weight_decay: float = 1e-4       # L2 regularisation c (paper: 1e-4)
    grad_clip: float = 1.0
    warmup_steps: int = 200          # linear warmup before main schedule

    # ── Evaluation ────────────────────────────────────────────────────
    eval_games: int = 40             # games per evaluation round
    accept_threshold: float = 0.55   # win-rate to accept new model
    eval_mcts_sims: int = 400        # sims during evaluation
    eval_start_iter: int = 3         # skip eval for first few iters
    use_eval_gating: bool = False    # paper: no gating (always use latest)

    # ── Main loop ─────────────────────────────────────────────────────
    num_iterations: int = 5

    # ── Device ────────────────────────────────────────────────────────
    device: str = "mps"

    # ── Parallel ──────────────────────────────────────────────────────
    num_workers: int = 4             # self-play worker processes

    # ── Paths ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "MCTS/checkpoints"
    log_dir: str = "MCTS/logs"
    data_dir: str = "MCTS/data"
    sl_model_path: str = "SL/checkpoints/final_model.pt"
