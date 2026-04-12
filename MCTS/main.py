"""
MCTS/main.py — AlphaZero training pipeline orchestrator.

Implements the training loop from "Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning Algorithm"
(Silver et al., 2017):

  1. Load SL pre-trained model f_θ(s) → (p, v)
  2. Self-play with MCTS → collect (s, π, z) triples
  3. Train f_θ:  L = (z − v)² − π⊤ log p + c‖θ‖²
  4. (Optional) Evaluate new model vs best; accept/reject
  5. Repeat from step 2

The SL model provides a strong initialisation so that the MCTS
self-play loop starts from a competent policy rather than random play.

Usage:
    python MCTS/main.py                  # full training
    python MCTS/main.py --quick          # quick smoke-test
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import time
import logging
import numpy as np
import torch
import torch.multiprocessing as mp

from MCTS.config import Config
from MCTS.model import AlphaZeroNet, save_checkpoint, load_checkpoint
from MCTS.self_play import run_self_play, DiskReplayBuffer
from MCTS.train import Trainer
from MCTS.evaluate import evaluate_models
from MCTS.utils import get_device, ensure_dirs, Timer, format_time


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these instead of using CLI flags
# ═══════════════════════════════════════════════════════════════════════════

QUICK_MODE       = False         # True = tiny run for testing
SL_MODEL_PATH    = None          # path to SL model (None = Config default)


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "alphazero.log")

    logger = logging.getLogger("alphazero")
    logger.setLevel(logging.DEBUG)

    # Don't add duplicate handlers
    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
#  SL MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_sl_model(path: str, device: str) -> AlphaZeroNet:
    """Load the SL-pretrained model, auto-detecting its architecture."""
    data = torch.load(path, map_location=device, weights_only=False)
    state = data["model_state"]
    num_filters = state["conv_block.conv.weight"].shape[0]
    num_blocks = sum(1 for k in state if k.endswith(".conv1.weight")
                     and k.startswith("res_blocks."))
    print(f"  SL model arch: {num_filters} filters, {num_blocks} blocks")
    net = AlphaZeroNet(
        num_filters=num_filters,
        num_blocks=num_blocks,
    ).to(device)
    net.load_state_dict(state)
    return net


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT RESUME
# ═══════════════════════════════════════════════════════════════════════════

def try_resume(ckpt_dir: str, device: str):
    """Try to resume from latest.pt.  Returns (net, metadata) or None."""
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.isfile(latest_path):
        return None
    try:
        data = torch.load(latest_path, map_location=device, weights_only=False)
        state = data["model_state"]
        num_filters = state["conv_block.conv.weight"].shape[0]
        num_blocks = sum(1 for k in state if k.endswith(".conv1.weight")
                         and k.startswith("res_blocks."))
        net = AlphaZeroNet(num_filters=num_filters,
                           num_blocks=num_blocks).to(device)
        net.load_state_dict(state)
        return net, data
    except Exception as e:
        print(f"  ⚠ Could not resume from {latest_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ALPHAZERO LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = get_device()
    ensure_dirs(Config.checkpoint_dir, Config.log_dir, Config.data_dir)
    log = setup_logging(Config.log_dir)

    sl_model_path = SL_MODEL_PATH or Config.sl_model_path

    # ── Quick mode overrides ──────────────────────────────────────────
    if QUICK_MODE:
        num_iterations   = 3
        self_play_games  = 10
        train_steps      = 50
        eval_games       = 4
        num_sims         = 100
        eval_mcts_sims   = 50
    else:
        num_iterations   = Config.num_iterations
        self_play_games  = Config.self_play_games
        train_steps      = Config.train_steps
        eval_games       = Config.eval_games
        num_sims         = Config.num_mcts_sims
        eval_mcts_sims   = Config.eval_mcts_sims

    # ── Banner ────────────────────────────────────────────────────────
    banner = f"""
{'=' * 64}
  AlphaZero Training Pipeline (Chess)
  Silver et al., 2017 — with SL initialisation
{'=' * 64}
  Device           : {device}
  SL model         : {sl_model_path}
  Iterations       : {num_iterations}
  Self-play games  : {self_play_games}
  MCTS simulations : {num_sims}
  Train steps/iter : {train_steps}
  Batch size       : {Config.batch_size}
  Optimizer        : {Config.optimizer} (lr={Config.learning_rate})
  Data window      : {Config.data_window} iterations
  Eval gating      : {Config.use_eval_gating}
  FPU reduction    : {Config.fpu_reduction}
  Dynamic cPUCT    : {Config.use_dynamic_cpuct}
  Network          : {Config.num_filters} filters, {Config.num_res_blocks} blocks (SE-Net)
{'=' * 64}"""
    print(banner)
    log.info(banner)

    # Save config snapshot
    config_snapshot = {k: v for k, v in vars(Config).items()
                       if not k.startswith("_")}
    config_path = os.path.join(Config.log_dir, "config_snapshot.json")
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)

    total_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    #  STAGE 1: Load SL model f_θ (single network for both p and v)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 64}")
    print("  Stage 1: Loading SL pre-trained model f_θ(s) → (p, v)")
    print(f"{'=' * 64}")

    # Check for resume
    start_iter = 0
    resumed = try_resume(Config.checkpoint_dir, device)
    if resumed is not None:
        net, meta = resumed
        start_iter = meta.get("iteration", 0) + 1
        print(f"  ✓ Resumed from iteration {start_iter}")
        log.info(f"Resumed from iteration {start_iter}")
    elif os.path.isfile(sl_model_path):
        net = load_sl_model(sl_model_path, device)
        print(f"  ✓ Loaded SL model from {sl_model_path}")
    else:
        print(f"  ⚠ No SL model at {sl_model_path} — starting from scratch")
        net = AlphaZeroNet().to(device)
    net.eval()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Best network (for evaluation gating)
    best_net = AlphaZeroNet(
        num_filters=net.conv_block.conv.weight.shape[0],
        num_blocks=len(net.res_blocks),
    ).to(device)
    best_net.load_state_dict(net.state_dict())
    best_net.eval()

    # Create trainer — persists across iterations for optimizer momentum
    total_train_steps = num_iterations * train_steps
    trainer = Trainer(
        net, device=device,
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
        optimizer_type=Config.optimizer,
        momentum=Config.momentum,
        total_steps=total_train_steps,
    )

    # Disk replay buffer
    replay = DiskReplayBuffer(Config.data_dir, window=Config.data_window)

    # Stats tracking
    iter_stats = []

    # ══════════════════════════════════════════════════════════════════
    #  MAIN LOOP — iterative self-play + training  (paper Section 1)
    #
    #  "Self-play games are generated by using the latest parameters
    #   for this neural network."
    # ══════════════════════════════════════════════════════════════════
    for iteration in range(start_iter, num_iterations):
        iter_start = time.time()

        print(f"\n{'=' * 64}")
        print(f"  Iteration {iteration + 1} / {num_iterations}")
        print(f"{'=' * 64}")

        # ── 1. SELF-PLAY WITH MCTS ───────────────────────────────────
        # Paper: "Games are played by selecting moves for both players
        # by MCTS, at ≈ π_t"
        print(f"\n  ── Self-play ({self_play_games} games, "
              f"{num_sims} sims/move) ──")

        resign_thresh = (Config.resign_threshold
                         if iteration >= Config.resign_enabled_after_iter
                         else -1.0)

        # Use best_net for self-play when gating; otherwise latest net
        play_net = best_net if Config.use_eval_gating else net

        sp_stats = run_self_play(
            play_net,
            num_games=self_play_games,
            num_sims=num_sims,
            device=device,              # inference server uses GPU/MPS
            num_workers=Config.num_workers,
            verbose=True,
            resign_threshold=resign_thresh,
            iteration=iteration,
            data_dir=Config.data_dir,
        )

        # ── 2. TRAIN ON REPLAY BUFFER ────────────────────────────────
        # Paper Eq. 1: L = (z − v)² − π⊤ log p + c‖θ‖²
        print(f"\n  ── Training ({train_steps} steps, "
              f"bs={Config.batch_size}) ──")

        replay.refresh()
        if len(replay) > 0:
            history = trainer.train(
                replay,
                num_steps=train_steps,
                batch_size=Config.batch_size,
                verbose=True,
            )
        else:
            print("  ⚠ Empty replay buffer — skipping training")
            history = {"total": [], "policy": [], "value": []}

        # ── 3. EVALUATION  (optional gating) ─────────────────────────
        # Paper: "AlphaZero simply maintains a single neural network
        # that is updated continually" (no gating).
        # We offer optional gating for safety.
        accepted = True
        if Config.use_eval_gating and iteration >= Config.eval_start_iter:
            print(f"\n  ── Evaluation ({eval_games} games, "
                  f"{eval_mcts_sims} sims) ──")
            eval_result = evaluate_models(
                net, best_net,
                num_games=eval_games,
                num_sims=eval_mcts_sims,
                device=device,
                verbose=True,
            )
            accepted = eval_result["accepted"]
            if accepted:
                best_net.load_state_dict(net.state_dict())
                print("  ✓ New model ACCEPTED — updating best")
            else:
                net.load_state_dict(best_net.state_dict())
                print("  ✗ New model REJECTED — reverting to best")
        else:
            # No gating: always use latest (paper's approach)
            best_net.load_state_dict(net.state_dict())

        # ── 4. CHECKPOINT ─────────────────────────────────────────────
        save_checkpoint(
            best_net,
            os.path.join(Config.checkpoint_dir, "latest.pt"),
            extra={"iteration": iteration,
                   "global_step": trainer.global_step},
        )

        # Save a checkpoint every iteration
        ckpt_path = os.path.join(
            Config.checkpoint_dir,
            f"weights_iter_{iteration:04d}.pt")
        save_checkpoint(best_net, ckpt_path,
                        extra={"iteration": iteration,
                               "global_step": trainer.global_step})
        print(f"  Saved checkpoint: {ckpt_path}")

        # ── ITERATION SUMMARY ─────────────────────────────────────────
        iter_elapsed = time.time() - iter_start
        avg_loss = (np.mean(history["total"][-100:])
                    if history["total"] else 0.0)
        avg_ploss = (np.mean(history["policy"][-100:])
                     if history["policy"] else 0.0)
        avg_vloss = (np.mean(history["value"][-100:])
                     if history["value"] else 0.0)

        summary = {
            "iteration": iteration + 1,
            "self_play_games": sp_stats.get("total_games", 0),
            "total_examples": sp_stats.get("total_examples", 0),
            "avg_game_length": sp_stats.get("avg_length", 0),
            "replay_size": len(replay),
            "avg_loss": avg_loss,
            "avg_policy_loss": avg_ploss,
            "avg_value_loss": avg_vloss,
            "lr": trainer.scheduler.get_lr(),
            "global_step": trainer.global_step,
            "accepted": accepted,
            "elapsed": iter_elapsed,
        }
        iter_stats.append(summary)

        msg = (f"\n  ┌─────────────────────────────────────────┐\n"
               f"  │  Iteration {iteration+1:3d} Summary"
               f"{'':>20s}│\n"
               f"  ├─────────────────────────────────────────┤\n"
               f"  │  Games played  : {sp_stats.get('total_games', 0):6d}"
               f"{'':>15s}│\n"
               f"  │  Training data : {len(replay):6d} positions"
               f"{'':>8s}│\n"
               f"  │  Loss (π/v/tot): {avg_ploss:.4f} / "
               f"{avg_vloss:.4f} / {avg_loss:.4f}  │\n"
               f"  │  LR            : {trainer.scheduler.get_lr():.2e}"
               f"{'':>15s}│\n"
               f"  │  Global step   : {trainer.global_step:6d}"
               f"{'':>15s}│\n"
               f"  │  Accepted      : {'Yes' if accepted else 'No'}"
               f"{'':>19s}│\n"
               f"  │  Time          : {format_time(iter_elapsed)}"
               f"{'':>15s}│\n"
               f"  └─────────────────────────────────────────┘")
        print(msg)
        log.info(msg)

    # ══════════════════════════════════════════════════════════════════
    #  FINAL SAVE
    # ══════════════════════════════════════════════════════════════════
    final_path = os.path.join(Config.checkpoint_dir, "final_model.pt")
    save_checkpoint(best_net, final_path,
                    extra={"iterations": num_iterations,
                           "global_step": trainer.global_step})

    # Also copy to top-level data/ for easy access
    data_final = os.path.join("data", "final_model.pt")
    os.makedirs("data", exist_ok=True)
    save_checkpoint(best_net, data_final,
                    extra={"iterations": num_iterations,
                           "global_step": trainer.global_step})

    total_elapsed = time.time() - total_start

    # Save training stats
    stats_path = os.path.join(Config.log_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(iter_stats, f, indent=2, default=str)

    final_msg = f"""
{'=' * 64}
  ✓ AlphaZero Training Complete!
  ├─ SL initialisation   : {sl_model_path}
  ├─ Final model          : {final_path}
  ├─ Latest checkpoint    : {os.path.join(Config.checkpoint_dir, 'latest.pt')}
  ├─ Iterations completed : {num_iterations - start_iter}
  ├─ Total gradient steps : {trainer.global_step}
  ├─ Total time           : {format_time(total_elapsed)}
  └─ Play: python MCTS/play_gui.py
{'=' * 64}"""
    print(final_msg)
    log.info(final_msg)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Parse simple CLI flags
    if "--quick" in sys.argv:
        QUICK_MODE = True
    for i, arg in enumerate(sys.argv):
        if arg == "--sl-model" and i + 1 < len(sys.argv):
            SL_MODEL_PATH = sys.argv[i + 1]

    main()
