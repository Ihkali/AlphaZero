"""
MCTS/main.py — AlphaZero MCTS training orchestrator.

Starts from an SL-pretrained model (data/sl_model.pt) and refines it
via self-play + MCTS.

Data pipeline:
  1. Self-play (800 sims/move) → save to MCTS/data/iter_NNNN.npz
  2. DiskReplayBuffer loads last N files → trains on that window
  3. Save checkpoint → repeat

Usage:
    python MCTS/main.py                      # 3 iterations from SL model
    python MCTS/main.py --quick              # quick test (2 iters, tiny)
    python MCTS/main.py --resume PATH        # resume from checkpoint
    python MCTS/main.py --sl-model data/sl_model.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

from MCTS.config import Config
from MCTS.model import AlphaZeroNet, save_checkpoint, load_checkpoint
from MCTS.self_play import (
    run_self_play,
    DiskReplayBuffer,
)
from MCTS.train import Trainer
from MCTS.utils import get_device, ensure_dirs, Timer, Logger, format_time


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str):
    """Configure both file and console logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "alphazero.log")

    logger = logging.getLogger("alphazero")
    logger.setLevel(logging.DEBUG)

    # File handler — detailed
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

    # Console handler — concise
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
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero MCTS Chess Training")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny run for testing")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--sl-model", type=str, default=None,
                        help="Path to SL pretrained model (default: data/sl_model.pt)")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--self-play-games", type=int, default=None)
    parser.add_argument("--mcts-sims", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data-window", type=int, default=None,
                        help="Number of past iterations to train on")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = get_device()
    ensure_dirs(Config.checkpoint_dir, Config.log_dir, Config.data_dir)

    log = setup_logging(Config.log_dir)

    if args.quick:
        num_iterations = 2
        sp_games = 4
        mcts_sims = 25
        train_steps = 20
        num_workers = 2
        data_window = 2
    else:
        num_iterations = args.iterations or Config.num_iterations
        sp_games = args.self_play_games or Config.self_play_games
        mcts_sims = args.mcts_sims or Config.num_mcts_sims
        train_steps = args.train_steps or Config.train_steps
        num_workers = args.workers or Config.num_workers
        data_window = args.data_window or Config.data_window

    sims_label = str(mcts_sims)
    sl_model_path = args.sl_model or Config.sl_model_path

    # ── Print banner ──────────────────────────────────────────────────
    banner = f"""
{'=' * 64}
  AlphaZero MCTS Chess Training
{'=' * 64}
  Device         : {device}
  Workers        : {num_workers} parallel processes
  Iterations     : {num_iterations}
  Self-play games: {sp_games} × {sims_label} MCTS sims/move
  Max game moves : {Config.max_game_moves}
  Leaf batch     : {Config.mcts_leaf_batch} (batched NN inference)
  Train steps    : {train_steps}  (batch {Config.batch_size})
  Data window    : last {data_window} iterations
  Evaluation     : manual (play_gui.py)
  Network        : {Config.num_filters} filters, {Config.num_res_blocks} res blocks (SE-Net)
  SL pretrained  : {sl_model_path}
  Data directory : {Config.data_dir}
  Log directory  : {Config.log_dir}
{'=' * 64}"""
    print(banner)
    log.info(banner)

    # ── Save config snapshot ──────────────────────────────────────────
    config_snapshot = {k: v for k, v in vars(Config).items()
                       if not k.startswith("_")}
    config_path = os.path.join(Config.log_dir, "config_snapshot.json")
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)
    log.info(f"Config saved to {config_path}")

    # ── Build models (from SL pretrained weights) ────────────────────
    start_iter = 0

    if args.resume:
        log.info(f"Resuming from {args.resume}...")
        current_net = load_sl_model(args.resume, device)  # uses same loader
        data = torch.load(args.resume, map_location=device, weights_only=False)
        start_iter = data.get("iteration", 0) + 1
        log.info(f"  Resumed at iteration {start_iter}")
    elif os.path.isfile(sl_model_path):
        log.info(f"Loading SL pretrained model: {sl_model_path}")
        current_net = load_sl_model(sl_model_path, device)
        log.info("  SL weights loaded successfully.")
    else:
        log.info(f"  No SL model found at {sl_model_path} — starting from scratch.")
        current_net = AlphaZeroNet().to(device)

    best_net = AlphaZeroNet(
        num_filters=current_net.conv_block.conv.weight.shape[0],
        num_blocks=len(current_net.res_blocks),
    ).to(device)
    best_net.load_state_dict(current_net.state_dict())

    disk_buffer = DiskReplayBuffer(
        data_dir=Config.data_dir, window=data_window)
    trainer = Trainer(current_net, device=device)
    csv_logger = Logger(os.path.join(Config.log_dir, "training_log.csv"))

    total_params = sum(p.numel() for p in current_net.parameters())
    trainable = sum(p.numel() for p in current_net.parameters() if p.requires_grad)
    log.info(f"  Model: {total_params:,} params ({trainable:,} trainable)")

    end_iter = start_iter + num_iterations
    cumulative_time = 0.0
    all_losses = []

    for iteration in range(start_iter, end_iter):
        iter_start = time.time()
        done = iteration - start_iter
        remaining = end_iter - iteration
        pct = done / num_iterations * 100

        header = (
            f"\n{'═' * 64}\n"
            f"  ITERATION {iteration + 1} / {end_iter}   "
            f"[{'█' * (done * 20 // max(1, num_iterations))}"
            f"{'░' * (20 - done * 20 // max(1, num_iterations))}]  "
            f"{pct:.0f}%"
        )
        if done > 0:
            avg_iter_time = cumulative_time / done
            eta = avg_iter_time * remaining
            header += (
                f"\n  ETA: ~{format_time(eta)}  "
                f"(avg {format_time(avg_iter_time)}/iter, "
                f"elapsed {format_time(cumulative_time)})"
            )
        header += f"\n{'═' * 64}"
        print(header)
        log.info(header)

        # ── A. SELF-PLAY ──────────────────────────────────────────────
        log.info(f"  [A] Self-play ({sp_games} games, "
                 f"{mcts_sims} sims/move, {num_workers} workers)...")
        best_net.eval()

        # Decide resign threshold
        resign_thr = -1.0  # disabled
        if iteration >= Config.resign_enabled_after_iter:
            resign_thr = Config.resign_threshold

        with Timer("Self-play") as sp_timer:
            sp_stats = run_self_play(
                best_net, num_games=sp_games, num_sims=mcts_sims,
                device=device, num_workers=num_workers, verbose=True,
                resign_threshold=resign_thr,
                iteration=iteration, data_dir=Config.data_dir,
            )
        log.info(f"  Data streamed to disk: {sp_stats['total_examples']} examples")

        # ── B. TRAINING ───────────────────────────────────────────────
        log.info(f"  [B] Training ({train_steps} steps, "
                 f"window={data_window} iters)...")
        current_net.load_state_dict(best_net.state_dict())
        trainer = Trainer(current_net, device=device)

        # Refresh disk buffer (loads last N iterations)
        disk_buffer.refresh()

        with Timer("Training") as tr_timer:
            history = trainer.train(
                disk_buffer, num_steps=train_steps,
                batch_size=min(Config.batch_size, len(disk_buffer)),
            )

        avg_loss = np.mean(history["total"][-100:]) if history["total"] else 0
        avg_ploss = np.mean(history["policy"][-100:]) if history["policy"] else 0
        avg_vloss = np.mean(history["value"][-100:]) if history["value"] else 0
        all_losses.append(avg_loss)

        # ── C. SAVE MODEL (always accept — eval is manual) ───────────
        best_net.load_state_dict(current_net.state_dict())
        ckpt_path = os.path.join(
            Config.checkpoint_dir, f"weights_iter_{iteration:04d}.pt")
        save_checkpoint(best_net, ckpt_path,
                        extra={"iteration": iteration,
                               "loss": avg_loss})
        # Always keep a latest.pt for play_gui / resume
        latest_path = os.path.join(Config.checkpoint_dir, "latest.pt")
        save_checkpoint(best_net, latest_path,
                        extra={"iteration": iteration,
                               "loss": avg_loss})
        log.info(f"  💾 Checkpoint saved: {ckpt_path}  (+latest.pt)")

        # ── D. ITERATION SUMMARY ──────────────────────────────────────
        iter_elapsed = time.time() - iter_start
        cumulative_time += iter_elapsed

        summary = (
            f"\n  ┌─ Iteration {iteration + 1} Summary "
            f"{'─' * 36}\n"
            f"  │ Loss: {avg_loss:.4f}  "
            f"(policy {avg_ploss:.4f} + value {avg_vloss:.4f})\n"
            f"  │ Self-play: "
            f"W{sp_stats['White wins']}/B{sp_stats['Black wins']}"
            f"/D{sp_stats['Draw']}  "
            f"avg {sp_stats['avg_length']:.0f} moves  "
            f"({sp_stats['total_examples']} examples)\n"
        )
        summary += (
            f"  │ Checkpoint: {ckpt_path}\n"
            f"  │ Disk data files: {len(os.listdir(Config.data_dir))} "
            f"in {Config.data_dir}/\n"
            f"  │ Time: {format_time(iter_elapsed)}  "
            f"(total {format_time(cumulative_time)})\n"
            f"  └{'─' * 55}"
        )
        print(summary)
        log.info(summary)

        # ── E. CSV LOGGING ────────────────────────────────────────────
        csv_logger.log({
            "iteration": iteration,
            "sp_white_wins": sp_stats["White wins"],
            "sp_black_wins": sp_stats["Black wins"],
            "sp_draws": sp_stats["Draw"],
            "sp_avg_length": f"{sp_stats['avg_length']:.1f}",
            "sp_examples": sp_stats['total_examples'],
            "buffer_examples": len(disk_buffer),
            "avg_loss": f"{avg_loss:.4f}",
            "avg_policy_loss": f"{avg_ploss:.4f}",
            "avg_value_loss": f"{avg_vloss:.4f}",
            "iter_time_s": f"{iter_elapsed:.1f}",
            "cumulative_time_s": f"{cumulative_time:.1f}",
        })

    # ── FINAL ─────────────────────────────────────────────────────────
    final_path = os.path.join(Config.checkpoint_dir, "final_model.pt")
    save_checkpoint(best_net, final_path,
                    extra={"iteration": end_iter - 1})

    final_msg = f"""
{'═' * 64}
  ✓ Training complete!
  ├─ Iterations   : {num_iterations}
  ├─ Total time   : {format_time(cumulative_time)}
  ├─ Final model  : {final_path}
  ├─ Data files   : {Config.data_dir}/ ({len(os.listdir(Config.data_dir))} files)
  ├─ CSV log      : {os.path.join(Config.log_dir, 'training_log.csv')}
  ├─ Full log     : {os.path.join(Config.log_dir, 'alphazero.log')}
  └─ Play against : python MCTS/play_gui.py
{'═' * 64}"""
    print(final_msg)
    log.info(final_msg)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
