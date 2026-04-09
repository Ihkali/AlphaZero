"""
SL/main.py — Supervised-learning training orchestrator.

Pipeline:
  1. Parse chess_games.csv → build numpy cache (one-time)
  2. Create DataLoaders from mmap'd cache
  3. Train AlphaZeroNet with policy CE + value MSE
  4. Save checkpoints periodically

Usage:
    python SL/main.py                          # full training
    python SL/main.py --quick                  # tiny test run
    python SL/main.py --resume latest.pt       # resume from checkpoint
    python SL/main.py --max-games 100000       # limit dataset size
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch
from torch.utils.data import DataLoader

from SL.config import Config
from SL.model import AlphaZeroNet, load_checkpoint
from SL.dataset import build_cache, make_datasets, chunk_iterator
from SL.train import Trainer


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(Config.log_dir, exist_ok=True)
    logger = logging.getLogger("sl")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(os.path.join(Config.log_dir, "sl_training.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Supervised Learning Chess Trainer")
    p.add_argument("--quick", action="store_true",
                   help="Tiny run for testing (1000 games, 1 epoch)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--csv", type=str, default=Config.csv_path,
                   help="Path to chess_games_2000.csv")
    p.add_argument("--max-games", type=int, default=None,
                   help="Max games to use (0=all)")
    p.add_argument("--min-elo", type=int, default=None,
                   help="Minimum Elo filter for both players")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (0 = main process only, saves RAM)")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force rebuild of numpy cache")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    logger = setup_logging()

    # Apply CLI overrides
    if args.max_games is not None:
        Config.max_games = args.max_games
    if args.min_elo is not None:
        Config.min_elo = args.min_elo
    if args.epochs is not None:
        Config.num_epochs = args.epochs
    if args.batch_size is not None:
        Config.batch_size = args.batch_size
    if args.lr is not None:
        Config.learning_rate = args.lr
    if args.num_workers is not None:
        Config.num_data_workers = args.num_workers

    if args.quick:
        Config.max_games = 1000
        Config.num_epochs = 1
        Config.min_elo = 0
        Config.log_every_steps = 10
        Config.eval_every_steps = 50
        Config.save_every_steps = 100
        logger.info("Quick mode: 1000 games, 1 epoch")

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # Ensure dirs
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)

    # ── Step 1: Build cache ───────────────────────────────────────────
    if args.rebuild_cache:
        import shutil
        cache_dir = "SL/data"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info("Cleared old cache.")

    n_positions = build_cache(
        csv_path=args.csv,
        max_games=Config.max_games,
        min_elo=Config.min_elo,
    )
    logger.info(f"Total positions: {n_positions:,}")

    # ── Step 2: Model ─────────────────────────────────────────────────
    model = AlphaZeroNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    start_step = 0
    if args.resume:
        ckpt_path = args.resume
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(Config.checkpoint_dir, ckpt_path)
        if os.path.isfile(ckpt_path):
            data = load_checkpoint(model, ckpt_path, device=device)
            start_step = data.get("global_step", 0)
            logger.info(f"Resumed from {ckpt_path} (step {start_step})")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path} — training from scratch")

    # ── Step 3: Chunked training ──────────────────────────────────────
    #  Load one chunk at a time into RAM, train on it, free it, repeat.
    #  This keeps peak RAM at ~3 GB regardless of total dataset size.

    logger.info(f"Training for {Config.num_epochs} epochs, "
                f"batch_size={Config.batch_size}, lr={Config.learning_rate}, "
                f"chunk={Config.chunk_positions:,} positions")
    logger.info("=" * 70)

    trainer = None  # created on first chunk (needs a loader for LR schedule)

    for epoch in range(1, Config.num_epochs + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{Config.num_epochs}")
        logger.info(f"{'='*70}")

        for train_ds, val_ds, ci, n_chunks in chunk_iterator():
            train_loader = DataLoader(
                train_ds,
                batch_size=Config.batch_size,
                shuffle=True,
                num_workers=Config.num_data_workers,
                pin_memory=False,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.batch_size * 2,
                shuffle=False,
                num_workers=Config.num_data_workers,
                pin_memory=False,
            )

            if trainer is None:
                # Estimate total steps across all epochs & chunks for LR schedule
                est_batches_per_chunk = max(1, len(train_loader))
                est_total = est_batches_per_chunk * n_chunks * Config.num_epochs
                # Temporarily set total so the Trainer LR schedule is correct
                _orig_len = train_loader.__class__.__len__
                class _FakeLoader:
                    """Thin wrapper so Trainer sees total_steps correctly."""
                    def __init__(self, real, total_batches):
                        self._real = real
                        self._total = total_batches
                    def __len__(self):
                        return self._total
                    def __iter__(self):
                        return iter(self._real)

                fake_loader = _FakeLoader(train_loader, est_total)
                trainer = Trainer(model, fake_loader, val_loader, device,
                                  start_step=start_step)
            else:
                trainer.train_loader = train_loader
                trainer.val_loader = val_loader

            trainer.train_epoch(epoch)

            # Free chunk data before loading next
            del train_loader, val_loader

    # Final save
    from SL.model import save_checkpoint as sc
    final_path = os.path.join(Config.checkpoint_dir, "final_model.pt")
    sc(model, final_path, optimizer=trainer.optimizer,
       extra={"global_step": trainer.global_step})
    logger.info(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
