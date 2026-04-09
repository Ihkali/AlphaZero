"""
SL/train.py — Supervised-learning trainer (policy CE + value MSE).
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from SL.config import Config
from SL.model import AlphaZeroNet, save_checkpoint

logger = logging.getLogger("sl")


class Trainer:
    def __init__(self, model: AlphaZeroNet, train_loader: DataLoader,
                 val_loader: DataLoader, device: str, start_step: int = 0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.global_step = start_step

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # Cosine schedule with warmup
        total_steps = Config.num_epochs * len(train_loader)
        self.total_steps = total_steps

        def lr_lambda(step):
            if step < Config.warmup_steps:
                return step / max(1, Config.warmup_steps)
            progress = (step - Config.warmup_steps) / max(1, total_steps - Config.warmup_steps)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return max(Config.lr_min / Config.learning_rate, cosine)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda, last_epoch=start_step - 1 if start_step > 0 else -1
        )

        self.best_val_loss = float("inf")

    # ── Training ──────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}",
                    unit="batch", dynamic_ncols=True, leave=True)

        for batch_idx, (states, moves, values) in enumerate(pbar):
            states = states.to(self.device)
            moves = moves.to(self.device)
            values = values.to(self.device)

            p_logits, v_pred = self.model(states)

            policy_loss = F.cross_entropy(p_logits, moves)
            value_loss = F.mse_loss(v_pred.squeeze(-1), values)
            loss = (Config.policy_weight * policy_loss
                    + Config.value_weight * value_loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Metrics
            bs = states.size(0)
            epoch_policy_loss += policy_loss.item() * bs
            epoch_value_loss += value_loss.item() * bs
            epoch_total_loss += loss.item() * bs
            preds = p_logits.argmax(dim=1)
            epoch_correct += (preds == moves).sum().item()
            epoch_samples += bs

            # Update progress bar
            acc = epoch_correct / epoch_samples * 100
            lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                pol=f"{policy_loss.item():.3f}",
                val=f"{value_loss.item():.3f}",
                acc=f"{acc:.1f}%",
                lr=f"{lr:.1e}",
            )

            # Logging to file
            if self.global_step % Config.log_every_steps == 0:
                logger.debug(
                    f"  step {self.global_step:>7d} | "
                    f"loss {loss.item():.4f}  "
                    f"policy {policy_loss.item():.4f}  "
                    f"value {value_loss.item():.4f}  "
                    f"acc {acc:.1f}%  "
                    f"lr {lr:.2e}"
                )

            # Validation
            if self.global_step % Config.eval_every_steps == 0:
                self._validate()

            # Checkpoint
            if self.global_step % Config.save_every_steps == 0:
                self._save(tag=f"step_{self.global_step}")

        pbar.close()

        elapsed = time.time() - t0
        avg_pl = epoch_policy_loss / epoch_samples
        avg_vl = epoch_value_loss / epoch_samples
        avg_tl = epoch_total_loss / epoch_samples
        acc = epoch_correct / epoch_samples * 100
        logger.info(
            f"Epoch {epoch} done in {elapsed:.0f}s | "
            f"loss {avg_tl:.4f}  policy {avg_pl:.4f}  value {avg_vl:.4f}  "
            f"acc {acc:.1f}%"
        )
        return avg_tl

    # ── Validation ────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_pl = 0.0
        total_vl = 0.0
        total_correct = 0
        total_samples = 0

        for states, moves, values in tqdm(self.val_loader, desc="  Validation",
                                            unit="batch", dynamic_ncols=True,
                                            leave=False):
            states = states.to(self.device)
            moves = moves.to(self.device)
            values = values.to(self.device)

            p_logits, v_pred = self.model(states)
            pl = F.cross_entropy(p_logits, moves, reduction="sum")
            vl = F.mse_loss(v_pred.squeeze(-1), values, reduction="sum")

            bs = states.size(0)
            total_pl += pl.item()
            total_vl += vl.item()
            total_correct += (p_logits.argmax(1) == moves).sum().item()
            total_samples += bs

        avg_pl = total_pl / total_samples
        avg_vl = total_vl / total_samples
        acc = total_correct / total_samples * 100
        total_loss = avg_pl + avg_vl

        logger.info(
            f"  ── val step {self.global_step} | "
            f"loss {total_loss:.4f}  policy {avg_pl:.4f}  "
            f"value {avg_vl:.4f}  acc {acc:.1f}%"
        )

        if total_loss < self.best_val_loss:
            self.best_val_loss = total_loss
            self._save(tag="best")
            logger.info(f"     ★ New best val loss: {total_loss:.4f}")

        self.model.train()

    # ── Checkpointing ─────────────────────────────────────────────────

    def _save(self, tag: str):
        path = os.path.join(Config.checkpoint_dir, f"sl_{tag}.pt")
        save_checkpoint(self.model, path, optimizer=self.optimizer,
                        extra={"global_step": self.global_step,
                               "best_val_loss": self.best_val_loss})
        # Also save as latest
        latest = os.path.join(Config.checkpoint_dir, "latest.pt")
        save_checkpoint(self.model, latest, optimizer=self.optimizer,
                        extra={"global_step": self.global_step,
                               "best_val_loss": self.best_val_loss})
        logger.info(f"     Saved checkpoint: {path}")
