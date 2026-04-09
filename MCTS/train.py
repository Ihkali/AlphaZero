"""
MCTS/train.py — Training loop for AlphaZero.

Loss: L = MSE(v, z) - π⊤ log(softmax(p)) + c·‖θ‖²

Supports both in-memory ReplayBuffer and disk-based DiskReplayBuffer.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from MCTS.config import Config
from MCTS.model import AlphaZeroNet, save_checkpoint


class WarmupCosineScheduler:
    """Linear warmup → cosine annealing."""
    def __init__(self, optimizer, warmup_steps, total_steps, lr_min=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = max(0, 0.5 * (1 + np.cos(np.pi * progress)))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.lr_min, base_lr * scale)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(self, net, device=Config.device, lr=Config.learning_rate,
                 weight_decay=Config.weight_decay):
        self.net = net
        self.device = device
        self.optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=Config.warmup_steps,
            total_steps=Config.train_steps,
            lr_min=Config.lr_min,
        )

    def train(self, replay_buffer, num_steps=Config.train_steps,
              batch_size=Config.batch_size, verbose=True):
        """
        Train from any buffer that implements .sample(batch_size) and __len__.
        Works with both ReplayBuffer and DiskReplayBuffer.
        """
        self.net.train()
        history = {"total": [], "policy": [], "value": []}

        buf_len = len(replay_buffer)
        if buf_len == 0:
            print("  ⚠ Empty replay buffer, skipping training.")
            return history
        batch_size = min(batch_size, buf_len)

        t0 = time.time()
        log_interval = max(1, num_steps // 10)

        for step in range(num_steps):
            states, target_pis, target_vs = replay_buffer.sample(batch_size)
            states_t = torch.from_numpy(states).to(self.device)
            target_pis_t = torch.from_numpy(target_pis).to(self.device)
            target_vs_t = torch.from_numpy(target_vs).unsqueeze(1).to(self.device)

            policy_logits, values = self.net(states_t)
            value_loss = F.mse_loss(values, target_vs_t)
            policy_loss = self._cross_entropy_loss(policy_logits, target_pis_t)
            total_loss = value_loss + policy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(),
                                     max_norm=Config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            history["total"].append(total_loss.item())
            history["policy"].append(policy_loss.item())
            history["value"].append(value_loss.item())

            if verbose and (step + 1) % log_interval == 0:
                elapsed = time.time() - t0
                lr = self.scheduler.get_lr()
                avg_t = np.mean(history["total"][-log_interval:])
                avg_p = np.mean(history["policy"][-log_interval:])
                avg_v = np.mean(history["value"][-log_interval:])
                print(
                    f"  Step {step+1:5d}/{num_steps} | "
                    f"loss={avg_t:.4f} "
                    f"(π={avg_p:.4f}, v={avg_v:.4f}) | "
                    f"lr={lr:.2e} | {elapsed:.1f}s"
                )

        self.net.eval()
        if verbose:
            avg_total = np.mean(history["total"][-100:])
            print(f"\n  Training done. Final avg loss: {avg_total:.4f}  "
                  f"({buf_len} examples in buffer)")
        return history

    @staticmethod
    def _cross_entropy_loss(logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(targets * log_probs, dim=1).mean()

    def save(self, path, extra=None):
        save_checkpoint(self.net, path, self.optimizer, extra)
