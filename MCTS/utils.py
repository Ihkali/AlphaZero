"""
MCTS/utils.py — Miscellaneous helpers: timing, logging, device selection.
"""

import os
import time
import torch
from pathlib import Path


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"  ⏱ {self.label}: {self.elapsed:.1f}s")


class Logger:
    def __init__(self, path: str):
        self.path = path
        self._header_written = False

    def log(self, data: dict):
        if not self._header_written:
            with open(self.path, "w") as f:
                f.write(",".join(data.keys()) + "\n")
            self._header_written = True
        with open(self.path, "a") as f:
            f.write(",".join(str(v) for v in data.values()) + "\n")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"
