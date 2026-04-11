"""
tictactoe/model.py — Tiny neural network for Tic-Tac-Toe  f(s) → (π, v).

Input:  (B, 3, 3, 3)   [current pieces, opponent pieces, turn indicator]
Output: policy (B, 9),  value (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # Small conv tower
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Policy head
        self.p_conv = nn.Conv2d(64, 4, 1)
        self.p_bn = nn.BatchNorm2d(4)
        self.p_fc = nn.Linear(4 * 3 * 3, 9)

        # Value head
        self.v_conv = nn.Conv2d(64, 2, 1)
        self.v_bn = nn.BatchNorm2d(2)
        self.v_fc1 = nn.Linear(2 * 3 * 3, hidden)
        self.v_fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Policy
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)                       # raw logits (B, 9)

        # Value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))          # in [-1, 1]

        return p, v

    def predict(self, board_tensor: torch.Tensor):
        """Single-sample inference returning numpy arrays."""
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            p_logits, v = self(board_tensor)
            p = F.softmax(p_logits, dim=1)
            return p.cpu().numpy()[0], v.cpu().numpy()[0, 0]
