"""
MCTS/model.py — Neural network architectures for AlphaGo.

AlphaZeroNet  f_θ(s) → (p, v)     — shared architecture for pσ, pρ, vθ
RolloutPolicy f_π(s) → p           — fast lightweight CNN for rollout policy pπ

Architecture (AlphaZeroNet):
  Input  (B, 119, 8, 8)   [T=8 history stack, paper-style encoding]
   → ConvBlock  → N × ResBlock (SE-Net)
   → PolicyHead  (B, 4672)
   → ValueHead   (B, 1)

Architecture (RolloutPolicy):
  Input  (B, 119, 8, 8)
   → Small ConvBlock  → K × Conv layers  (K=2, 32 filters)
   → PolicyFC  (B, 4672)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from MCTS.config import Config


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (KataGo-style)."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, x):
        w = self.pool(x).flatten(1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int = Config.se_reduction):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, policy_size: int = Config.policy_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, policy_size)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ValueHead(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_size: int = Config.value_head_hidden):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        in_channels: int = Config.input_planes,
        num_filters: int = Config.num_filters,
        num_blocks: int = Config.num_res_blocks,
        policy_size: int = Config.policy_size,
    ):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, num_filters)
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_blocks)]
        )
        self.policy_head = PolicyHead(num_filters, policy_size)
        self.value_head = ValueHead(num_filters)

    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    def predict(self, board_tensor: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            p_logits, v = self(board_tensor)
            p = F.softmax(p_logits, dim=1)
            return p.cpu().numpy()[0], v.cpu().numpy()[0, 0]


class RolloutPolicy(nn.Module):
    """Fast rollout policy pπ — lightweight CNN for quick move sampling.

    In the AlphaGo paper, pπ is a "linear softmax of small pattern features"
    that runs in ~2µs per move (accuracy 24.2%).  For chess, we use a small
    CNN (2 conv layers, 32 filters) which is much faster than the full
    ResNet and captures basic tactical patterns.
    """
    def __init__(
        self,
        in_channels: int = Config.input_planes,
        filters: int = Config.rollout_filters,
        num_blocks: int = Config.rollout_blocks,
        policy_size: int = Config.policy_size,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        ]
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(filters, filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(filters),
                nn.ReLU(),
            ])
        self.backbone = nn.Sequential(*layers)
        self.policy_fc = nn.Linear(filters * 8 * 8, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw policy logits (B, policy_size)."""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.policy_fc(x)


def save_checkpoint(model: AlphaZeroNet, path: str, optimizer=None, extra=None):
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if extra is not None:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint(model: AlphaZeroNet, path: str, optimizer=None, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    return data
