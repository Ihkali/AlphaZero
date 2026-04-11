"""
tictactoe/mcts.py — Monte Carlo Tree Search for Tic-Tac-Toe.

Same algorithm as MCTS/mcts.py (UCB + neural-network expansion + backup)
but adapted for TicTacToe instead of chess.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from tictactoe.game import TicTacToe


class MCTSNode:
    __slots__ = [
        "game", "parent", "parent_action", "children",
        "visit_count", "value_sum", "prior", "is_expanded",
    ]

    def __init__(self, game: TicTacToe, parent=None, parent_action=None,
                 prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.parent_action = parent_action
        self.children: dict[int, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    def terminal_value(self) -> float:
        """Value from the perspective of the player who just moved INTO this node."""
        result = self.game.result()
        if result == 0:
            return 0.0
        # current_player already flipped after last push,
        # so if result == current_player, that means "I" won
        if result == self.game.current_player:
            return 1.0
        return -1.0


# ═══════════════════════════════════════════════════════════════════════════
#  MCTS SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def mcts_search(
    game: TicTacToe,
    neural_net: torch.nn.Module,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.5,
    dirichlet_epsilon: float = 0.25,
    temperature: float = 1.0,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Run MCTS from the given game state. Returns (policy[9], root_value)."""
    root = MCTSNode(game=game.copy())
    _expand(root, neural_net, device)

    # Dirichlet noise at root for exploration
    if dirichlet_epsilon > 0 and len(root.children) > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i]

    for _ in range(num_simulations):
        node = root
        path = [node]

        # ── SELECT ────────────────────────────────────────────────────
        while node.is_expanded and not node.is_terminal():
            _, node = _select_child(node, c_puct)
            path.append(node)

        # ── EXPAND & EVALUATE ─────────────────────────────────────────
        if node.is_terminal():
            value = node.terminal_value()
        else:
            value = _expand(node, neural_net, device)

        # ── BACKUP ────────────────────────────────────────────────────
        _backup(path, value)

    # ── Build policy from visit counts ────────────────────────────────
    policy = np.zeros(9, dtype=np.float32)
    if temperature == 0:
        best = max(root.children, key=lambda a: root.children[a].visit_count)
        policy[best] = 1.0
    else:
        counts = np.array(
            [(a, c.visit_count) for a, c in root.children.items()]
        )
        actions = counts[:, 0].astype(int)
        visits = counts[:, 1].astype(float)
        if temperature == 1.0:
            probs = visits / visits.sum()
        else:
            visits_t = visits ** (1.0 / temperature)
            probs = visits_t / visits_t.sum()
        for a, p in zip(actions, probs):
            policy[a] = p

    return policy, root.q_value


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _expand(node: MCTSNode, neural_net: torch.nn.Module, device: str) -> float:
    """Expand node and return the neural net value estimate."""
    game = node.game
    encoded = game.encode()
    tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)

    policy_probs, value = neural_net.predict(tensor.squeeze(0).to(device))

    legal = game.legal_moves
    if len(legal) == 0:
        node.is_expanded = True
        return 0.0

    prior_sum = sum(policy_probs[a] for a in legal)
    for action in legal:
        child_game = game.copy()
        child_game.push(action)
        prior = policy_probs[action] / prior_sum if prior_sum > 0 else 1.0 / len(legal)
        child = MCTSNode(game=child_game, parent=node,
                         parent_action=action, prior=prior)
        node.children[action] = child

    node.is_expanded = True
    return float(value)


def _select_child(node: MCTSNode, c_puct: float) -> tuple[int, MCTSNode]:
    best_score = -float("inf")
    best_action = -1
    best_child = None
    sqrt_parent = math.sqrt(node.visit_count + 1)

    for action, child in node.children.items():
        if child.visit_count == 0:
            q = 0.0
        else:
            q = -child.q_value          # opponent's value is negated
        u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child


def _backup(path: list[MCTSNode], value: float):
    """Propagate value back up the tree, negating at each level."""
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def select_action(policy: np.ndarray, temperature: float) -> int:
    if temperature == 0:
        return int(np.argmax(policy))
    nonzero = np.where(policy > 0)[0]
    probs = policy[nonzero]
    probs = probs / probs.sum()
    return int(np.random.choice(nonzero, p=probs))
