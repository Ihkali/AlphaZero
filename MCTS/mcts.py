"""
MCTS/mcts.py — Monte Carlo Tree Search for AlphaZero.
"""

import math
import numpy as np
import chess
import torch
import torch.nn.functional as F

from MCTS.config import Config
from MCTS.encode import encode_board, move_to_index, index_to_move, get_legal_move_indices


class MCTSNode:
    __slots__ = [
        "board", "parent", "parent_action", "children",
        "visit_count", "value_sum", "prior", "is_expanded",
    ]

    def __init__(self, board: chess.Board, parent=None, parent_action=None, prior: float = 0.0):
        self.board = board
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
        return self.board.is_game_over()

    def terminal_value(self) -> float:
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        else:
            return 0.0


def mcts_search(
    root_board: chess.Board,
    neural_net: torch.nn.Module,
    num_simulations: int = Config.num_mcts_sims,
    c_puct: float = Config.c_puct,
    dirichlet_alpha: float = Config.dirichlet_alpha,
    dirichlet_epsilon: float = Config.dirichlet_epsilon,
    temperature: float = 1.0,
    device: str = "cpu",
    leaf_batch_size: int = Config.mcts_leaf_batch,
) -> tuple[np.ndarray, float]:
    """Batched MCTS with virtual-loss for efficient high-sim-count search."""
    root = MCTSNode(board=root_board.copy())
    _expand(root, neural_net, device)

    if dirichlet_epsilon > 0 and len(root.children) > 0:
        _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    sims_done = 0
    while sims_done < num_simulations:
        bs = min(leaf_batch_size, num_simulations - sims_done)
        root.visit_count += bs  # account for root visits (virtual loss)

        # ── Select paths to leaves with virtual loss ──────────────────
        paths = []
        leaves = []
        for _ in range(bs):
            node = root
            path = []
            while node.is_expanded and not node.is_terminal():
                _, node = _select_child(node, c_puct)
                path.append(node)
                node.visit_count += 1          # virtual loss
            paths.append(path)
            leaves.append(node)

        # ── Batch-evaluate unique unexpanded leaves ───────────────────
        unique_map = {}                         # id(node) → (node, [path indices])
        for i, node in enumerate(leaves):
            if not node.is_terminal() and not node.is_expanded:
                nid = id(node)
                if nid not in unique_map:
                    unique_map[nid] = (node, [])
                unique_map[nid][1].append(i)

        leaf_values = {}
        if unique_map:
            nodes_list = [entry[0] for entry in unique_map.values()]
            states = np.stack([encode_board(n.board) for n in nodes_list])
            tensor = torch.from_numpy(states).to(device)
            with torch.no_grad():
                p_logits, vals = neural_net(tensor)
                p_probs = F.softmax(p_logits, dim=1).cpu().numpy()
                vals_np = vals.cpu().numpy().ravel()

            for j, (nid, (node, indices)) in enumerate(unique_map.items()):
                _expand_with_policy(node, p_probs[j])
                for idx in indices:
                    leaf_values[idx] = float(vals_np[j])

        # ── Backup (visit counts already applied via virtual loss) ────
        for i, (path, leaf) in enumerate(zip(paths, leaves)):
            if leaf.is_terminal():
                value = leaf.terminal_value()
            elif i in leaf_values:
                value = leaf_values[i]
            else:
                value = 0.0
            _backup_value_only(path, value)

        sims_done += bs

    policy = _get_policy(root, temperature)
    root_value = root.q_value
    return policy, root_value


def _expand(node: MCTSNode, neural_net: torch.nn.Module, device: str) -> float:
    board = node.board
    encoded = encode_board(board)
    tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
    policy_probs, value = neural_net.predict(tensor.squeeze(0).to(device))

    legal_moves = get_legal_move_indices(board)
    if len(legal_moves) == 0:
        node.is_expanded = True
        return 0.0

    prior_sum = 0.0
    move_priors = []
    for move, idx in legal_moves:
        prior_sum += policy_probs[idx]
        move_priors.append((move, idx, policy_probs[idx]))

    for move, idx, prior in move_priors:
        child_board = board.copy()
        child_board.push(move)
        prior_normalised = prior / prior_sum if prior_sum > 0 else 1.0 / len(legal_moves)
        child = MCTSNode(
            board=child_board, parent=node, parent_action=idx,
            prior=prior_normalised,
        )
        node.children[idx] = child

    node.is_expanded = True
    return value


def _select_child(node: MCTSNode, c_puct: float) -> tuple[int, MCTSNode]:
    best_score = -float("inf")
    best_action = -1
    best_child = None
    sqrt_parent = math.sqrt(node.visit_count + 1)

    # First-Play Urgency: unvisited children get parent's value minus a reduction
    fpu_value = -node.q_value - Config.fpu_reduction

    for action_idx, child in node.children.items():
        if child.visit_count == 0:
            q = fpu_value
        else:
            q = -child.q_value
        u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action_idx
            best_child = child

    return best_action, best_child


def _backup(path: list[MCTSNode], value: float):
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _backup_value_only(path: list[MCTSNode], value: float):
    """Backup value only — visit counts already incremented via virtual loss."""
    for node in reversed(path):
        node.value_sum += value
        value = -value


def _expand_with_policy(node: MCTSNode, policy_probs: np.ndarray):
    """Expand a node using pre-computed policy probabilities (batch path)."""
    board = node.board
    legal_moves = get_legal_move_indices(board)
    if len(legal_moves) == 0:
        node.is_expanded = True
        return

    prior_sum = 0.0
    move_priors = []
    for move, idx in legal_moves:
        p = float(policy_probs[idx])
        prior_sum += p
        move_priors.append((move, idx, p))

    for move, idx, prior in move_priors:
        child_board = board.copy()
        child_board.push(move)
        prior_norm = prior / prior_sum if prior_sum > 0 else 1.0 / len(legal_moves)
        child = MCTSNode(
            board=child_board, parent=node, parent_action=idx,
            prior=prior_norm,
        )
        node.children[idx] = child

    node.is_expanded = True


def _add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
    noise = np.random.dirichlet([alpha] * len(node.children))
    for i, child in enumerate(node.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def _get_policy(root: MCTSNode, temperature: float) -> np.ndarray:
    policy = np.zeros(Config.policy_size, dtype=np.float32)
    if temperature == 0:
        best_action = max(root.children, key=lambda a: root.children[a].visit_count)
        policy[best_action] = 1.0
    else:
        visit_counts = np.array(
            [(action, child.visit_count) for action, child in root.children.items()]
        )
        actions = visit_counts[:, 0].astype(int)
        counts = visit_counts[:, 1].astype(float)
        if temperature == 1.0:
            probs = counts / counts.sum()
        else:
            counts_temp = counts ** (1.0 / temperature)
            probs = counts_temp / counts_temp.sum()
        for action, prob in zip(actions, probs):
            policy[action] = prob
    return policy


def select_action(policy: np.ndarray, temperature: float) -> int:
    if temperature == 0:
        return int(np.argmax(policy))
    else:
        nonzero = np.where(policy > 0)[0]
        probs = policy[nonzero]
        probs = probs / probs.sum()
        choice = np.random.choice(nonzero, p=probs)
        return int(choice)
