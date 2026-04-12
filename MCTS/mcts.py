"""
MCTS/mcts.py — AlphaGo-style Monte Carlo Tree Search.

Implements the MCTS algorithm from the original AlphaGo paper
(Silver et al., 2016, Section 4):

  - SL policy network pσ provides prior probabilities P(s,a)
  - Value network vθ evaluates leaf positions
  - Fast rollout policy pπ plays games to completion
  - Leaf evaluation mixes both: V(sL) = (1-λ)·vθ(sL) + λ·zL  (Eq. 6)
  - UCT selection: at = argmax_a [Q(s,a) + u(s,a)]              (Eq. 5)
  - u(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
  - Backup: Q(s,a) = (1/N(s,a)) · Σ 1(s,a,i)·V(sᵢ_L)         (Eq. 7-8)
  - Move selection: most visited action from root
"""

import math
import numpy as np
import chess
import torch
import torch.nn.functional as F

from MCTS.config import Config
from MCTS.encode import (
    encode_board, move_to_index, index_to_move,
    get_legal_move_indices, get_legal_mask,
)


# ═══════════════════════════════════════════════════════════════════════════
#  MCTS NODE
# ═══════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """A single node in the MCTS search tree.

    Each edge (s, a) stores:  Q(s,a), N(s,a), P(s,a).
    """
    __slots__ = [
        "board", "parent", "parent_action", "children",
        "visit_count", "value_sum", "prior", "is_expanded",
    ]

    def __init__(self, board: chess.Board, parent=None,
                 parent_action=None, prior: float = 0.0):
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
        """Mean action-value Q(s,a) = value_sum / N(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def terminal_value(self) -> float:
        """Terminal value from the perspective of the player to move."""
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        else:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN MCTS SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def mcts_search(
    root_board: chess.Board,
    policy_net: torch.nn.Module,            # pσ — SL policy network (priors)
    value_net: torch.nn.Module = None,      # vθ — value network (None → use policy_net value head)
    rollout_net: torch.nn.Module = None,    # pπ — fast rollout policy (None → λ forced to 0)
    num_simulations: int = Config.num_mcts_sims,
    c_puct: float = Config.c_puct,
    lambda_mix: float = Config.lambda_mix,
    dirichlet_alpha: float = Config.dirichlet_alpha,
    dirichlet_epsilon: float = Config.dirichlet_epsilon,
    temperature: float = 1.0,
    device: str = "cpu",
    leaf_batch_size: int = Config.mcts_leaf_batch,
    reuse_root: "MCTSNode | None" = None,
) -> tuple[np.ndarray, float, MCTSNode]:
    """AlphaGo-style MCTS with batched evaluation and virtual loss.

    The search uses the SL policy network pσ for prior probabilities
    (stored as P(s,a) on each edge), and evaluates leaf nodes by mixing
    the value network with rollout outcomes (paper Eq. 6):

        V(sL) = (1 − λ) · vθ(sL)  +  λ · zL

    where zL is the outcome of a rollout played to termination with pπ.
    If no rollout_net is provided, λ is forced to 0 (pure value eval).
    If no separate value_net is provided, the policy_net's value head is used.

    Returns (policy, root_value, root_node).
    """
    # If no rollout net provided, force pure value evaluation (λ=0)
    if rollout_net is None:
        lambda_mix = 0.0

    # If no separate value net, fall back to policy_net's value head
    v_net = value_net if value_net is not None else policy_net

    # ── Build or reuse root ───────────────────────────────────────────
    if reuse_root is not None and reuse_root.is_expanded:
        root = reuse_root
    else:
        root = MCTSNode(board=root_board.copy())
        _expand(root, policy_net, device)

    # Dirichlet noise at root for exploration
    if dirichlet_epsilon > 0 and len(root.children) > 0:
        _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    # ── Simulation loop (batched with virtual loss) ───────────────────
    sims_done = 0
    while sims_done < num_simulations:
        bs = min(leaf_batch_size, num_simulations - sims_done)

        # ── SELECT: trace paths to leaves ─────────────────────────────
        paths = []
        leaves = []
        for _ in range(bs):
            node = root
            path = []
            root.visit_count += 1       # virtual loss on root (per sim)
            while node.is_expanded and not node.is_terminal():
                _, node = _select_child(node, c_puct)
                path.append(node)
                node.visit_count += 1   # virtual loss on child
            paths.append(path)
            leaves.append(node)

        # ── EXPAND + EVALUATE unique unexpanded leaves ────────────────
        unique_map = {}                 # id(node) → (node, [path indices])
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

            # Evaluate policy (pσ) and value (vθ) — single forward
            # pass when using the same network (avoids duplicate
            # inference round-trips to the server)
            with torch.no_grad():
                if v_net is policy_net:
                    p_logits, v_vals = policy_net(tensor)
                else:
                    p_logits, _ = policy_net(tensor)
                    _, v_vals = v_net(tensor)
                p_probs = F.softmax(p_logits, dim=1).cpu().numpy()
                v_np = v_vals.cpu().numpy().ravel()

            for j, (nid, (node, indices)) in enumerate(unique_map.items()):
                # Expand node with SL policy priors  (paper Fig. 3b)
                _expand_with_policy(node, p_probs[j])

                # Leaf evaluation: V = (1-λ)·vθ + λ·zL  (paper Eq. 6)
                v_theta = float(v_np[j])
                if lambda_mix > 0:
                    z_rollout = _rollout(node.board, rollout_net, device)
                    leaf_val = (1.0 - lambda_mix) * v_theta + lambda_mix * z_rollout
                else:
                    leaf_val = v_theta

                for idx in indices:
                    leaf_values[idx] = leaf_val

        # ── BACKUP: propagate values up the tree  (paper Eq. 7-8) ────
        for i, (path, leaf) in enumerate(zip(paths, leaves)):
            if leaf.is_terminal():
                value = leaf.terminal_value()
            elif i in leaf_values:
                value = leaf_values[i]
            else:
                value = 0.0
            _backup_value_only(path, value)

        sims_done += bs

    # ── Extract policy from root visit counts ─────────────────────────
    policy = _get_policy(root, temperature)
    # Root value = average of children's Q from root's perspective
    if root.children:
        total_child_visits = sum(c.visit_count for c in root.children.values())
        if total_child_visits > 0:
            root_value = sum(-c.q_value * c.visit_count
                            for c in root.children.values()) / total_child_visits
        else:
            root_value = 0.0
    else:
        root_value = 0.0
    return policy, root_value, root


# ═══════════════════════════════════════════════════════════════════════════
#  ROLLOUT — play to terminal state with fast policy pπ  (paper Section 4)
# ═══════════════════════════════════════════════════════════════════════════

def _rollout(
    board: chess.Board,
    rollout_net: torch.nn.Module,
    device: str,
    max_depth: int = Config.rollout_max_depth,
) -> float:
    """Play a game to completion with the fast rollout policy pπ.

    Returns the outcome from the perspective of the player to move
    at *board* (the leaf being evaluated), +1 win / −1 loss / 0 draw.
    """
    perspective = board.turn
    temp_board = board.copy()
    depth = 0

    while not temp_board.is_game_over() and depth < max_depth:
        encoded = encode_board(temp_board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = rollout_net(tensor)

        # Mask illegal moves and sample
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        legal_mask = get_legal_mask(temp_board)
        masked = probs * legal_mask
        prob_sum = masked.sum()
        if prob_sum > 0:
            masked /= prob_sum
        else:
            # Fallback: uniform over legal moves
            masked = legal_mask / max(legal_mask.sum(), 1.0)

        action = np.random.choice(len(masked), p=masked)
        move = index_to_move(action, temp_board)
        if move not in temp_board.legal_moves:
            move = list(temp_board.legal_moves)[0]

        temp_board.push(move)
        depth += 1

    # Evaluate terminal outcome from the leaf player's perspective
    if temp_board.is_game_over():
        result = temp_board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if perspective == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if perspective == chess.WHITE else 1.0
        else:
            return 0.0

    return 0.0      # max depth reached → draw


# ═══════════════════════════════════════════════════════════════════════════
#  TREE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _expand(node: MCTSNode, policy_net: torch.nn.Module, device: str) -> float:
    """Expand a leaf node; prior P(s,a) = pσ(a|s) from SL policy (Fig. 3b)."""
    board = node.board
    encoded = encode_board(board)
    tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
    policy_probs, value = policy_net.predict(tensor.squeeze(0).to(device))

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


def _select_child(node: MCTSNode, c_puct: float,
                  fpu_reduction: float = Config.fpu_reduction) -> tuple[int, MCTSNode]:
    """Select child using PUCT formula with FPU and dynamic cPUCT.

    at = argmax_a [ Q(s,a) + u(s,a) ]
    u(s,a) = c(s) · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))

    Dynamic cPUCT:  c(s) = ln((1 + N(s) + base) / base) + init
    First Play Urgency: unvisited Q = parent_Q − fpu_reduction
    """
    best_score = -float("inf")
    best_action = -1
    best_child = None
    total_visits = node.visit_count
    sqrt_parent = math.sqrt(total_visits)

    # Dynamic cPUCT — exploration grows with visit count
    if Config.use_dynamic_cpuct:
        c = (math.log((1.0 + total_visits + Config.cpuct_base) / Config.cpuct_base)
             + Config.cpuct_init)
    else:
        c = c_puct

    # First Play Urgency: unvisited children assume parent's Q minus a reduction
    parent_q = node.q_value if node.visit_count > 0 else 0.0
    fpu_value = parent_q - fpu_reduction

    for action_idx, child in node.children.items():
        # Q from parent's perspective = negative of child's Q
        if child.visit_count > 0:
            q = -child.q_value
        else:
            q = fpu_value    # FPU for unvisited children
        u = c * child.prior * sqrt_parent / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action_idx
            best_child = child

    return best_action, best_child


def _backup_value_only(path: list[MCTSNode], value: float):
    """Backup value; visit counts already incremented via virtual loss."""
    for node in reversed(path):
        node.value_sum += value
        value = -value


def _expand_with_policy(node: MCTSNode, policy_probs: np.ndarray):
    """Expand node using pre-computed policy probs (batched-evaluate path)."""
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
    """Add Dirichlet noise to root priors for exploration."""
    noise = np.random.dirichlet([alpha] * len(node.children))
    for i, child in enumerate(node.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def _get_policy(root: MCTSNode, temperature: float) -> np.ndarray:
    """Extract move policy from root visit counts.

    Paper: "the algorithm chooses the most visited move from the root."
    Temperature adds exploration in early game moves.
    """
    policy = np.zeros(Config.policy_size, dtype=np.float32)
    if temperature == 0:
        best_action = max(root.children,
                          key=lambda a: root.children[a].visit_count)
        policy[best_action] = 1.0
    else:
        visit_counts = np.array(
            [(action, child.visit_count)
             for action, child in root.children.items()]
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


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_subtree_for_action(root: MCTSNode, action: int) -> "MCTSNode | None":
    """Detach the subtree for *action* from *root* for reuse.

    Pass the returned node as ``reuse_root`` to the next ``mcts_search()``
    call.  Returns None if the action was not in the tree.
    """
    child = root.children.get(action)
    if child is not None:
        child.parent = None
        child.parent_action = None
    return child


def select_action(policy: np.ndarray, temperature: float) -> int:
    """Sample an action from the policy distribution."""
    if temperature == 0:
        return int(np.argmax(policy))
    else:
        nonzero = np.where(policy > 0)[0]
        probs = policy[nonzero]
        probs = probs / probs.sum()
        choice = np.random.choice(nonzero, p=probs)
        return int(choice)
