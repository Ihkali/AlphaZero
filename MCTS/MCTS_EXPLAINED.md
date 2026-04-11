# AlphaZero MCTS — Complete Technical Documentation

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Board Encoding (`encode.py`)](#2-board-encoding)
3. [Move Encoding / Decoding (`encode.py`)](#3-move-encoding--decoding)
4. [Neural Network Architecture (`model.py`)](#4-neural-network-architecture)
5. [MCTS Algorithm — Step by Step (`mcts.py`)](#5-mcts-algorithm--step-by-step)
6. [Self-Play Pipeline (`self_play.py`)](#6-self-play-pipeline)
7. [Training Loop (`train.py`)](#7-training-loop)
8. [Evaluation / Gating (`evaluate.py`)](#8-evaluation--gating)
9. [Main Orchestrator (`main.py`)](#9-main-orchestrator)
10. [Configuration Reference (`config.py`)](#10-configuration-reference)
11. [Worked Example — One Complete MCTS Simulation](#11-worked-example)
12. [Worked Example — One Complete Iteration](#12-worked-example--one-complete-iteration)

---

## 1. High-Level Overview

The system implements the **AlphaZero** algorithm (Silver et al., 2017) with an **SL pre-training** warm start. The full pipeline is:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SL Pre-training (done in SL/)                               │
│     Train f_θ(s) → (p, v) on human games from a CSV dataset     │
│     This gives the network a "starting brain" (~1200-1500 Elo)  │
├─────────────────────────────────────────────────────────────────┤
│  2. MCTS Self-Play Loop (this code — MCTS/)                     │
│     for iteration in range(num_iterations):                      │
│       a) Self-play: use MCTS + f_θ to play games against itself │
│          → collect training data (s, π, z)                       │
│       b) Train: update f_θ on collected data                     │
│       c) (Optional) Evaluate: pit new f_θ vs best f_θ           │
│       d) Checkpoint                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The neural network f_θ(s) has two outputs:
- **Policy head p**: probability distribution over all 4672 possible moves
- **Value head v**: scalar in [-1, +1] predicting the game outcome

MCTS uses `p` as prior probabilities to guide tree search, and `v` to evaluate positions without playing to the end.

---

## 2. Board Encoding

**File**: `encode.py` → `encode_board(board) → np.ndarray (119, 8, 8)`

The board is encoded as a 3D tensor with **119 planes**, each 8×8. This is the input to the neural network.

### Plane Layout

| Planes | Count | Description |
|--------|-------|-------------|
| 0–111 | 112 | History stack: 8 time-steps × 14 planes each |
| 112 | 1 | Colour to move (1.0 = White, 0.0 = Black) |
| 113 | 1 | Move number (fullmove_number / 500) |
| 114 | 1 | Our kingside castling rights |
| 115 | 1 | Our queenside castling rights |
| 116 | 1 | Opponent kingside castling rights |
| 117 | 1 | Opponent queenside castling rights |
| 118 | 1 | Halfmove clock (halfmove_clock / 100) |

### Each History Time-Step (14 planes per step)

For each of the T=8 history positions (current + 7 previous):

| Offset | Content |
|--------|---------|
| +0 | Our pawns |
| +1 | Our knights |
| +2 | Our bishops |
| +3 | Our rooks |
| +4 | Our queens |
| +5 | Our king |
| +6 | Opponent pawns |
| +7 | Opponent knights |
| +8 | Opponent bishops |
| +9 | Opponent rooks |
| +10 | Opponent queens |
| +11 | Opponent king |
| +12 | 2-fold repetition flag (all 1s if position repeated 2×) |
| +13 | 3-fold repetition flag (all 1s if position repeated 3×) |

### Perspective Flipping

All planes are encoded **from the current player's perspective**:
- When it's White's turn: rank 0 = rank 1 (a1), rank 7 = rank 8 (a8) — normal orientation
- When it's Black's turn: the board is flipped both vertically and horizontally (so Black always "looks" up the board like White)

```python
def _square_to_rc(sq, perspective):
    rank = chess.square_rank(sq)    # 0-7
    file = chess.square_file(sq)    # 0-7
    if perspective == chess.WHITE:
        return rank, file           # normal
    else:
        return 7 - rank, 7 - file   # flipped
```

### Example

Starting position, White to move:
```
Plane 0 (our pawns):          Plane 6 (opponent pawns):
0 0 0 0 0 0 0 0               0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0               1 1 1 1 1 1 1 1    ← rank 7 (Black's pawns)
0 0 0 0 0 0 0 0               0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0               0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0               0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0               0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0               0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1  ← rank 1     0 0 0 0 0 0 0 0

Plane 112 (colour): all 1.0 (White to move)
Plane 113 (move number): all 1/500 = 0.002
```

---

## 3. Move Encoding / Decoding

**File**: `encode.py`

Every possible chess move is mapped to an integer index in `[0, 4671]`.

The encoding uses **73 move-type planes** per source square:
```
index = source_square_index × 73 + move_type
```

where `source_square_index = row × 8 + col` (in the current player's perspective).

### The 73 Move Types

| Range | Count | Description |
|-------|-------|-------------|
| 0–55 | 56 | Queen-type moves: 8 directions × 7 distances |
| 56–63 | 8 | Knight moves: 8 possible L-shapes |
| 64–72 | 9 | Underpromotions: 3 pieces × 3 directions |

#### Queen-type moves (planes 0–55)

8 directions: N, NE, E, SE, S, SW, W, NW
For each direction, 7 possible distances (1 to 7 squares).

```
move_type = direction_index × 7 + (distance - 1)
```

This covers all rook, bishop, queen, king (1-square), and pawn moves.
**Queen promotions** are encoded here too (they're just pawn moves to the 8th rank with distance=1).

#### Knight moves (planes 56–63)

```python
KNIGHT_MOVES = [
    (2,1), (2,-1), (-2,1), (-2,-1),
    (1,2), (1,-2), (-1,2), (-1,-2),
]
```

#### Underpromotions (planes 64–72)

For pawn promotions to knight/bishop/rook (NOT queen — queens use the normal queen-direction slot):

```
move_type = 64 + piece_index × 3 + direction_index
```
- `piece_index`: 0=knight, 1=bishop, 2=rook
- `direction_index`: 0=capture-left, 1=straight, 2=capture-right

### Example

White plays e2→e4 (pawn double push):
```
from_sq = e2 → (row=1, col=4) in White's perspective
to_sq   = e4 → (row=3, col=4)
dr = 3-1 = 2, dc = 4-4 = 0

Direction: (1,0) = North → index 0
Distance: 2 → dist-1 = 1

source_index = 1 × 8 + 4 = 12
move_type = 0 × 7 + 1 = 1
final_index = 12 × 73 + 1 = 877
```

White plays Ng1→f3 (knight move):
```
from_sq = g1 → (row=0, col=6)
to_sq   = f3 → (row=2, col=5)
dr = 2, dc = -1

(2,-1) is KNIGHT_MOVES[1]
move_type = 56 + 1 = 57

source_index = 0 × 8 + 6 = 6
final_index = 6 × 73 + 57 = 495
```

---

## 4. Neural Network Architecture

**File**: `model.py`

### AlphaZeroNet: f_θ(s) → (p, v)

```
Input: (batch, 119, 8, 8)
  │
  ▼
ConvBlock(119 → 256, 3×3)     ← initial convolution + BN + ReLU
  │
  ▼
ResBlock × 15                    ← residual blocks with SE attention
  │ Each: Conv3×3 → BN → ReLU → Conv3×3 → BN → SE → + residual → ReLU
  │
  ├──────────────────┐
  ▼                  ▼
PolicyHead          ValueHead
  │                  │
Conv1×1(256→32)     Conv1×1(256→4)
  BN + ReLU          BN + ReLU
  Flatten             Flatten
  FC(2048→4672)      FC(256→512)  → ReLU
  │                  FC(512→1)    → tanh
  ▼                  ▼
 logits (4672)      value [-1, +1]
```

**Parameters**: ~40M (256 filters, 15 blocks)

### SE-Block (Squeeze-and-Excitation)

Each residual block has a channel attention mechanism:

```
Input x: (B, 256, 8, 8)
  │
  ▼ GlobalAvgPool
  (B, 256)
  │
  ▼ FC(256 → 32) + ReLU        ← squeeze (reduction = 8)
  ▼ FC(32 → 256) + Sigmoid     ← excitation
  (B, 256, 1, 1)
  │
  ▼ Multiply with x              ← scale each channel
```

This lets the network learn "which feature channels are important for this position."

### RolloutPolicy (Legacy)

A small CNN (2 blocks, 32 filters) for fast rollout evaluation. **Not used** in AlphaZero mode (λ=0), kept for backward compatibility with AlphaGo-style search.

---

## 5. MCTS Algorithm — Step by Step

**File**: `mcts.py`

This is the core of AlphaZero. MCTS builds a search tree to decide which move to play. Each call to `mcts_search()` runs 800 simulations (by default) and returns a visit-count-based policy.

### 5.1 Data Structures

Each **MCTSNode** stores:

| Field | Type | Description |
|-------|------|-------------|
| `board` | `chess.Board` | The board state at this node |
| `parent` | `MCTSNode` | Parent node (None for root) |
| `parent_action` | `int` | Action index that led here |
| `children` | `dict[int, MCTSNode]` | action_index → child node |
| `visit_count` | `int` | N(s,a) — times this node was visited |
| `value_sum` | `float` | W(s,a) — accumulated value |
| `prior` | `float` | P(s,a) — prior probability from neural net |
| `is_expanded` | `bool` | Whether children have been created |

**Q-value**: `Q(s,a) = value_sum / visit_count`

### 5.2 The Four Phases of Each Simulation

Each of the 800 simulations does:

```
╔══════════╗     ╔══════════╗     ╔══════════╗     ╔══════════╗
║ 1.SELECT ║ ──> ║ 2.EXPAND ║ ──> ║3.EVALUATE║ ──> ║ 4.BACKUP ║
╚══════════╝     ╚══════════╝     ╚══════════╝     ╚══════════╝
```

---

#### Phase 1: SELECT — Walk Down the Tree

Starting from the root, repeatedly pick the child with the highest **PUCT score** until we reach an **unexpanded** or **terminal** node.

**PUCT Formula** (Paper Eq. 5):

```
a_t = argmax_a [ Q(s,a) + U(s,a) ]
```

where:

```
U(s,a) = c(s) × P(s,a) × √(N_parent) / (1 + N(s,a))
```

- `Q(s,a)`: mean value of this action (exploitation)
- `U(s,a)`: exploration bonus (high prior + low visits = explore)
- `c(s)`: exploration constant (dynamic cPUCT)

**Dynamic cPUCT** (grows logarithmically with visit count):

```
c(s) = ln((1 + N(s) + 19652) / 19652) + 2.5
```

At low visits c ≈ 2.5, at high visits c grows slightly → more exploration with more budget.

**First Play Urgency (FPU)**: Unvisited children don't have a Q value yet. Instead of assuming Q=0, we use:

```
Q_unvisited = Q_parent − 0.25
```

This means unvisited children of a strong node get a reasonable pessimistic estimate instead of zero, preventing the search from wasting simulations on clearly bad unexplored moves in winning positions.

**Sign convention**: The value stored in a child node is from that child's (opponent's) perspective. So from the parent's perspective:

```python
q_from_parent = -child.q_value      # negate to flip perspective
```

**Example** — selecting at root after some simulations:

```
Root (White to move, 100 total visits)
├── e4  (N=40, Q_child=-0.15) → Q_parent=+0.15, P=0.30
│   Score = 0.15 + 2.5 × 0.30 × √100 / 41 = 0.15 + 0.183 = 0.333
│
├── d4  (N=35, Q_child=-0.10) → Q_parent=+0.10, P=0.25
│   Score = 0.10 + 2.5 × 0.25 × √100 / 36 = 0.10 + 0.174 = 0.274
│
├── Nf3 (N=20, Q_child=-0.05) → Q_parent=+0.05, P=0.15
│   Score = 0.05 + 2.5 × 0.15 × √100 / 21 = 0.05 + 0.179 = 0.229
│
├── c4  (N=5, Q_child=+0.02)  → Q_parent=-0.02, P=0.08
│   Score = -0.02 + 2.5 × 0.08 × √100 / 6 = -0.02 + 0.333 = 0.313
│
└── a3  (N=0, unvisited)       → Q=FPU=-0.15-0.25=-0.40, P=0.01
    Score = -0.40 + 2.5 × 0.01 × √100 / 1 = -0.40 + 0.250 = -0.150

Selected: e4 (score 0.333)
```

Notice how c4 (low visits, reasonable prior) scores almost as high as e4 (many visits) — PUCT naturally balances exploitation and exploration.

---

#### Phase 2: EXPAND

When we reach an unexpanded leaf, we create child nodes for every legal move and assign **prior probabilities** from the neural network's policy head.

```python
# Neural net produces raw logits
p_logits, _ = policy_net(encoded_state)      # (1, 4672)
p_probs = softmax(p_logits)                  # (1, 4672)

# Only legal moves get children; priors are renormalised
for move in legal_moves:
    idx = move_to_index(move, board)
    prior = p_probs[idx]                     # neural net's probability for this move
    child = MCTSNode(board_after_move, prior=prior / sum_of_legal_priors)
    node.children[idx] = child
```

The priors guide the initial exploration: the neural net says "e4 looks 30% likely to be best" → MCTS explores e4 more in early simulations.

---

#### Phase 3: EVALUATE

The newly expanded leaf gets a value from the **value head**:

```python
_, v = value_net(encoded_state)    # v ∈ [-1, +1]
leaf_value = v                     # from the leaf player's perspective
```

In pure AlphaZero mode (λ=0), this is just the value head output.

In AlphaGo mode (λ>0, legacy), it would be mixed with a rollout:
```
V = (1 − λ) × v_θ + λ × z_rollout
```

---

#### Phase 4: BACKUP

Propagate the evaluation value **back up the tree** to every node on the path from the leaf to the root:

```python
def _backup_value_only(path, value):
    for node in reversed(path):     # leaf → root
        node.value_sum += value
        value = -value              # flip sign at each layer!
```

**Why negate?** If the value is +0.3 from White's perspective at a leaf:
- White's parent node gets +0.3 (good for White)
- Black's grandparent node gets -0.3 (bad for Black = good for White)
- White's great-grandparent gets +0.3 again

This alternating sign maintains correct perspective at every level.

**Example**:

```
Root (White, N=100)
  └── e4 (Black, N=40)
        └── e5 (White, N=15)
              └── Nf3 (Black, N=8)
                    └── [NEW LEAF] ← value network says +0.2 (good for Black)

Backup:
  Nf3:  value_sum += +0.2,  then value = -0.2
  e5:   value_sum += -0.2,  then value = +0.2
  e4:   value_sum += +0.2,  then value = -0.2
  (root visit count was pre-incremented by virtual loss)
```

### 5.3 Batched Evaluation with Virtual Loss

Instead of one simulation at a time, the code runs **8 simulations in parallel** (configurable via `mcts_leaf_batch`):

1. **Select** 8 paths simultaneously from root to leaves
2. **Virtual loss**: pre-increment `visit_count` along each path before evaluation. This discourages other paths in the same batch from going to the same node (diversity).
3. **Batch-evaluate** all unique leaf positions in a single forward pass through the neural net
4. **Backup** all 8 paths

This is ~8× more efficient on GPU/MPS since one batched forward pass costs nearly the same as one single pass.

### 5.4 Dirichlet Noise at Root

Before starting simulations, noise is added to root priors:

```
P'(s,a) = (1 − ε) × P(s,a) + ε × η_a
```

where `η ~ Dir(0.3)` and `ε = 0.25`.

This ensures the search explores moves the neural net didn't consider. Without it, MCTS would be trapped by the neural net's biases and could never discover novel moves during self-play.

**Example**: If the net gives 0% to a move, after Dirichlet noise it might have 2-5% prior — enough to get explored if it turns out to be good.

### 5.5 Move Selection from Root

After all 800 simulations, select a move based on **visit counts** (not Q values):

**Temperature = 1.0** (first 30 moves — exploration):
```
π(a) = N(s,a) / Σ_b N(s,b)
```
Then sample from this distribution. More-visited moves are more likely, but there's randomness.

**Temperature = 0** (move 31+ — exploitation):
```
π(a) = 1 if a = argmax N(s,a), else 0
```
Always pick the most-visited move.

**Why visit counts, not Q values?** Q values can be noisy. Visit counts reflect the cumulative search effort — if MCTS spent 200 of 800 simulations on e4, it's confident about e4 regardless of exact Q oscillations.

### 5.6 Subtree Reuse

After making a move, the subtree rooted at the chosen action is **reused** for the next search:

```python
reuse_root = get_subtree_for_action(root_node, action)
# Next call:
mcts_search(..., reuse_root=reuse_root)
```

This preserves all the search work from previous moves. If e4 had 400 visits, the subtree under e4 starts with those visits already present, saving significant computation.

---

## 6. Self-Play Pipeline

**File**: `self_play.py`

### 6.1 Game Generation

One self-play game works like this:

```
board = starting_position

for move_count in range(max_moves):
    1. MCTS search (800 sims) → policy π, value V
    2. Save (encoded_board, π, current_player) to game_history
    3. Sample action from π (with temperature)
    4. Play the move
    5. Check resignation (if V < -0.9 for 10 consecutive moves)

# After game ends:
outcome z = +1 (White wins) / -1 (Black wins) / 0 (draw)

# Assign values to each position:
for (state, policy, player) in game_history:
    value = z from player's perspective
    training_example = (state, policy, value)
```

### 6.2 Training Data Format

Each example is a tuple `(s, π, z)`:
- `s`: encoded board (119, 8, 8)
- `π`: MCTS visit-count policy (4672,)
- `z`: game outcome from that player's perspective ∈ {-1, 0, +1}

### 6.3 Resignation

To speed up self-play, the engine can resign early:
- After **iteration 10**, if root value < -0.9 for **10 consecutive moves**, resign
- 10% of games **ignore** resignation (play out fully) as a calibration check
- The resigning side loses; the training labels reflect this

### 6.4 Parallel Execution

```
Main process
  ├── Worker 0: plays 125 games (500 / 4 workers)
  ├── Worker 1: plays 125 games
  ├── Worker 2: plays 125 games
  └── Worker 3: plays 125 games
          │
          ▼
  StreamingDataWriter: appends each game to .npy files immediately
  (zero RAM — all data on disk)
```

Workers run on **CPU** (each uses the full AlphaZeroNet). Results stream to disk via `StreamingDataWriter` which writes raw numpy data to `.npy` files and patches the headers after all games complete.

### 6.5 Disk Data Storage

Each iteration produces 3 files:
```
MCTS/data/iter_0000_states.npy     # (N, 119, 8, 8)
MCTS/data/iter_0000_policies.npy   # (N, 4672)
MCTS/data/iter_0000_values.npy     # (N,)
```

---

## 7. Training Loop

**File**: `train.py`

### 7.1 Loss Function

Paper Eq. 1:

```
L = (z − v)² − π⊤ log p + c‖θ‖²
```

| Term | Name | Purpose |
|------|------|---------|
| `(z − v)²` | Value loss (MSE) | Train value head to predict game outcomes |
| `−π⊤ log p` | Policy loss (cross-entropy) | Train policy head to match MCTS visit counts |
| `c‖θ‖²` | L2 regularisation | Prevent overfitting (c = 1e-4) |

**Note**: Cross-entropy with soft targets π is equivalent to KL divergence up to a constant (since π is fixed during training). The gradients are identical.

```python
# Policy loss implementation:
log_probs = F.log_softmax(logits, dim=1)
policy_loss = -torch.sum(targets * log_probs, dim=1).mean()

# Value loss:
value_loss = F.mse_loss(predicted_v, target_z)

# Total:
total_loss = value_loss + policy_loss
# (L2 regularisation is handled by optimizer weight_decay)
```

### 7.2 Optimizer

**SGD** with momentum 0.9, Nesterov acceleration:
- Initial LR: 0.01
- Warmup: 200 steps (linear from 0 → 0.01)
- Step decay: LR × 0.1 at iterations 30, 60, 80

### 7.3 Replay Buffer (Disk-Based)

Training samples from a **sliding window** of the last 20 iterations' data:

```python
class DiskReplayBuffer:
    # Memory-maps .npy files from disk
    # Samples random indices across all files
    # Only the requested batch is loaded into RAM
```

This is zero-copy: the OS memory-maps the files, and only the sampled rows are read into RAM.

### 7.4 Training Step

```
for step in range(1000):
    batch = replay_buffer.sample(256)         # random 256 positions
    states, target_π, target_z = batch

    logits, v = neural_net(states)            # forward pass
    loss = MSE(v, z) + CrossEntropy(logits, π)

    loss.backward()                            # compute gradients
    clip_grad_norm_(1.0)                       # prevent exploding gradients
    optimizer.step()                           # update weights
    scheduler.step()                           # adjust learning rate
```

---

## 8. Evaluation / Gating

**File**: `evaluate.py`

When `use_eval_gating = True`, the new model plays against the current best:

```
for game in range(40):
    new_model plays White in even games, Black in odd games
    Both sides use MCTS with 400 simulations
    Temperature = 0, no Dirichlet noise (pure evaluation)

win_rate = (new_wins + 0.5 × draws) / total_games
if win_rate > 0.55:
    accept new model as "best"
else:
    revert to previous best
```

**Current setting**: `use_eval_gating = False` (paper's approach — always use the latest model, no gating).

---

## 9. Main Orchestrator

**File**: `main.py`

```python
# 1. Load SL pre-trained model (or resume from checkpoint)
net = load_sl_model("SL/checkpoints/final_model.pt")

# 2. Create trainer (persists optimizer state across iterations)
trainer = Trainer(net)

# 3. Main loop
for iteration in range(num_iterations):
    # Self-play → disk
    run_self_play(net, num_games=500, num_sims=800)

    # Train from disk replay buffer
    replay.refresh()
    trainer.train(replay, num_steps=1000, batch_size=256)

    # (Optional) Evaluate
    if use_eval_gating:
        evaluate_models(net, best_net)

    # Checkpoint
    save_checkpoint(net, "MCTS/checkpoints/latest.pt")
```

---

## 10. Configuration Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Encoding** | | |
| `history_length` | 8 | Past positions to encode |
| `input_planes` | 119 | 8×14 + 7 feature planes |
| `policy_size` | 4672 | 8×8×73 possible move encodings |
| **Network** | | |
| `num_filters` | 256 | Channels in residual blocks |
| `num_res_blocks` | 15 | Depth of the network |
| `se_reduction` | 8 | SE-block compression ratio |
| **MCTS** | | |
| `num_mcts_sims` | 800 | Simulations per move |
| `mcts_leaf_batch` | 8 | Leaves evaluated per batch |
| `cpuct_init` | 2.5 | Initial exploration constant |
| `cpuct_base` | 19652 | Dynamic cPUCT log-base |
| `fpu_reduction` | 0.25 | First-play urgency reduction |
| `dirichlet_alpha` | 0.3 | Dirichlet noise parameter |
| `dirichlet_epsilon` | 0.25 | Noise mixing weight |
| `temp_threshold_move` | 30 | Temperature → 0 after this move |
| **Self-Play** | | |
| `self_play_games` | 500 | Games per iteration |
| `max_game_moves` | 200 | Maximum moves before draw |
| `resign_threshold` | -0.9 | Resign when V < this |
| `resign_consecutive` | 10 | Must be below for N moves |
| **Training** | | |
| `optimizer` | SGD | Paper's optimizer |
| `batch_size` | 256 | Minibatch size |
| `train_steps` | 1000 | Gradient steps per iteration |
| `learning_rate` | 0.01 | Initial SGD learning rate |
| `momentum` | 0.9 | SGD momentum |
| `weight_decay` | 1e-4 | L2 regularisation |
| `data_window` | 20 | Use last N iterations' data |
| **Evaluation** | | |
| `eval_games` | 40 | Games per evaluation |
| `accept_threshold` | 0.55 | Win-rate to accept new model |
| `use_eval_gating` | False | Paper: always use latest |
| **Pipeline** | | |
| `num_iterations` | 5 | Total training iterations |
| `num_workers` | 4 | Parallel self-play workers |
| `device` | mps | Apple Silicon GPU |

---

## 11. Worked Example — One Complete MCTS Simulation

Let's trace **one simulation** from start to finish in a mid-game position.

**Position**: White to move, move 15. Root has been expanded (from a previous simulation), some children already have visits.

```
Root: (White, N=50)
  ├── Nf3→e5 (Black, N=20, W=-3.0, P=0.25)  ← Q = -3.0/20 = -0.15
  ├── Bb5→d3 (Black, N=15, W=-1.5, P=0.20)  ← Q = -1.5/15 = -0.10
  ├── O-O     (Black, N=10, W=-0.5, P=0.15)  ← Q = -0.5/10 = -0.05
  ├── d2→d4  (Black, N=5,  W=+0.2, P=0.10)  ← Q = +0.2/5  = +0.04
  └── h2→h3  (Black, N=0,  P=0.02)           ← unvisited
```

### Step 1: SELECT

Compute PUCT scores (c ≈ 2.5, √50 ≈ 7.07):

```
Nf3→e5: q=-(-0.15)=0.15,  u = 2.5 × 0.25 × 7.07 / 21 = 0.211  → score = 0.361
Bb5→d3: q=-(-0.10)=0.10,  u = 2.5 × 0.20 × 7.07 / 16 = 0.221  → score = 0.321
O-O:    q=-(-0.05)=0.05,  u = 2.5 × 0.15 × 7.07 / 11 = 0.241  → score = 0.291
d2→d4:  q=-(+0.04)=-0.04, u = 2.5 × 0.10 × 7.07 / 6  = 0.295  → score = 0.255
h2→h3:  q=FPU=0.15-0.25=-0.10, u = 2.5 × 0.02 × 7.07 / 1 = 0.354  → score = 0.254
```

**Selected: Nf3→e5** (0.361).

We apply virtual loss: `Root.N → 51`, `Ne5_node.N → 21`.

Now at the Ne5 node (Black to move), suppose it has children — we continue selecting until we reach an unexpanded leaf. Say we traverse Ne5 → ...e6 → ...Bc4 → [LEAF].

### Step 2: EXPAND

The leaf node has not been expanded. We run the neural network:

```python
encoded = encode_board(leaf.board)                # (119, 8, 8)
p_logits, _ = policy_net(encoded)                  # (4672,)
p_probs = softmax(p_logits)

# Create children for all legal moves with their priors:
# Qd3: prior=0.22,  Rc1: prior=0.18,  Nbd2: prior=0.12, ...
```

### Step 3: EVALUATE

```python
_, v = value_net(encoded)
v = 0.15    # network thinks this position is slightly good for the player to move
```

### Step 4: BACKUP

```
Leaf (to-move perspective): value = +0.15
  ← Bc4 node: value_sum += +0.15, then value = -0.15
  ← e6 node:  value_sum += -0.15, then value = +0.15
  ← Ne5 node: value_sum += +0.15, then value = -0.15
  (root visit count was already incremented by virtual loss)
```

After this simulation, Ne5's stats become: N=21, W = -3.0 + 0.15 = -2.85, Q = -2.85/21 = -0.1357

---

## 12. Worked Example — One Complete Iteration

**Iteration 3** (after 2 previous iterations of ~100,000 positions each):

### Phase A: Self-Play (500 games, 4 workers)

```
Worker 0: plays game 1 of 125
  Move 1: MCTS(800 sims) → π=[0.0, ..., 0.31, ..., 0.25, ...], V=0.05
           Sample action (temp=1.0) → e2e4
  Move 2: MCTS(800 sims, reusing subtree from e4) → π, V=0.02
           Sample action → e7e5
  ...
  Move 45: MCTS(800 sims) → π, V=-0.12
           Temp=0 (move 45 > 30), pick argmax → Rd1
  ...
  Move 87: Checkmate! White wins (z=+1)

  → 87 training examples:
    (state_1, π_1, +1.0)    ← White played move 1, White won
    (state_2, π_2, -1.0)    ← Black played move 2, White won (bad for Black)
    (state_3, π_3, +1.0)    ← White played move 3, White won
    ...
  → Flush all 87 examples to disk immediately

Worker 0: plays game 2 of 125 ...
```

After all 500 games:
```
MCTS/data/iter_0002_states.npy     # ~50,000 positions × (119,8,8) ≈ 2.3 GB
MCTS/data/iter_0002_policies.npy   # ~50,000 × 4672 ≈ 890 MB
MCTS/data/iter_0002_values.npy     # ~50,000 × 1 ≈ 200 KB
```

### Phase B: Training (1000 steps)

```
Replay buffer: iter_0000 + iter_0001 + iter_0002 ≈ 150,000 positions
Total disk: ~9 GB memory-mapped

for step in 1..1000:
    batch = random 256 positions from all 3 iterations
    forward: logits, v = net(batch_states)
    loss = MSE(v, z) + CE(logits, π) = 0.42 + 3.85 = 4.27  (early)
    backward + SGD step

Final loss ≈ 0.35 + 3.20 = 3.55 (after 1000 steps)
```

### Phase C: Checkpoint

```
Save: MCTS/checkpoints/latest.pt
Save: MCTS/checkpoints/weights_iter_0002.pt
```

→ Proceed to iteration 4.
