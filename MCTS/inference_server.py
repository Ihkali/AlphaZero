"""
MCTS/inference_server.py — Centralized GPU/MPS inference server for self-play.

Instead of each CPU worker holding its own copy of the neural network,
a single InferenceServer process owns the model on the accelerator
(MPS / CUDA).  Workers send encoded board states via a shared queue;
the server collects them into a mega-batch, runs one forward pass,
and returns results to each worker's private response queue.

Architecture:

  ┌──────────┐                          ┌────────────────────┐
  │ Worker 0  │───► request_queue ──────►│                    │
  │ Worker 1  │───►                      │  InferenceServer   │
  │ Worker 2  │───►                      │  (model on MPS)    │
  │ Worker 3  │───►                      │                    │
  └──────────┘◄── response_queues[w] ◄──│                    │
                                         └────────────────────┘

Benefits:
  • Only one copy of the model in memory
  • Inference runs on GPU/MPS instead of CPU
  • Cross-worker batching amortises per-batch overhead
"""

import time
import queue
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from MCTS.config import Config


# Sentinel value to signal the server to shut down
SHUTDOWN = "__SHUTDOWN__"


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE SERVER LOOP  (runs in its own process)
# ═══════════════════════════════════════════════════════════════════════════

def inference_server_loop(
    model_state_dict: dict,
    device: str,
    request_queue: mp.Queue,
    response_queues: list,
    max_batch_size: int = Config.inference_batch_size,
    batch_timeout_s: float = Config.inference_batch_timeout,
):
    """Main loop for the centralized inference server.

    Blocks on the request_queue, collects up to ``max_batch_size``
    requests (or waits ``batch_timeout_s`` seconds), runs one batched
    forward pass on ``device``, and distributes results back.

    Request format:  (worker_id: int, states: np.ndarray[N, C, H, W])
    Response format: (policy_logits: np.ndarray[N, P], values: np.ndarray[N, 1])
    """
    from MCTS.model import AlphaZeroNet

    # Auto-detect architecture from state dict
    num_filters = model_state_dict["conv_block.conv.weight"].shape[0]
    num_blocks = sum(
        1 for k in model_state_dict
        if k.endswith(".conv1.weight") and k.startswith("res_blocks.")
    )

    net = AlphaZeroNet(num_filters=num_filters, num_blocks=num_blocks).to(device)
    net.load_state_dict(model_state_dict)
    net.eval()

    print(f"  [InferenceServer] Ready on {device} "
          f"(max_batch={max_batch_size}, "
          f"timeout={batch_timeout_s * 1000:.0f}ms)")

    total_batches = 0
    total_states = 0

    while True:
        # ── Block until the first request arrives ─────────────────────
        try:
            req = request_queue.get(timeout=120)
        except queue.Empty:
            continue

        if req == SHUTDOWN:
            break

        batch_requests = [req]

        # ── Phase 1: non-blocking drain — grab everything already queued
        #    (e.g. other workers that sent requests before us) ─────────
        while len(batch_requests) < max_batch_size:
            try:
                req = request_queue.get_nowait()
                if req == SHUTDOWN:
                    request_queue.put(SHUTDOWN)
                    break
                batch_requests.append(req)
            except queue.Empty:
                break

        # ── Phase 2: brief wait for stragglers (skip if queue was
        #    empty → lone worker / last game fires instantly) ──────────
        if len(batch_requests) < max_batch_size and len(batch_requests) > 0:
            deadline = time.monotonic() + batch_timeout_s
            while len(batch_requests) < max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = request_queue.get(timeout=max(remaining, 1e-5))
                    if req == SHUTDOWN:
                        request_queue.put(SHUTDOWN)
                        break
                    batch_requests.append(req)
                except queue.Empty:
                    break

        if not batch_requests:
            continue

        # ── Build mega-batch from all collected requests ──────────────
        all_states = []
        request_info = []          # (worker_id, num_states_in_this_request)
        for worker_id, states_np in batch_requests:
            all_states.append(states_np)
            request_info.append((worker_id, states_np.shape[0]))

        mega_batch = np.concatenate(all_states, axis=0)
        tensor = torch.from_numpy(mega_batch).to(device)

        # ── Single forward pass on GPU/MPS ────────────────────────────
        with torch.no_grad():
            p_logits, v_vals = net(tensor)
            p_np = p_logits.cpu().numpy()
            v_np = v_vals.cpu().numpy()

        # ── Distribute slices back to each worker ─────────────────────
        offset = 0
        for worker_id, n in request_info:
            p_slice = p_np[offset:offset + n]
            v_slice = v_np[offset:offset + n]
            response_queues[worker_id].put((p_slice, v_slice))
            offset += n

        total_batches += 1
        total_states += mega_batch.shape[0]

    avg_batch = total_states / max(total_batches, 1)
    print(f"  [InferenceServer] Done — "
          f"{total_batches} batches, {total_states} states "
          f"(avg {avg_batch:.1f} states/batch)")


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE CLIENT  (used by worker processes as a drop-in model proxy)
# ═══════════════════════════════════════════════════════════════════════════

class InferenceClient:
    """Drop-in replacement for AlphaZeroNet that routes inference
    requests to the centralized InferenceServer.

    Implements ``__call__`` and ``predict`` so that ``mcts_search``
    can use it exactly like a real neural network.
    """

    def __init__(self, worker_id: int,
                 request_queue: mp.Queue,
                 response_queue: mp.Queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue

    # ── forward pass (matches nn.Module.__call__) ─────────────────────

    def __call__(self, tensor):
        """Batched forward pass → (policy_logits, values) as CPU tensors.

        Used by the batched leaf evaluation inside ``mcts_search``.
        """
        if isinstance(tensor, torch.Tensor):
            states_np = tensor.detach().cpu().numpy()
        else:
            states_np = np.asarray(tensor, dtype=np.float32)
        if states_np.ndim == 3:
            states_np = states_np[np.newaxis]

        self.request_queue.put((self.worker_id, states_np))
        p_logits_np, v_vals_np = self.response_queue.get()

        return torch.from_numpy(p_logits_np), torch.from_numpy(v_vals_np)

    # ── single-state prediction (matches AlphaZeroNet.predict) ────────

    def predict(self, board_tensor):
        """Single-state prediction → (policy_probs_np, value_float).

        Used by ``_expand`` for the root node.
        """
        if isinstance(board_tensor, torch.Tensor):
            states_np = board_tensor.detach().cpu().numpy()
        else:
            states_np = np.asarray(board_tensor, dtype=np.float32)
        if states_np.ndim == 3:
            states_np = states_np[np.newaxis]

        self.request_queue.put((self.worker_id, states_np))
        p_logits_np, v_vals_np = self.response_queue.get()

        # Softmax over policy logits (same as AlphaZeroNet.predict)
        p_probs = F.softmax(
            torch.from_numpy(p_logits_np), dim=1
        ).numpy()[0]
        value = float(v_vals_np.ravel()[0])
        return p_probs, value

    # ── Compatibility stubs ───────────────────────────────────────────

    def eval(self):
        """No-op — server model is always in eval mode."""
        return self

    def state_dict(self):
        """No-op — state lives on the server."""
        return {}
