"""
SL/self_play_gui.py — Watch the SL-trained model play against itself (GUI).

Both sides are controlled by the policy-head network (no MCTS).
A side panel shows live eval bars, move log, and game statistics.

Usage:
    python SL/self_play_gui.py                                     # latest checkpoint
    python SL/self_play_gui.py --checkpoint SL/checkpoints/sl_best.pt
    python SL/self_play_gui.py --temperature 0.3
    python SL/self_play_gui.py --delay 0.0                      # seconds between moves
    python SL/self_play_gui.py --games 10                          # auto-reset after each game
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import time
import threading
import chess
import torch
import torch.nn.functional as F
import numpy as np
import pygame

from SL.config import Config
from SL.encode import encode_board, move_to_index, index_to_move, get_legal_mask
from SL.model import AlphaZeroNet

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════
SQ_SIZE = 80
BOARD_PX = SQ_SIZE * 8
PANEL_W = 300
WIN_W = BOARD_PX + PANEL_W
WIN_H = BOARD_PX

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
HIGHLIGHT = (246, 246, 105, 160)
LAST_MOVE_CLR = (205, 210, 106, 130)
CHECK_CLR = (235, 64, 52, 160)
PANEL_BG = (40, 40, 40)
TEXT_CLR = (220, 220, 220)
ACCENT = (76, 175, 80)
MUTED = (150, 150, 150)
WHITE_BAR = (220, 220, 220)
BLACK_BAR = (60, 60, 60)

PIECE_UNICODE = {
    (chess.KING, chess.WHITE): "♔", (chess.QUEEN, chess.WHITE): "♕",
    (chess.ROOK, chess.WHITE): "♖", (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘", (chess.PAWN, chess.WHITE): "♙",
    (chess.KING, chess.BLACK): "♚", (chess.QUEEN, chess.BLACK): "♛",
    (chess.ROOK, chess.BLACK): "♜", (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞", (chess.PAWN, chess.BLACK): "♟",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint auto-discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(ckpt_dir: str = Config.checkpoint_dir) -> str | None:
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    best = os.path.join(ckpt_dir, "sl_best.pt")
    if os.path.isfile(best):
        return best
    pattern = os.path.join(ckpt_dir, "sl_step_*.pt")
    files = sorted(glob.glob(pattern))
    if files:
        return files[-1]
    final = os.path.join(ckpt_dir, "final_model.pt")
    if os.path.isfile(final):
        return final
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  AI move selection
# ═══════════════════════════════════════════════════════════════════════════

def ai_select_move(net, board: chess.Board, device: str,
                   temperature: float = 0.1) -> tuple[chess.Move, float]:
    """Pick a move using the policy head, masked to legal moves."""
    encoded = encode_board(board)
    state_t = torch.from_numpy(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        p_logits, v = net(state_t)

    p_logits = p_logits[0].cpu()
    value = v.item()

    # Mask illegal moves
    legal_mask = torch.from_numpy(get_legal_mask(board))
    p_logits[legal_mask == 0] = float("-inf")

    if temperature <= 0:
        action = p_logits.argmax().item()
    else:
        probs = F.softmax(p_logits / temperature, dim=0)
        action = torch.multinomial(probs, 1).item()

    move = index_to_move(action, board)
    if move not in board.legal_moves:
        move = list(board.legal_moves)[0]

    return move, value


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Play GUI
# ═══════════════════════════════════════════════════════════════════════════

class SelfPlayGUI:
    def __init__(self, net, temperature: float, device: str,
                 move_delay: float, max_games: int):
        self.net = net
        self.temperature = temperature
        self.device = device
        self.move_delay = move_delay
        self.max_games = max_games  # 0 = infinite

        # Game state
        self.board = chess.Board()
        self.last_move = None
        self.move_log: list[str] = []
        self.white_eval = None       # value from White's perspective
        self.black_eval = None       # value from Black's perspective
        self.game_over = False
        self.paused = False
        self.status = "Playing..."

        # Statistics across games
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0

        # AI thread control
        self._thinking = False
        self._stop = False

        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("AlphaZero SL — Self-Play")
        self.clock = pygame.time.Clock()

        self.piece_font = pygame.font.SysFont("Apple Symbols", SQ_SIZE - 12)
        self.label_font = pygame.font.SysFont("Helvetica", 14)
        self.panel_font = pygame.font.SysFont("Helvetica", 15)
        self.panel_bold = pygame.font.SysFont("Helvetica", 15, bold=True)
        self.title_font = pygame.font.SysFont("Helvetica", 20, bold=True)
        self.status_font = pygame.font.SysFont("Helvetica", 16, bold=True)
        self.small_font = pygame.font.SysFont("Helvetica", 12)

    # ── Coordinate helpers ────────────────────────────────────────────
    def sq_to_px(self, sq):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        return file * SQ_SIZE, (7 - rank) * SQ_SIZE

    def px_to_sq(self, px, py):
        col, row = px // SQ_SIZE, py // SQ_SIZE
        file, rank = col, 7 - row
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, rank)
        return None

    # ── Drawing ───────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(PANEL_BG)
        self._draw_board()
        self._draw_overlays()
        self._draw_pieces()
        self._draw_labels()
        self._draw_panel()
        pygame.display.flip()

    def _draw_board(self):
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            color = LIGHT_SQ if (f + r) % 2 == 1 else DARK_SQ
            x, y = self.sq_to_px(sq)
            pygame.draw.rect(self.screen, color, (x, y, SQ_SIZE, SQ_SIZE))

    def _draw_overlays(self):
        overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)

        if self.last_move:
            overlay.fill(LAST_MOVE_CLR)
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                self.screen.blit(overlay, self.sq_to_px(sq))

        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                overlay.fill(CHECK_CLR)
                self.screen.blit(overlay, self.sq_to_px(king_sq))

    def _draw_pieces(self):
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                x, y = self.sq_to_px(sq)
                char = PIECE_UNICODE[(piece.piece_type, piece.color)]
                clr = (255, 255, 255) if piece.color == chess.BLACK else (20, 20, 20)
                try:
                    text = self.piece_font.render(char, True, clr)
                    if text.get_width() == 0:
                        raise ValueError
                except (pygame.error, ValueError):
                    fb = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
                          chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}
                    text = self.panel_bold.render(fb[piece.piece_type], True, clr)
                rect = text.get_rect(center=(x + SQ_SIZE // 2,
                                             y + SQ_SIZE // 2))
                self.screen.blit(text, rect)

    def _draw_labels(self):
        for i in range(8):
            lbl = chr(ord('a') + i)
            clr = DARK_SQ if i % 2 == 0 else LIGHT_SQ
            self.screen.blit(self.label_font.render(lbl, True, clr),
                             (i * SQ_SIZE + SQ_SIZE - 12, BOARD_PX - 16))
            r = 7 - i
            self.screen.blit(self.label_font.render(str(r + 1), True,
                             DARK_SQ if (7 - i) % 2 == 0 else LIGHT_SQ),
                             (3, i * SQ_SIZE + 3))

    def _draw_panel(self):
        px, py = BOARD_PX + 10, 10

        # Title
        title = self.title_font.render("Self-Play", True, ACCENT)
        self.screen.blit(title, (px, py)); py += 35

        # Setup info
        self.screen.blit(self.panel_font.render(
            f"White ♔ vs Black ♚  (T={self.temperature})", True, MUTED),
            (px, py)); py += 22
        self.screen.blit(self.panel_font.render(
            f"Delay: {self.move_delay:.1f}s  |  Move {len(self.move_log)}",
            True, MUTED), (px, py)); py += 28

        # Status
        if self.game_over:
            sc = (255, 200, 50)
        elif self.paused:
            sc = (255, 165, 0)
        else:
            sc = ACCENT
        self.screen.blit(self.status_font.render(self.status, True, sc),
                         (px, py)); py += 30

        # Eval bars
        bw = PANEL_W - 30
        bh = 16

        # White eval
        if self.white_eval is not None:
            ev = (self.white_eval + 1) / 2  # map [-1, 1] -> [0, 1]
            pygame.draw.rect(self.screen, (80, 80, 80),
                             (px, py, bw, bh), border_radius=3)
            fw = max(2, int(ev * bw))
            pygame.draw.rect(self.screen, WHITE_BAR,
                             (px, py, fw, bh), border_radius=3)
            self.screen.blit(self.label_font.render(
                f"White V(s): {self.white_eval:+.3f}", True, TEXT_CLR),
                (px, py + bh + 2)); py += bh + 20

        # Black eval
        if self.black_eval is not None:
            ev = (self.black_eval + 1) / 2
            pygame.draw.rect(self.screen, (80, 80, 80),
                             (px, py, bw, bh), border_radius=3)
            fw = max(2, int(ev * bw))
            pygame.draw.rect(self.screen, (100, 100, 100),
                             (px, py, fw, bh), border_radius=3)
            self.screen.blit(self.label_font.render(
                f"Black V(s): {self.black_eval:+.3f}", True, TEXT_CLR),
                (px, py + bh + 2)); py += bh + 20

        py += 5

        # Game stats
        pygame.draw.line(self.screen, (70, 70, 70),
                         (px, py), (px + PANEL_W - 30, py)); py += 10
        total = self.games_played
        self.screen.blit(self.panel_bold.render(
            f"Score  ({total} game{'s' if total != 1 else ''})", True, TEXT_CLR),
            (px, py)); py += 22
        self.screen.blit(self.panel_font.render(
            f"White wins: {self.white_wins}", True, TEXT_CLR),
            (px, py)); py += 20
        self.screen.blit(self.panel_font.render(
            f"Black wins: {self.black_wins}", True, TEXT_CLR),
            (px, py)); py += 20
        self.screen.blit(self.panel_font.render(
            f"Draws:      {self.draws}", True, TEXT_CLR),
            (px, py)); py += 28

        # Move log
        pygame.draw.line(self.screen, (70, 70, 70),
                         (px, py), (px + PANEL_W - 30, py)); py += 10
        self.screen.blit(self.panel_bold.render("Moves", True, TEXT_CLR),
                         (px, py)); py += 24

        max_lines = (WIN_H - py - 50) // 20
        pairs = []
        for i in range(0, len(self.move_log), 2):
            n = i // 2 + 1
            w = self.move_log[i] if i < len(self.move_log) else ""
            b = self.move_log[i + 1] if i + 1 < len(self.move_log) else ""
            pairs.append(f"{n}. {w}  {b}")
        visible = pairs[-max_lines:] if len(pairs) > max_lines else pairs
        for line in visible:
            self.screen.blit(self.panel_font.render(line, True, TEXT_CLR),
                             (px, py)); py += 20

        # Controls help
        self.screen.blit(self.label_font.render(
            "SPACE=pause  R=new game  Q=quit", True, MUTED),
            (px, WIN_H - 40))
        self.screen.blit(self.label_font.render(
            "UP/DOWN = adjust delay", True, MUTED),
            (px, WIN_H - 22))

    # ── AI thread ─────────────────────────────────────────────────────
    def _play_loop(self):
        """Background thread: plays moves one at a time with delay."""
        while not self._stop:
            if self.paused or self.game_over:
                time.sleep(0.05)
                continue

            if self.board.is_game_over():
                self._end_game()
                # Auto-start next game if max_games not reached
                if self.max_games > 0 and self.games_played >= self.max_games:
                    self.status = f"Done — {self.games_played} games"
                    self._stop = True
                    break
                time.sleep(1.5)  # pause between games
                if not self._stop:
                    self._reset_board()
                continue

            # Enforce max move limit (prevent endless games)
            if len(self.move_log) >= 500:
                self._end_game()
                continue

            self._thinking = True
            move, value = ai_select_move(
                self.net, self.board, self.device,
                temperature=self.temperature,
            )

            # Store eval from side-to-move perspective
            if self.board.turn == chess.WHITE:
                self.white_eval = value
            else:
                self.black_eval = value

            san = self.board.san(move)
            self.board.push(move)
            self.last_move = move
            self.move_log.append(san)

            side = "White" if self.board.turn == chess.BLACK else "Black"
            self.status = f"{side} played {san}"
            self._thinking = False

            time.sleep(self.move_delay)

    def _end_game(self):
        self.game_over = True
        self.games_played += 1
        result = self.board.result()
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            if winner == "White":
                self.white_wins += 1
            else:
                self.black_wins += 1
            self.status = f"{winner} wins by checkmate  ({result})"
        elif self.board.is_stalemate():
            self.draws += 1
            self.status = "Draw — Stalemate"
        elif self.board.is_insufficient_material():
            self.draws += 1
            self.status = "Draw — Insufficient material"
        elif self.board.can_claim_threefold_repetition():
            self.draws += 1
            self.status = "Draw — Threefold repetition"
        elif self.board.can_claim_fifty_moves():
            self.draws += 1
            self.status = "Draw — 50-move rule"
        elif len(self.move_log) >= 500:
            self.draws += 1
            self.status = "Draw — Move limit (500)"
        else:
            self.draws += 1
            self.status = f"Game over: {result}"
        print(f"  Game {self.games_played}: {self.status} "
              f"({len(self.move_log)} moves)")

    def _reset_board(self):
        self.board = chess.Board()
        self.last_move = None
        self.move_log = []
        self.white_eval = None
        self.black_eval = None
        self.game_over = False
        self.status = "Playing..."

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self):
        thread = threading.Thread(target=self._play_loop, daemon=True)
        thread.start()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        if self.paused:
                            self.status = "Paused"
                        else:
                            self.status = "Playing..."
                    elif event.key == pygame.K_r:
                        self._reset_board()
                    elif event.key == pygame.K_UP:
                        self.move_delay = max(0.0, self.move_delay - 0.1)
                    elif event.key == pygame.K_DOWN:
                        self.move_delay = min(5.0, self.move_delay + 0.1)

            self.draw()
            self.clock.tick(30)

        self._stop = True
        thread.join(timeout=2)
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Watch the SL model play against itself (GUI)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default 0.3)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between moves (default 0.8)")
    parser.add_argument("--games", type=int, default=0,
                        help="Number of games to play (0 = infinite)")
    parser.add_argument("--device", type=str, default=Config.device)
    args = parser.parse_args()

    device = args.device

    ckpt = args.checkpoint or find_latest_checkpoint()

    print("Loading SL model for self-play...")
    if ckpt and os.path.isfile(ckpt):
        data = torch.load(ckpt, map_location=device, weights_only=False)
        state = data["model_state"]
        num_filters = state["conv_block.conv.weight"].shape[0]
        num_blocks = sum(1 for k in state if k.endswith(".conv1.weight")
                         and k.startswith("res_blocks."))
        vh_hidden = state["value_head.fc1.weight"].shape[0]
        print(f"  Arch: {num_filters} filters, {num_blocks} blocks, "
              f"vh_hidden={vh_hidden}")
        net = AlphaZeroNet(
            num_filters=num_filters,
            num_blocks=num_blocks,
        ).to(device)
        net.load_state_dict(state)
        print(f"  Loaded: {ckpt}")
    else:
        print(f"  No checkpoint found (looked in {Config.checkpoint_dir}/)")
        print("  Running with untrained network.")
        net = AlphaZeroNet().to(device)
    net.eval()

    print(f"  Temperature: {args.temperature}")
    print(f"  Move delay:  {args.delay}s")
    print(f"  Games:       {'infinite' if args.games == 0 else args.games}")
    print(f"  Controls:    SPACE=pause  R=new game  UP/DOWN=speed  Q=quit\n")

    gui = SelfPlayGUI(net, args.temperature, device, args.delay, args.games)
    gui.run()


if __name__ == "__main__":
    main()
