"""
MCTS/play_gui.py — Play against the AlphaZero MCTS model (GUI).

The AI uses full MCTS search (800 simulations by default) with batched
neural-network evaluation — the same search used during self-play.

Usage:
    python MCTS/play_gui.py                                        # latest checkpoint, White
    python MCTS/play_gui.py --color black                          # play as Black
    python MCTS/play_gui.py --checkpoint MCTS/checkpoints/weights_iter_0010.pt
    python MCTS/play_gui.py --sims 400                             # fewer sims = faster
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import threading
import chess
import torch
import numpy as np
import pygame

from MCTS.config import Config
from MCTS.encode import encode_board, index_to_move
from MCTS.model import AlphaZeroNet, load_checkpoint
from MCTS.mcts import mcts_search, select_action

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════
SQ_SIZE = 80
BOARD_PX = SQ_SIZE * 8
PANEL_W = 280
WIN_W = BOARD_PX + PANEL_W
WIN_H = BOARD_PX

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
HIGHLIGHT = (246, 246, 105, 160)
LEGAL_DOT = (100, 100, 100, 120)
LAST_MOVE_CLR = (205, 210, 106, 130)
CHECK_CLR = (235, 64, 52, 160)
PANEL_BG = (40, 40, 40)
TEXT_CLR = (220, 220, 220)
ACCENT = (76, 175, 80)
MUTED = (150, 150, 150)

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
    """Return the path to latest.pt, or the highest-iteration weight file."""
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    # Fallback: pick highest-numbered weights_iter_*.pt
    pattern = os.path.join(ckpt_dir, "weights_iter_*.pt")
    files = sorted(glob.glob(pattern))
    if files:
        return files[-1]
    # Fallback: final_model.pt
    final = os.path.join(ckpt_dir, "final_model.pt")
    if os.path.isfile(final):
        return final
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════

class PlayGUI:
    def __init__(self, net, human_is_white: bool, num_sims: int, device: str):
        self.net = net
        self.human_is_white = human_is_white
        self.num_sims = num_sims
        self.device = device

        self.board = chess.Board()
        self.selected_sq = None
        self.legal_targets = []
        self.last_move = None
        self.move_log = []
        self.status = "Your move" if self.is_human_turn() else "AI thinking..."
        self.ai_thinking = False
        self.ai_eval = None          # root V after AI move
        self.position_eval = None    # V(s) for current position
        self.game_over = False
        self.promotion_pending = None

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(f"AlphaZero MCTS — {num_sims} sims/move")
        self.clock = pygame.time.Clock()

        self.piece_font = pygame.font.SysFont("Apple Symbols", SQ_SIZE - 12)
        self.label_font = pygame.font.SysFont("Helvetica", 14)
        self.panel_font = pygame.font.SysFont("Helvetica", 15)
        self.panel_bold = pygame.font.SysFont("Helvetica", 15, bold=True)
        self.title_font = pygame.font.SysFont("Helvetica", 20, bold=True)
        self.status_font = pygame.font.SysFont("Helvetica", 16, bold=True)
        self.promo_font = pygame.font.SysFont("Apple Symbols", 48)

    # ── Coordinate helpers ────────────────────────────────────────────
    def sq_to_px(self, sq):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if self.human_is_white:
            return file * SQ_SIZE, (7 - rank) * SQ_SIZE
        else:
            return (7 - file) * SQ_SIZE, rank * SQ_SIZE

    def px_to_sq(self, px, py):
        col, row = px // SQ_SIZE, py // SQ_SIZE
        if self.human_is_white:
            file, rank = col, 7 - row
        else:
            file, rank = 7 - col, row
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, rank)
        return None

    def is_human_turn(self):
        return (self.board.turn == chess.WHITE) == self.human_is_white

    # ── Drawing ───────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(PANEL_BG)
        self.draw_board()
        self.draw_overlays()
        self.draw_pieces()
        self.draw_labels()
        self.draw_panel()
        if self.promotion_pending:
            self.draw_promotion_dialog()
        pygame.display.flip()

    def draw_board(self):
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            color = LIGHT_SQ if (f + r) % 2 == 1 else DARK_SQ
            x, y = self.sq_to_px(sq)
            pygame.draw.rect(self.screen, color, (x, y, SQ_SIZE, SQ_SIZE))

    def draw_overlays(self):
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

        if self.selected_sq is not None:
            overlay.fill(HIGHLIGHT)
            self.screen.blit(overlay, self.sq_to_px(self.selected_sq))

        dot_surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        for sq in self.legal_targets:
            x, y = self.sq_to_px(sq)
            if self.board.piece_at(sq):
                pygame.draw.circle(dot_surf, LEGAL_DOT,
                                   (SQ_SIZE // 2, SQ_SIZE // 2),
                                   SQ_SIZE // 2 - 2, 4)
            else:
                dot_surf.fill((0, 0, 0, 0))
                pygame.draw.circle(dot_surf, LEGAL_DOT,
                                   (SQ_SIZE // 2, SQ_SIZE // 2),
                                   SQ_SIZE // 7)
            self.screen.blit(dot_surf, (x, y))

    def draw_pieces(self):
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

    def draw_labels(self):
        for i in range(8):
            f = i if self.human_is_white else 7 - i
            lbl = chr(ord('a') + f)
            clr = DARK_SQ if i % 2 == 0 else LIGHT_SQ
            self.screen.blit(self.label_font.render(lbl, True, clr),
                             (i * SQ_SIZE + SQ_SIZE - 12, BOARD_PX - 16))
            r = 7 - i if self.human_is_white else i
            self.screen.blit(self.label_font.render(str(r + 1), True,
                             DARK_SQ if (7 - i) % 2 == 0 else LIGHT_SQ),
                             (3, i * SQ_SIZE + 3))

    def draw_panel(self):
        px, py = BOARD_PX + 10, 10

        title = self.title_font.render("AlphaZero MCTS", True, ACCENT)
        self.screen.blit(title, (px, py)); py += 35

        you = "White ♔" if self.human_is_white else "Black ♚"
        ai = "Black ♚" if self.human_is_white else "White ♔"
        self.screen.blit(self.panel_font.render(f"You: {you}", True, TEXT_CLR),
                         (px, py)); py += 22
        self.screen.blit(self.panel_font.render(
            f"AI:  {ai}  ({self.num_sims} sims)", True, MUTED),
            (px, py)); py += 30

        if self.game_over:
            sc = (255, 200, 50)
        elif self.ai_thinking:
            sc = (255, 165, 0)
        else:
            sc = ACCENT
        self.screen.blit(self.status_font.render(self.status, True, sc),
                         (px, py)); py += 30

        # AI eval bar
        if self.ai_eval is not None:
            ev = (self.ai_eval + 1) / 2
            bw = PANEL_W - 30; bh = 14
            pygame.draw.rect(self.screen, (80, 80, 80),
                             (px, py, bw, bh), border_radius=3)
            fw = max(2, int(ev * bw))
            pygame.draw.rect(self.screen, (220, 220, 220),
                             (px, py, fw, bh), border_radius=3)
            self.screen.blit(self.label_font.render(
                f"AI eval: {self.ai_eval:+.2f}", True, MUTED),
                (px, py + bh + 2))
            py += bh + 22

        # V(s) bar for current position
        if self.position_eval is not None:
            v = self.position_eval
            ev2 = (v + 1) / 2
            bw = PANEL_W - 30; bh = 14
            pygame.draw.rect(self.screen, (80, 80, 80),
                             (px, py, bw, bh), border_radius=3)
            fw2 = max(2, int(ev2 * bw))
            bar_clr = (76, 175, 80) if v > 0.1 else (235, 64, 52) if v < -0.1 else (180, 180, 180)
            pygame.draw.rect(self.screen, bar_clr,
                             (px, py, fw2, bh), border_radius=3)
            self.screen.blit(self.label_font.render(
                f"V(s): {v:+.3f}", True, TEXT_CLR),
                (px, py + bh + 2))
            py += bh + 22
        py += 10

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
        for line in (pairs[-max_lines:] if len(pairs) > max_lines else pairs):
            self.screen.blit(self.panel_font.render(line, True, TEXT_CLR),
                             (px, py)); py += 20

        self.screen.blit(self.label_font.render(
            "Click piece → target  |  R=reset  Q=quit", True, MUTED),
            (px, WIN_H - 25))

    def draw_promotion_dialog(self):
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        dw, dh = 320, 120
        dx = (BOARD_PX - dw) // 2
        dy = (BOARD_PX - dh) // 2
        pygame.draw.rect(self.screen, (50, 50, 50),
                         (dx, dy, dw, dh), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT,
                         (dx, dy, dw, dh), 2, border_radius=10)

        t = self.panel_bold.render("Promote to:", True, TEXT_CLR)
        self.screen.blit(t, (dx + dw // 2 - t.get_width() // 2, dy + 10))

        color = chess.WHITE if self.board.turn == chess.WHITE else chess.BLACK
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        self.promo_rects = []
        mx, my = pygame.mouse.get_pos()
        for i, pt in enumerate(pieces):
            char = PIECE_UNICODE[(pt, color)]
            tc = (255, 255, 255) if color == chess.BLACK else (20, 20, 20)
            try:
                txt = self.promo_font.render(char, True, tc)
                if txt.get_width() == 0:
                    raise ValueError
            except (pygame.error, ValueError):
                fb = {chess.QUEEN: "Q", chess.ROOK: "R",
                      chess.BISHOP: "B", chess.KNIGHT: "N"}
                txt = self.title_font.render(fb[pt], True, tc)
            bx = dx + 20 + i * 75
            by = dy + 50
            rect = pygame.Rect(bx, by, 60, 60)
            self.promo_rects.append((rect, pt))
            if rect.collidepoint(mx, my):
                pygame.draw.rect(self.screen, (100, 100, 100),
                                 rect, border_radius=6)
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    # ── Game logic ────────────────────────────────────────────────────
    def handle_click(self, px, py):
        if self.game_over or self.ai_thinking or not self.is_human_turn():
            return

        if self.promotion_pending:
            for rect, pt in self.promo_rects:
                if rect.collidepoint(px, py):
                    fsq, tsq = self.promotion_pending
                    self.make_move(chess.Move(fsq, tsq, promotion=pt))
                    self.promotion_pending = None
                    return
            return

        sq = self.px_to_sq(px, py)
        if sq is None:
            return

        if self.selected_sq is not None:
            if sq in self.legal_targets:
                piece = self.board.piece_at(self.selected_sq)
                if (piece and piece.piece_type == chess.PAWN
                        and chess.square_rank(sq) in (0, 7)):
                    self.promotion_pending = (self.selected_sq, sq)
                    self.selected_sq = None
                    self.legal_targets = []
                    return
                self.make_move(chess.Move(self.selected_sq, sq))
            elif sq == self.selected_sq:
                self.selected_sq = None
                self.legal_targets = []
            else:
                self._select(sq)
        else:
            self._select(sq)

    def _select(self, sq):
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.board.turn:
            self.selected_sq = sq
            self.legal_targets = [m.to_square for m in self.board.legal_moves
                                  if m.from_square == sq]
        else:
            self.selected_sq = None
            self.legal_targets = []

    def _update_position_eval(self):
        """Quick V(s) from the value head (no MCTS)."""
        if self.board.is_game_over():
            return
        encoded = encode_board(self.board)
        st = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, v = self.net(st)
        self.position_eval = v.item()

    def make_move(self, move):
        self.move_log.append(self.board.san(move))
        self.board.push(move)
        self.last_move = move
        self.selected_sq = None
        self.legal_targets = []

        if self.board.is_game_over():
            self._end_game()
        else:
            self._update_position_eval()
            self.status = "AI thinking..."
            self.ai_thinking = True
            threading.Thread(target=self._ai_turn, daemon=True).start()

    def _ai_turn(self):
        """Run full MCTS search and pick the best move."""
        policy, root_value = mcts_search(
            self.board, self.net,
            num_simulations=self.num_sims,
            temperature=0.0,          # greedy for play
            device=self.device,
        )
        self.ai_eval = root_value

        action = select_action(policy, temperature=0.0)
        move = index_to_move(action, self.board)
        if move not in self.board.legal_moves:
            move = list(self.board.legal_moves)[0]

        self.move_log.append(self.board.san(move))
        self.board.push(move)
        self.last_move = move
        self.ai_thinking = False

        if self.board.is_game_over():
            self._end_game()
        else:
            self._update_position_eval()
            self.status = "Your move"

    def _end_game(self):
        self.game_over = True
        result = self.board.result()
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            you_won = (winner == "White") == self.human_is_white
            self.status = f"{'You win!' if you_won else 'AI wins.'} {result}"
        elif self.board.is_stalemate():
            self.status = "Stalemate — Draw"
        elif self.board.is_insufficient_material():
            self.status = "Draw — Insufficient material"
        else:
            self.status = f"Game over: {result}"

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self):
        if not self.is_human_turn():
            self.status = "AI thinking..."
            self.ai_thinking = True
            threading.Thread(target=self._ai_turn, daemon=True).start()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.board = chess.Board()
                        self.selected_sq = None
                        self.legal_targets = []
                        self.last_move = None
                        self.move_log = []
                        self.ai_eval = None
                        self.position_eval = None
                        self.game_over = False
                        self.promotion_pending = None
                        self.status = ("Your move" if self.is_human_turn()
                                       else "AI thinking...")
                        if not self.is_human_turn():
                            self.ai_thinking = True
                            threading.Thread(target=self._ai_turn,
                                             daemon=True).start()
            self.draw()
            self.clock.tick(30)
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Play against the AlphaZero MCTS chess AI (GUI)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect latest)")
    parser.add_argument("--color", type=str, default="white",
                        choices=["white", "black"], help="Your color")
    parser.add_argument("--sims", type=int, default=800,
                        help="MCTS simulations per AI move (default: 800)")
    parser.add_argument("--device", type=str, default=Config.device)
    args = parser.parse_args()

    device = args.device
    human_is_white = args.color.lower() == "white"

    # Resolve checkpoint
    ckpt = args.checkpoint or find_latest_checkpoint()

    print("Loading AlphaZero MCTS model...")
    net = AlphaZeroNet().to(device)
    if ckpt and os.path.isfile(ckpt):
        load_checkpoint(net, ckpt, device=device)
        print(f"  Loaded: {ckpt}")
    else:
        print(f"  No checkpoint found (looked in {Config.checkpoint_dir}/)")
        print("  Playing with untrained network.")
    net.eval()

    print(f"  You are {'White' if human_is_white else 'Black'}")
    print(f"  MCTS sims: {args.sims}  |  R=reset  Q=quit\n")

    gui = PlayGUI(net, human_is_white, args.sims, device)
    gui.run()


if __name__ == "__main__":
    main()
