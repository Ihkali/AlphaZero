"""
tictactoe/game.py — Tic-Tac-Toe game environment.

Mirrors the chess.Board interface used by MCTS so the same search
algorithm can be tested on a trivially solvable game.
"""

import numpy as np


class TicTacToe:
    """Minimal board class with an API close to python-chess."""

    def __init__(self):
        # 0 = empty, 1 = X (player 1), -1 = O (player 2)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1        # X goes first
        self.move_stack: list[int] = []
        self._winner = None
        self._done = False

    # ── core interface ────────────────────────────────────────────────

    def copy(self):
        new = TicTacToe()
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.move_stack = list(self.move_stack)
        new._winner = self._winner
        new._done = self._done
        return new

    @property
    def legal_moves(self) -> list[int]:
        if self._done:
            return []
        return [i for i in range(9) if self.board[i] == 0]

    def push(self, action: int):
        assert self.board[action] == 0, f"Illegal move {action}"
        self.board[action] = self.current_player
        self.move_stack.append(action)
        self._check_game_over()
        self.current_player *= -1

    def pop(self):
        action = self.move_stack.pop()
        self.current_player *= -1
        self.board[action] = 0
        self._done = False
        self._winner = None

    def is_game_over(self) -> bool:
        return self._done

    def result(self) -> float:
        """Return +1 if X wins, -1 if O wins, 0 for draw/ongoing."""
        if self._winner is not None:
            return float(self._winner)
        return 0.0

    # ── encoding ──────────────────────────────────────────────────────

    def encode(self) -> np.ndarray:
        """Encode board as (3, 3, 3) tensor.

        Plane 0: current player's pieces
        Plane 1: opponent's pieces
        Plane 2: 1.0 if current player is X, else 0.0
        """
        planes = np.zeros((3, 3, 3), dtype=np.float32)
        b = self.board.reshape(3, 3)
        planes[0] = (b == self.current_player).astype(np.float32)
        planes[1] = (b == -self.current_player).astype(np.float32)
        if self.current_player == 1:
            planes[2] = 1.0
        return planes

    # ── display ───────────────────────────────────────────────────────

    def display(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            row = " ".join(symbols[self.board[r * 3 + c]] for c in range(3))
            rows.append(row)
        return "\n".join(rows)

    def __repr__(self):
        turn = "X" if self.current_player == 1 else "O"
        return f"TicTacToe(turn={turn}, moves={len(self.move_stack)})\n{self.display()}"

    # ── internal ──────────────────────────────────────────────────────

    _WIN_LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6),             # diags
    ]

    def _check_game_over(self):
        for a, b, c in self._WIN_LINES:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                self._winner = self.board[a]
                self._done = True
                return
        if not any(self.board[i] == 0 for i in range(9)):
            self._done = True
            self._winner = None
