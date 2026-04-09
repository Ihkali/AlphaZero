"""
SL/encode.py — Board ↔ tensor encoding and Move ↔ index mapping.

Board encoding : chess.Board  →  np.ndarray  (119, 8, 8)
Move  encoding : chess.Move   →  int  (0 .. 4671)
Move  decoding : int          →  chess.Move
Legal mask     : chess.Board  →  np.ndarray  (4672,)

Identical to MCTS/encode.py but self-contained so SL/ has no cross-deps.
"""

import chess
import numpy as np
from SL.config import Config

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def encode_board(board: chess.Board) -> np.ndarray:
    T = Config.history_length
    planes = np.zeros((Config.input_planes, 8, 8), dtype=np.float32)
    us = board.turn

    temp = board.copy()
    history_boards: list[chess.Board | None] = [temp.copy()]
    for _ in range(T - 1):
        if temp.move_stack:
            temp.pop()
            history_boards.append(temp.copy())
        else:
            history_boards.append(None)

    for t, hist in enumerate(history_boards):
        if hist is None:
            continue
        base = t * 14
        for i, pt in enumerate(PIECE_TYPES):
            for sq in hist.pieces(pt, us):
                r, c = _square_to_rc(sq, us)
                planes[base + i, r, c] = 1.0
            for sq in hist.pieces(pt, not us):
                r, c = _square_to_rc(sq, us)
                planes[base + 6 + i, r, c] = 1.0
        if hist.is_repetition(2):
            planes[base + 12, :, :] = 1.0
        if hist.is_repetition(3):
            planes[base + 13, :, :] = 1.0

    if board.turn == chess.WHITE:
        planes[112, :, :] = 1.0
    planes[113, :, :] = board.fullmove_number / 500.0
    if us == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[114, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[115, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[116, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[117, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[114, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[115, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[116, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[117, :, :] = 1.0
    planes[118, :, :] = board.halfmove_clock / 100.0
    return planes


def _square_to_rc(sq: int, perspective: bool) -> tuple[int, int]:
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    if perspective == chess.WHITE:
        return rank, file
    else:
        return 7 - rank, 7 - file


# ═══════════════════════════════════════════════════════════════════════════
#  MOVE ENCODING
# ═══════════════════════════════════════════════════════════════════════════

QUEEN_DIRS = [
    ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
    (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
]

KNIGHT_MOVES = [
    ( 2,  1), ( 2, -1), (-2,  1), (-2, -1),
    ( 1,  2), ( 1, -2), (-1,  2), (-1, -2),
]

UNDER_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDER_PROMO_DIRS = [-1, 0, 1]


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    us = board.turn
    from_sq = move.from_square
    to_sq = move.to_square
    from_r, from_c = _square_to_rc(from_sq, us)
    to_r, to_c = _square_to_rc(to_sq, us)
    dr = to_r - from_r
    dc = to_c - from_c
    src_idx = from_r * 8 + from_c
    move_type = _classify_move(dr, dc, move.promotion)
    return src_idx * 73 + move_type


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    us = board.turn
    src_idx, move_type = divmod(index, 73)
    from_r, from_c = divmod(src_idx, 8)
    dr, dc, promotion = _decode_move_type(move_type)
    to_r = from_r + dr
    to_c = from_c + dc
    from_sq = _rc_to_square(from_r, from_c, us)
    to_sq = _rc_to_square(to_r, to_c, us)
    return chess.Move(from_sq, to_sq, promotion=promotion)


def _rc_to_square(row: int, col: int, perspective: bool) -> int:
    if perspective == chess.WHITE:
        return chess.square(col, row)
    else:
        return chess.square(7 - col, 7 - row)


def _classify_move(dr: int, dc: int, promotion) -> int:
    if promotion is not None and promotion != chess.QUEEN:
        piece_idx = UNDER_PROMO_PIECES.index(promotion)
        dir_idx = UNDER_PROMO_DIRS.index(dc)
        return 64 + piece_idx * 3 + dir_idx
    if (dr, dc) in KNIGHT_MOVES:
        return 56 + KNIGHT_MOVES.index((dr, dc))
    dist, direction = _decompose_queen_move(dr, dc)
    return direction * 7 + (dist - 1)


def _decompose_queen_move(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0:
        udr, udc = 0, (1 if dc > 0 else -1)
    elif dc == 0:
        udr, udc = (1 if dr > 0 else -1), 0
    else:
        udr = 1 if dr > 0 else -1
        udc = 1 if dc > 0 else -1
    dist = max(abs(dr), abs(dc))
    direction = QUEEN_DIRS.index((udr, udc))
    return dist, direction


def _decode_move_type(move_type: int) -> tuple[int, int, int | None]:
    if move_type < 56:
        direction, dist_minus_1 = divmod(move_type, 7)
        dist = dist_minus_1 + 1
        udr, udc = QUEEN_DIRS[direction]
        return udr * dist, udc * dist, None
    elif move_type < 64:
        idx = move_type - 56
        dr, dc = KNIGHT_MOVES[idx]
        return dr, dc, None
    else:
        idx = move_type - 64
        piece_idx, dir_idx = divmod(idx, 3)
        promotion = UNDER_PROMO_PIECES[piece_idx]
        dc = UNDER_PROMO_DIRS[dir_idx]
        dr = 1
        return dr, dc, promotion


def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(Config.policy_size, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        mask[idx] = 1.0
    return mask


def get_legal_move_indices(board: chess.Board) -> list[tuple[chess.Move, int]]:
    result = []
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        result.append((move, idx))
    return result
