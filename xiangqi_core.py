import re
from collections import Counter, defaultdict


START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
FILES = "abcdefghi"
RESULT_TOKENS = {"*", "1-0", "0-1", "1/2-1/2"}

PIECE_NAMES = {
    "K": "Red King",
    "A": "Red Advisor",
    "B": "Red Elephant",
    "N": "Red Horse",
    "R": "Red Chariot",
    "C": "Red Cannon",
    "P": "Red Pawn",
    "k": "Black King",
    "a": "Black Advisor",
    "b": "Black Elephant",
    "n": "Black Horse",
    "r": "Black Chariot",
    "c": "Black Cannon",
    "p": "Black Pawn",
}

CHINESE_PIECE_NAMES = {
    "K": "帅",
    "A": "仕",
    "B": "相",
    "N": "马",
    "R": "车",
    "C": "炮",
    "P": "兵",
    "k": "将",
    "a": "士",
    "b": "象",
    "n": "马",
    "r": "车",
    "c": "炮",
    "p": "卒",
}

RED_NUMERALS = "一二三四五六七八九"
BLACK_NUMERALS = "１２３４５６７８９"

MAX_PIECES = {
    "K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5,
    "k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5,
}

ADVISOR_POINTS = {
    "red": {(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)},
    "black": {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)},
}

ELEPHANT_POINTS = {
    "red": {(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)},
    "black": {(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)},
}

WXF_TO_FEN = {
    "K": "K",
    "G": "K",
    "A": "A",
    "B": "B",
    "E": "B",
    "N": "N",
    "H": "N",
    "R": "R",
    "C": "C",
    "P": "P",
}


def opponent(side):
    return "black" if side == "red" else "red"


def side_to_turn(side):
    return "w" if side == "red" else "b"


def turn_to_side(turn):
    return "red" if turn == "w" else "black"


def piece_side(piece):
    if not piece:
        return None
    return "red" if piece.isupper() else "black"


def fen_to_board(fen):
    parts = fen.strip().split()
    placement = parts[0]
    side = turn_to_side(parts[1]) if len(parts) > 1 else "red"
    board = []
    for row_text in placement.split("/"):
        row = []
        for ch in row_text:
            if ch.isdigit():
                row.extend([""] * int(ch))
            else:
                row.append(ch)
        if len(row) != 9:
            raise ValueError(f"Bad FEN row width: {row_text}")
        board.append(row)
    if len(board) != 10:
        raise ValueError("Bad FEN row count")
    return board, side


def board_to_fen(board, side="red"):
    rows = []
    for row in board:
        empty = 0
        text = []
        for piece in row:
            if piece:
                if empty:
                    text.append(str(empty))
                    empty = 0
                text.append(piece)
            else:
                empty += 1
        if empty:
            text.append(str(empty))
        rows.append("".join(text))
    return f"{'/'.join(rows)} {side_to_turn(side)} - - 0 1"


def clone_board(board):
    return [row[:] for row in board]


def in_bounds(row, col):
    return 0 <= row < 10 and 0 <= col < 9


def uci_to_coords(move):
    move = normalize_uci(move)
    src_col = FILES.index(move[0])
    src_row = 9 - int(move[1])
    dst_col = FILES.index(move[2])
    dst_row = 9 - int(move[3])
    return src_row, src_col, dst_row, dst_col


def coords_to_uci(src_row, src_col, dst_row, dst_col):
    return f"{FILES[src_col]}{9 - src_row}{FILES[dst_col]}{9 - dst_row}"


def chinese_file_number(side, col):
    index = 8 - col if side == "red" else col
    numerals = RED_NUMERALS if side == "red" else BLACK_NUMERALS
    return numerals[index]


def chinese_step_number(side, steps):
    numerals = RED_NUMERALS if side == "red" else BLACK_NUMERALS
    return numerals[max(0, min(8, steps - 1))]


def chinese_piece_label(board, piece, side, src_row, src_col):
    name = CHINESE_PIECE_NAMES.get(piece, piece)
    same_file_rows = [
        row
        for row in range(10)
        if board[row][src_col] == piece
    ]
    if len(same_file_rows) < 2:
        return f"{name}{chinese_file_number(side, src_col)}"

    front_to_back = sorted(same_file_rows, reverse=(side == "black"))
    index = front_to_back.index(src_row)
    if len(front_to_back) == 2:
        prefix = "前" if index == 0 else "后"
    elif len(front_to_back) == 3:
        prefix = ("前", "中", "后")[index]
    else:
        middle_labels = RED_NUMERALS if side == "red" else BLACK_NUMERALS
        labels = ["前"] + list(middle_labels[1:len(front_to_back) - 1]) + ["后"]
        prefix = labels[index]
    return f"{prefix}{name}"


def move_to_chinese(board, move):
    src_row, src_col, dst_row, dst_col = uci_to_coords(move)
    piece = board[src_row][src_col]
    if not piece:
        return normalize_uci(move)

    side = piece_side(piece)
    label = chinese_piece_label(board, piece, side, src_row, src_col)
    forward = dst_row < src_row if side == "red" else dst_row > src_row

    if dst_row == src_row:
        action = "平"
        target = chinese_file_number(side, dst_col)
    elif dst_col == src_col:
        action = "进" if forward else "退"
        target = chinese_step_number(side, abs(dst_row - src_row))
    else:
        action = "进" if forward else "退"
        target = chinese_file_number(side, dst_col)

    return f"{label}{action}{target}"


def move_line_to_chinese(board, moves):
    display_moves = []
    current_board = clone_board(board)
    current_side = piece_side(current_board[uci_to_coords(moves[0])[0]][uci_to_coords(moves[0])[1]]) if moves else None
    for move in moves:
        try:
            display_moves.append(move_to_chinese(current_board, move))
            side = piece_side(current_board[uci_to_coords(move)[0]][uci_to_coords(move)[1]]) or current_side
            current_board = apply_uci(current_board, move)
            current_side = opponent(side) if side else None
        except Exception:
            display_moves.append(move)
    return display_moves


def normalize_uci(token):
    token = token.strip()
    token = token.rstrip("+#?!")
    match = re.fullmatch(r"([a-iA-I])([0-9])[-x:]?([a-iA-I])([0-9])", token)
    if not match:
        raise ValueError(f"Not an ICCS coordinate move: {token}")
    return f"{match.group(1).lower()}{match.group(2)}{match.group(3).lower()}{match.group(4)}"


def is_coordinate_move(token):
    try:
        normalize_uci(token)
        return True
    except ValueError:
        return False


def palace_contains(side, row, col):
    if col < 3 or col > 5:
        return False
    return 7 <= row <= 9 if side == "red" else 0 <= row <= 2


def can_land(board, side, row, col):
    return in_bounds(row, col) and piece_side(board[row][col]) != side


def ray_moves(board, side, row, col):
    moves = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        while in_bounds(rr, cc):
            target_side = piece_side(board[rr][cc])
            if target_side == side:
                break
            moves.append((rr, cc))
            if target_side == opponent(side):
                break
            rr += dr
            cc += dc
    return moves


def cannon_moves(board, side, row, col):
    moves = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        while in_bounds(rr, cc) and not board[rr][cc]:
            moves.append((rr, cc))
            rr += dr
            cc += dc

        rr += dr
        cc += dc
        while in_bounds(rr, cc):
            if board[rr][cc]:
                if piece_side(board[rr][cc]) == opponent(side):
                    moves.append((rr, cc))
                break
            rr += dr
            cc += dc
    return moves


def horse_moves(board, side, row, col):
    moves = []
    steps = (
        (-2, -1, -1, 0),
        (-2, 1, -1, 0),
        (2, -1, 1, 0),
        (2, 1, 1, 0),
        (-1, -2, 0, -1),
        (1, -2, 0, -1),
        (-1, 2, 0, 1),
        (1, 2, 0, 1),
    )
    for dr, dc, leg_r, leg_c in steps:
        rr = row + dr
        cc = col + dc
        if in_bounds(rr, cc) and not board[row + leg_r][col + leg_c] and can_land(board, side, rr, cc):
            moves.append((rr, cc))
    return moves


def elephant_moves(board, side, row, col):
    moves = []
    for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
        rr = row + dr
        cc = col + dc
        eye_r = row + dr // 2
        eye_c = col + dc // 2
        if not in_bounds(rr, cc) or board[eye_r][eye_c]:
            continue
        if side == "red" and rr < 5:
            continue
        if side == "black" and rr > 4:
            continue
        if can_land(board, side, rr, cc):
            moves.append((rr, cc))
    return moves


def advisor_moves(board, side, row, col):
    moves = []
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = row + dr
        cc = col + dc
        if palace_contains(side, rr, cc) and can_land(board, side, rr, cc):
            moves.append((rr, cc))
    return moves


def king_moves(board, side, row, col):
    moves = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        if palace_contains(side, rr, cc) and can_land(board, side, rr, cc):
            moves.append((rr, cc))

    rr = row + (-1 if side == "red" else 1)
    while in_bounds(rr, col):
        if board[rr][col]:
            if board[rr][col].upper() == "K" and piece_side(board[rr][col]) == opponent(side):
                moves.append((rr, col))
            break
        rr += -1 if side == "red" else 1
    return moves


def pawn_moves(board, side, row, col):
    moves = []
    forward = -1 if side == "red" else 1
    candidates = [(row + forward, col)]
    crossed_river = row <= 4 if side == "red" else row >= 5
    if crossed_river:
        candidates.extend(((row, col - 1), (row, col + 1)))
    for rr, cc in candidates:
        if can_land(board, side, rr, cc):
            moves.append((rr, cc))
    return moves


def pseudo_destinations(board, row, col):
    piece = board[row][col]
    side = piece_side(piece)
    if not side:
        return []
    kind = piece.upper()
    if kind == "R":
        return ray_moves(board, side, row, col)
    if kind == "C":
        return cannon_moves(board, side, row, col)
    if kind == "N":
        return horse_moves(board, side, row, col)
    if kind == "B":
        return elephant_moves(board, side, row, col)
    if kind == "A":
        return advisor_moves(board, side, row, col)
    if kind == "K":
        return king_moves(board, side, row, col)
    if kind == "P":
        return pawn_moves(board, side, row, col)
    return []


def apply_uci(board, move):
    src_row, src_col, dst_row, dst_col = uci_to_coords(move)
    out = clone_board(board)
    piece = out[src_row][src_col]
    if not piece:
        raise ValueError(f"No piece on source square: {move[:2]}")
    out[src_row][src_col] = ""
    out[dst_row][dst_col] = piece
    return out


def find_king(board, side):
    target = "K" if side == "red" else "k"
    for row in range(10):
        for col in range(9):
            if board[row][col] == target:
                return row, col
    return None


def is_in_check(board, side):
    king = find_king(board, side)
    if king is None:
        return True
    enemy = opponent(side)
    for row in range(10):
        for col in range(9):
            if piece_side(board[row][col]) != enemy:
                continue
            if king in pseudo_destinations(board, row, col):
                return True
    return False


def validate_fen(fen):
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("FEN is empty")

    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError("FEN must include side to move")
    if parts[1] not in {"w", "b"}:
        raise ValueError(f"Bad side to move: {parts[1]}")

    placement = parts[0]
    rows = placement.split("/")
    if len(rows) != 10:
        raise ValueError("Bad FEN row count")

    board = []
    counts = Counter()
    valid_pieces = set(PIECE_NAMES)
    for row_text in rows:
        row = []
        previous_digit = False
        for ch in row_text:
            if ch in "123456789":
                if previous_digit:
                    raise ValueError(f"Consecutive empty counts in row: {row_text}")
                row.extend([""] * int(ch))
                previous_digit = True
            elif ch.isdigit():
                raise ValueError(f"Bad empty count in row: {row_text}")
            elif ch in valid_pieces:
                row.append(ch)
                counts[ch] += 1
                previous_digit = False
            else:
                raise ValueError(f"Bad piece in FEN: {ch}")
        if len(row) != 9:
            raise ValueError(f"Bad FEN row width: {row_text}")
        board.append(row)

    if counts["K"] != 1:
        raise ValueError(f"Expected exactly one red king, found {counts['K']}")
    if counts["k"] != 1:
        raise ValueError(f"Expected exactly one black king, found {counts['k']}")
    for piece, cap in MAX_PIECES.items():
        if counts[piece] > cap:
            raise ValueError(f"Too many {PIECE_NAMES[piece]} pieces: {counts[piece]}/{cap}")

    for row_index, row in enumerate(board):
        for col_index, piece in enumerate(row):
            if not piece:
                continue
            side = piece_side(piece)
            kind = piece.upper()
            pos = (row_index, col_index)
            if kind == "K" and not palace_contains(side, row_index, col_index):
                raise ValueError(f"{PIECE_NAMES[piece]} outside palace at row {row_index}, col {col_index}")
            if kind == "A" and pos not in ADVISOR_POINTS[side]:
                raise ValueError(f"{PIECE_NAMES[piece]} on illegal square at row {row_index}, col {col_index}")
            if kind == "B" and pos not in ELEPHANT_POINTS[side]:
                raise ValueError(f"{PIECE_NAMES[piece]} on illegal square at row {row_index}, col {col_index}")
            if kind == "P":
                if side == "red" and row_index > 6:
                    raise ValueError(f"{PIECE_NAMES[piece]} behind starting rank at row {row_index}, col {col_index}")
                if side == "black" and row_index < 3:
                    raise ValueError(f"{PIECE_NAMES[piece]} behind starting rank at row {row_index}, col {col_index}")

    red_king = find_king(board, "red")
    black_king = find_king(board, "black")
    if red_king[1] == black_king[1]:
        col = red_king[1]
        start = min(red_king[0], black_king[0]) + 1
        end = max(red_king[0], black_king[0])
        if all(not board[row][col] for row in range(start, end)):
            raise ValueError("Kings face each other")

    if is_in_check(board, "red") and is_in_check(board, "black"):
        raise ValueError("Both kings are in check")

    return board, turn_to_side(parts[1])


def legal_moves_for_piece(board, row, col, enforce_check=True):
    side = piece_side(board[row][col])
    if not side:
        return []
    moves = []
    for dst_row, dst_col in pseudo_destinations(board, row, col):
        uci = coords_to_uci(row, col, dst_row, dst_col)
        if enforce_check and is_in_check(apply_uci(board, uci), side):
            continue
        moves.append(uci)
    return moves


def legal_moves(board, side, enforce_check=True):
    moves = []
    for row in range(10):
        for col in range(9):
            if piece_side(board[row][col]) == side:
                moves.extend(legal_moves_for_piece(board, row, col, enforce_check=enforce_check))
    return sorted(moves)


def is_legal_move(board, side, move):
    move = normalize_uci(move)
    src_row, src_col, _, _ = uci_to_coords(move)
    if piece_side(board[src_row][src_col]) != side:
        return False
    return move in legal_moves_for_piece(board, src_row, src_col)


def grid_changes(old_board, new_board):
    disappeared, appeared = [], []
    for r in range(10):
        for c in range(9):
            o, n = old_board[r][c], new_board[r][c]
            if o == n:
                continue
            if o:
                disappeared.append((r, c, o))
            if n:
                appeared.append((r, c, n))
    return disappeared, appeared


def detect_move_candidates_from_grids(old_board, new_board):
    disappeared, appeared = grid_changes(old_board, new_board)

    candidates = []
    for dr, dc, moved_piece in appeared:
        for sr, sc, source_piece in disappeared:
            if source_piece != moved_piece or (sr, sc) == (dr, dc):
                continue
            target_piece = old_board[dr][dc]
            if target_piece and piece_side(target_piece) == piece_side(moved_piece):
                continue
            if (dr, dc) in pseudo_destinations(old_board, sr, sc):
                candidates.append((moved_piece, sr, sc, dr, dc))
    return candidates


def detect_move_from_grids(old_board, new_board):
    """
    Compare two 10x9 grids and return (piece, sr, sc, dr, dc) if exactly one
    pseudo-legal move separates them.  Raises ValueError otherwise.

    Ghost pieces — recognition flickers that make a piece appear or vanish without a
    corresponding real board event — are tolerated:
      - Ghost disappearance: a disappeared source that cannot legally reach the
        moved piece's destination.
      - Ghost appearance: an appeared piece that no disappeared source can
        legally reach.
    """
    disappeared, appeared = grid_changes(old_board, new_board)
    candidates = detect_move_candidates_from_grids(old_board, new_board)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Cannot find legal move candidate "
            f"from {len(disappeared)} disappeared and {len(appeared)} appeared cells"
        )
    raise ValueError(f"Cannot find unique move candidate: {candidates}")


def apply_uci_to_fen(fen, move):
    board, side = fen_to_board(fen)
    move = normalize_uci(move)
    if not is_legal_move(board, side, move):
        raise ValueError(f"Illegal move for {side}: {move}")
    return board_to_fen(apply_uci(board, move), opponent(side))


def strip_pgn_tokens(text):
    text = re.sub(r"(?m)^\s*\[[^\]]*\]\s*$", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text, flags=re.S)
    while re.search(r"\([^()]*\)", text, flags=re.S):
        text = re.sub(r"\([^()]*\)", " ", text, flags=re.S)
    text = re.sub(r";[^\n\r]*", " ", text)
    tokens = []
    for raw in re.split(r"[\s,]+", text):
        token = raw.strip()
        if not token:
            continue
        token = re.sub(r"^\d+\.(?:\.\.)?", "", token)
        token = token.strip()
        if not token or token in RESULT_TOKENS:
            continue
        token = token.rstrip("!?+#")
        if token and token not in RESULT_TOKENS:
            tokens.append(token)
    return tokens


def file_to_col(file_number, side):
    n = int(file_number)
    if n < 1 or n > 9:
        raise ValueError(f"Bad Xiangqi file number: {file_number}")
    return 9 - n if side == "red" else n - 1


def forward_delta(side, op, amount):
    forward = -1 if side == "red" else 1
    return forward * amount if op == "+" else -forward * amount


def wxf_piece_letter(piece_code, side):
    fen_piece = WXF_TO_FEN.get(piece_code.upper())
    if not fen_piece:
        raise ValueError(f"Unsupported piece code: {piece_code}")
    return fen_piece if side == "red" else fen_piece.lower()


def wxf_to_uci(token, board, side):
    token = token.strip().upper().replace(".", "=")
    match = re.fullmatch(r"([KGAEBNHRCP])([1-9])([=+\-])([1-9])", token)
    if not match:
        raise ValueError(f"Unsupported ICCS/WXF move token: {token}")

    piece_code, source_file, op, target_value = match.groups()
    piece = wxf_piece_letter(piece_code, side)
    source_col = file_to_col(source_file, side)
    candidates = [
        (row, source_col)
        for row in range(10)
        if board[row][source_col] == piece
    ]
    matches = []
    for row, col in candidates:
        for move in legal_moves_for_piece(board, row, col):
            _, _, dst_row, dst_col = uci_to_coords(move)
            if wxf_move_matches(piece.upper(), side, row, col, dst_row, dst_col, op, target_value):
                matches.append(move)

    if len(matches) != 1:
        raise ValueError(f"Ambiguous or illegal ICCS/WXF move {token}: {matches}")
    return matches[0]


def wxf_move_matches(kind, side, src_row, src_col, dst_row, dst_col, op, target_value):
    if op == "=":
        return dst_row == src_row and dst_col == file_to_col(target_value, side)

    if kind in {"R", "C", "K", "P"}:
        distance = int(target_value)
        return dst_col == src_col and dst_row == src_row + forward_delta(side, op, distance)

    target_col = file_to_col(target_value, side)
    if dst_col != target_col:
        return False
    return dst_row < src_row if (side == "red") == (op == "+") else dst_row > src_row


def parse_pgn(text, start_fen=START_FEN):
    board, side = fen_to_board(start_fen)
    moves = []
    fens = [board_to_fen(board, side)]

    for token in strip_pgn_tokens(text):
        if is_coordinate_move(token):
            uci = normalize_uci(token)
        else:
            uci = wxf_to_uci(token, board, side)
        if not is_legal_move(board, side, uci):
            raise ValueError(f"Illegal move {token} ({uci}) for {side}")
        board = apply_uci(board, uci)
        side = opponent(side)
        fen = board_to_fen(board, side)
        moves.append({
            "ply": len(moves) + 1,
            "side": opponent(side),
            "notation": token,
            "uci": uci,
            "fen": fen,
        })
        fens.append(fen)

    return {
        "startFen": start_fen,
        "fen": board_to_fen(board, side),
        "side": side,
        "moves": moves,
        "fens": fens,
    }


def legal_moves_by_source(board, side):
    grouped = defaultdict(list)
    for move in legal_moves(board, side):
        grouped[move[:2]].append(move)
    return dict(grouped)
