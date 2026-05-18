import os
import pathlib
import sys
import time
import signal
import sqlite3
import subprocess
import threading
import random

import cv2

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

try:
    import msvcrt
except ImportError:
    msvcrt = None


def _hide_dock_icon():
    """Hide the Python dock icon on macOS.

    pynput / screen-capture libs make Python register as a regular GUI app,
    which surfaces a Python icon in the Dock. Switching the activation
    policy to "accessory" keeps the process headless. Must run BEFORE
    importing pynput / any library that touches the window server.
    """
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )
    except Exception:
        pass


from vision import Vision
from engine import Engine
from calibrate import capture_screen
from calibrate import calibration_exists
from calibrate import load_config
from calibrate import NEXT_GAME_BUTTON_TEMPLATE_NAME
from calibrate import run_next_game_recognition
from calibrate import run_calibration as run_calibration_flow
from calibrate import update_config
from app_paths import candidate_paths, resolve_resource_path
from xiangqi_core import (
    START_FEN,
    apply_uci_to_fen,
    apply_uci,
    board_to_fen,
    coords_to_uci,
    fen_to_board,
    is_in_check,
    is_legal_move,
    legal_moves,
    legal_moves_for_piece,
    move_to_chinese,
    opponent,
    parse_pgn,
    piece_side,
    validate_fen,
    uci_to_coords,
)


BOOK_PRIORITY_SCORE = "score"
BOOK_PRIORITY_WIN_RATE = "win_rate"


def normalize_book_priority(value):
    text = str(value or "").strip().lower()
    if text in {"win_rate", "winrate", "win", "胜率", "胜率优先"}:
        return BOOK_PRIORITY_WIN_RATE
    return BOOK_PRIORITY_SCORE


def book_priority_label(value):
    return "胜率优先" if normalize_book_priority(value) == BOOK_PRIORITY_WIN_RATE else "分数优先"


def find_opening_book_path(config=None):
    config = config or {}
    configured = config.get("opening_book_path")
    if configured and os.path.exists(configured):
        return configured
    for path in candidate_paths("book.xqb"):
        if os.path.exists(path):
            return path
    return None


def xqbook_fen_to_key(fen):
    placement, side = fen.strip().split()[:2]
    rows = placement.count("/") + 1
    first_row = placement.split("/", 1)[0]
    cols = sum(int(ch) if ch.isdigit() else 1 for ch in first_row)
    size = rows * cols
    turn_is_red = side != "b"
    piece_codes = {
        "X": 0, "x": 0,
        "R": 1, "N": 2, "B": 3, "A": 4, "K": 5, "C": 6, "P": 7,
        "r": 9, "n": 10, "b": 11, "a": 12, "k": 13, "c": 14, "p": 15,
    }
    cells = [-1] * size
    index = 0
    for ch in placement:
        if ch == "/":
            continue
        if ch.isdigit():
            index += int(ch)
            continue
        lookup = ch if turn_is_red else ch.swapcase()
        cells[index] = piece_codes.get(lookup, -1)
        index += 1

    mirror_ud = False
    if not turn_is_red:
        for row in range(rows // 2):
            for col in range(cols):
                a = row * cols + col
                b = (rows - 1 - row) * cols + (cols - 1 - col)
                cells[a], cells[b] = cells[b], cells[a]
        mirror_ud = True

    mirror_lr = False
    lr_done = False
    for row in range(rows):
        if lr_done:
            break
        for col in range(cols // 2):
            a = row * cols + col
            b = row * cols + (cols - 1 - col)
            if cells[a] != cells[b]:
                mirror_lr = cells[b] > cells[a]
                lr_done = True
                break
    if mirror_lr:
        for row in range(rows):
            for col in range(cols // 2):
                a = row * cols + col
                b = row * cols + (cols - 1 - col)
                cells[a], cells[b] = cells[b], cells[a]

    out = bytearray()
    buffer = 0
    bits = 0
    buffer_bits = 32
    code_bits = 4
    for idx, value in enumerate(cells):
        if value == -1:
            bits += 1
        else:
            buffer |= 1 << (buffer_bits - bits - 1)
            buffer |= value << (buffer_bits - bits - 1 - code_bits)
            bits += 1 + code_bits
        next_value = cells[idx + 1] if idx + 1 < len(cells) else None
        need_bits = 1 if next_value == -1 else code_bits + 1
        if idx == len(cells) - 1 or buffer_bits - bits < need_bits:
            threshold = 1 if idx == len(cells) - 1 else 8
            while bits >= threshold:
                out.append((buffer >> (buffer_bits - 8)) & 0xFF)
                buffer = (buffer << 8) & 0xFFFFFFFF
                bits -= 8
    return bytes(out), mirror_ud, mirror_lr, rows, cols


def xqbook_mirror_move(move, mirror_ud, mirror_lr, rows, cols):
    from_row = move >> 12
    from_col = (move >> 8) & 0xF
    to_row = (move >> 4) & 0xF
    to_col = move & 0xF
    if mirror_ud:
        from_row = rows - 1 - from_row
        to_row = rows - 1 - to_row
        from_col = cols - 1 - from_col
        to_col = cols - 1 - to_col
    if mirror_lr:
        from_col = cols - 1 - from_col
        to_col = cols - 1 - to_col
    return from_row, from_col, to_row, to_col


def book_number(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def book_win_rate(item):
    win = book_number(item.get("win"))
    draw = book_number(item.get("draw"))
    lost = book_number(item.get("lost"))
    total = max(1, win + draw + lost)
    return win / total


def sort_xqbook_moves(moves, priority=BOOK_PRIORITY_SCORE):
    priority = normalize_book_priority(priority)

    def key(item):
        score = book_number(item.get("score"))
        win = book_number(item.get("win"))
        item_id = book_number(item.get("id"))
        win_rate = book_win_rate(item)
        if priority == BOOK_PRIORITY_WIN_RATE:
            return (win_rate, score, win, -item_id)
        return (score, win_rate, win, -item_id)

    return sorted(moves, key=key, reverse=True)


def query_xqbook_file(path, fen, priority=BOOK_PRIORITY_SCORE):
    if not path or not fen or not os.path.exists(path):
        return []
    results = []
    try:
        key, mirror_ud, mirror_lr, rows, cols = xqbook_fen_to_key(fen)
        uri = pathlib.Path(path).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except Exception:
        return []
    try:
        query = "select Id,Move,Score,Win,Draw,Lost,Valid,Memo from book where key=?"
        for row in conn.execute(query, (sqlite3.Binary(key),)):
            item_id, raw_move, score, win, draw, lost, valid, memo = row
            src_row, src_col, dst_row, dst_col = xqbook_mirror_move(
                int(raw_move), mirror_ud, mirror_lr, rows, cols
            )
            results.append({
                "id": item_id,
                "uci": coords_to_uci(src_row, src_col, dst_row, dst_col),
                "score": score,
                "win": win,
                "draw": draw,
                "lost": lost,
                "valid": valid,
                "memo": memo or "",
            })
    finally:
        conn.close()
    return sort_xqbook_moves(results, priority)


def parse_engine_score(score):
    parts = str(score or "").split()
    if len(parts) < 2:
        return None
    kind, value = parts[0], parts[1]
    if kind not in {"cp", "mate"}:
        return None
    try:
        return kind, int(value)
    except (TypeError, ValueError):
        return None


def fen_red_to_move(fen):
    try:
        _board, side = fen_to_board(fen)
        return side == "red"
    except Exception:
        return True


def java_oriented_engine_score(score, fen=None, is_reverse=False):
    parsed = parse_engine_score(score)
    if parsed is None:
        return None
    kind, value = parsed
    red_go = fen_red_to_move(fen)
    if (red_go and is_reverse) or ((not red_go) and (not is_reverse)):
        value = -value
    mate = value if kind == "mate" else None
    chart_score = ((-30000 if value < 0 else 30000) - value) if mate is not None else value
    return {
        "kind": kind,
        "score": value,
        "mate": mate,
        "chart_score": chart_score,
    }


def clamp_score_chart_value(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0
    return max(-1000, min(1000, score))


def java_think_title(depth, pv, score_info, nps=0, time_ms=0):
    depth_text = depth if depth else "-"
    if score_info is None:
        label = "分数"
        score_text = "-"
    elif score_info["kind"] == "mate":
        label = "绝杀"
        score_text = f"{score_info['score']}步"
    else:
        label = "分数"
        score_text = str(score_info["score"])
    try:
        nps_k = int(int(nps or 0) / 1000)
    except (TypeError, ValueError):
        nps_k = 0
    try:
        seconds = int(time_ms or 0) / 1000.0
    except (TypeError, ValueError):
        seconds = 0.0
    return f"深度: {depth_text}  PV: {pv}  {label}: {score_text}  NPS: {nps_k}K  时间: {seconds:.1f}s"


class _MouseButtonFallback:
    left = "left"


Button = _MouseButtonFallback
MouseController = None


class _NoMouseController:
    position = None

    def press(self, button):
        raise RuntimeError("Mouse control is not available on this system.")

    def release(self, button):
        raise RuntimeError("Mouse control is not available on this system.")

    def click(self, button):
        raise RuntimeError("Mouse control is not available on this system.")


def _load_mouse_controls(hide_dock_icon=True):
    global Button, MouseController
    if MouseController is not None:
        return
    if hide_dock_icon:
        _hide_dock_icon()
    from pynput.mouse import Button as PynputButton
    from pynput.mouse import Controller as PynputMouseController
    Button = PynputButton
    MouseController = PynputMouseController


DEBUG = True
NEXT_GAME_SEARCH_RADIUS_DEFAULT = 220.0
NEXT_GAME_MATCH_THRESHOLD_DEFAULT = 0.88
NEXT_GAME_CLICK_COOLDOWN_DEFAULT = 5.0
NEXT_GAME_SCAN_INTERVAL_DEFAULT = 0.25
BOARD_STABLE_FRAMES_DEFAULT = 3


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def flip_move(bm):
    if len(bm) < 4:
        return bm
    sf = 8 - (ord(bm[0]) - ord('a'))
    sr = 9 - int(bm[1])
    ef = 8 - (ord(bm[2]) - ord('a'))
    er = 9 - int(bm[3])
    return f"{chr(sf + ord('a'))}{sr}{chr(ef + ord('a'))}{er}"


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _calibration_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--calibrate", "--no-prompt"]
    return [sys.executable, os.path.join(_app_dir(), "main.py"), "--calibrate", "--no-prompt"]


def _next_game_recognition_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--recognize-next", "--no-prompt"]
    return [sys.executable, os.path.join(_app_dir(), "main.py"), "--recognize-next", "--no-prompt"]


class App:
    def __init__(
        self,
        exit_on_error=True,
        hide_dock_icon=True,
        board_callback=None,
        move_callback=None,
        fen_callback=None,
        analysis_callback=None,
        record_callback=None,
        enable_mouse=True,
    ):
        self.board_callback = board_callback
        self.move_callback = move_callback
        self.fen_callback = fen_callback
        self.analysis_callback = analysis_callback
        self.record_callback = record_callback
        try:
            self.vis = Vision()
        except Exception as e:
            print(f"Vision Error: {e}")
            print("Please run `python calibrate.py` first!")
            if exit_on_error:
                sys.exit(1)
            raise

        try:
            self.eng = Engine()
        except Exception as e:
            print(f"Engine Error: {e}")
            if exit_on_error:
                sys.exit(1)
            raise

        self.last_seen_fen = None
        self.stable_count = 0
        self.last_status = None
        self.next_game_last_click_at = 0.0
        self.next_game_last_scan_at = 0.0
        self.waiting_for_new_game = False
        self.manual_result_hold_until = 0.0
        self.board_change_initialized = False
        self.board_change_baseline_key = None
        self.board_change_baseline_fen = None
        self.board_change_baseline_piece_count = 0
        self.board_change_invalid_key = None
        self.board_change_invalid_count = 0
        self.board_change_strict_baseline = False
        self.record_start_fen = None
        self.record_moves = []
        self.record_saved = False
        self._next_game_template = None
        self._next_game_template_path = None
        self._next_game_template_mtime = None
        self._next_game_missing_logged_at = 0.0
        self._recalc = threading.Event()
        self._reset_requested = threading.Event()
        self._reload_requested = threading.Event()
        self._manual_calculate_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._mouse_available = False
        self._mouse = _NoMouseController()
        if enable_mouse:
            try:
                _load_mouse_controls(hide_dock_icon=hide_dock_icon)
                self._mouse = MouseController()
                self._mouse_available = True
            except Exception as e:
                if self._config_bool("auto_move", default=True):
                    raise RuntimeError(
                        "Mouse control is unavailable. On Linux, run under an X11 session "
                        "or set auto_move to false in config.json."
                    ) from e
                print(f"[Mouse] unavailable; continuing without auto move: {e}")

    def request_reload(self):
        self._reload_requested.set()
        self._reset_requested.set()

    def request_calculate(self):
        self._manual_calculate_requested.set()
        self._reset_requested.set()
        try:
            self.eng.send_cmd("stop")
        except Exception:
            pass

    def stop(self):
        self._stop_requested.set()
        self._reset_requested.set()
        try:
            self.eng.send_cmd("stop")
        except Exception:
            pass

    def _should_cancel_search(self):
        return (
            self._reset_requested.is_set()
            or self._manual_calculate_requested.is_set()
            or self._stop_requested.is_set()
        )

    def _reload(self):
        self.vis = Vision()
        try:
            settings = self.eng.apply_runtime_config()
            if settings:
                dprint(
                    "[Reload] Engine settings applied: "
                    f"Threads={settings['threads']}, Hash={settings['hash_mb']} MB."
                )
        except Exception as e:
            dprint(f"[Reload] Engine config update failed: {e}")
        self._next_game_template = None
        self._next_game_template_path = None
        self._next_game_template_mtime = None
        self._reset_state()
        self._recalc.set()
        dprint("[Reload] Calibration and helper state reloaded.")

    def _reset_state(self):
        self.last_seen_fen = None
        self.stable_count = 0
        self.last_status = None
        self.waiting_for_new_game = False
        self.manual_result_hold_until = 0.0
        self.board_change_initialized = False
        self.board_change_baseline_key = None
        self.board_change_baseline_fen = None
        self.board_change_baseline_piece_count = 0
        self.board_change_invalid_key = None
        self.board_change_invalid_count = 0
        self.board_change_strict_baseline = False
        self.record_start_fen = None
        self.record_moves = []
        self.record_saved = False

    def _emit_move(self, move, label="PV"):
        if self.move_callback is not None:
            self.move_callback(move, label)

    def _emit_analysis(self, text):
        if self.analysis_callback is not None:
            self.analysis_callback(text)

    def _emit_record(self):
        if self.record_callback is not None:
            self.record_callback(self._record_history_text())

    def _format_engine_analysis(self, fen=None, flipped=False):
        lines = []
        pv_lines = getattr(self.eng, "pv_lines", {}) or {}
        pv_line = getattr(self.eng, "current_pv_line", []) or []
        if not pv_lines and pv_line:
            pv_lines = {1: pv_line}
        scores = getattr(self.eng, "pv_scores", {}) or {}
        depths = getattr(self.eng, "pv_depths", {}) or {}
        times = getattr(self.eng, "pv_times", {}) or {}
        nps_values = getattr(self.eng, "pv_nps", {}) or {}
        for rank in sorted(pv_lines)[:5]:
            moves = pv_lines.get(rank) or []
            if not moves:
                continue
            if not self._is_engine_move_legal_for_fen(fen, moves[0]):
                continue
            score = scores.get(rank, getattr(self.eng, "eval_score", "") if rank == 1 else "")
            score_info = java_oriented_engine_score(score, fen, flipped)
            depth = depths.get(rank, getattr(self.eng, "current_depth", 0) if rank == 1 else 0)
            time_ms = times.get(rank, getattr(self.eng, "current_time_ms", 0) if rank == 1 else 0)
            nps = nps_values.get(rank, getattr(self.eng, "current_nps", 0) if rank == 1 else 0)
            gui_moves = self._format_move_line(moves[:8], fen, flipped)
            body = "  ".join(gui_moves)
            if "null" in body:
                continue
            lines.append(java_think_title(depth, rank, score_info, nps, time_ms))
            if body:
                lines.append(body)

        if not lines:
            depth = getattr(self.eng, "current_depth", 0)
            score_info = java_oriented_engine_score(getattr(self.eng, "eval_score", ""), fen, flipped)
            if depth or score_info is not None:
                lines.append(java_think_title(depth, 1, score_info, getattr(self.eng, "current_nps", 0), getattr(self.eng, "current_time_ms", 0)))

        return "\n".join(lines)

    def _is_engine_move_legal_for_fen(self, fen, move):
        if not fen or not move:
            return False
        try:
            board, side = fen_to_board(fen)
            return is_legal_move(board, side, move)
        except Exception:
            return False

    def _repetition_position_key(self, fen):
        parts = fen.strip().split()
        if len(parts) < 2:
            return fen.strip()
        return f"{parts[0]} {parts[1]}"

    def _position_sequence(self):
        if not self.record_moves:
            return [], []
        start_fen = self.record_start_fen or self.record_moves[0].get("from_fen")
        if not start_fen:
            return [], []
        positions = [start_fen]
        moves = []
        for item in self.record_moves:
            moves.append(item)
            to_fen = item.get("to_fen")
            if to_fen:
                positions.append(to_fen)
        return positions, moves

    def _is_uncrossed_pawn(self, piece, row):
        if piece == "P":
            return row >= 5
        if piece == "p":
            return row <= 4
        return False

    def _capture_attacks_square(self, board, side, row, col):
        target = (row, col)
        attacks = []
        for move in legal_moves(board, side):
            src_row, src_col, dst_row, dst_col = uci_to_coords(move)
            if (dst_row, dst_col) == target and board[dst_row][dst_col]:
                attacks.append((src_row, src_col, move))
        return attacks

    def _capture_exception(self, board, side, move):
        src_row, src_col, dst_row, dst_col = uci_to_coords(move)
        attacker = board[src_row][src_col]
        target = board[dst_row][dst_col]
        if not attacker or not target:
            return True

        attacker_kind = attacker.upper()
        target_kind = target.upper()
        if attacker_kind in {"K", "P"}:
            return True
        if self._is_uncrossed_pawn(target, dst_row):
            return True

        try:
            after_capture = apply_uci(board, move)
            recaptured = bool(self._capture_attacks_square(after_capture, opponent(side), dst_row, dst_col))
        except Exception:
            recaptured = False
        if recaptured:
            if attacker_kind in {"N", "C"} and target_kind == "R":
                return False
            if attacker_kind in {"A", "B"} and target_kind in {"N", "C", "R"}:
                return False
            return True

        if attacker_kind == target_kind:
            reverse = coords_to_uci(dst_row, dst_col, src_row, src_col)
            try:
                if reverse in legal_moves_for_piece(board, dst_row, dst_col):
                    return True
            except Exception:
                pass

        return False

    def _chased_targets_after_move(self, fen, side):
        try:
            board, _ = fen_to_board(fen)
        except Exception:
            return set()
        targets = set()
        for move in legal_moves(board, side):
            src_row, src_col, dst_row, dst_col = uci_to_coords(move)
            target = board[dst_row][dst_col]
            if not target or piece_side(target) != opponent(side):
                continue
            if self._capture_exception(board, side, move):
                continue
            targets.add((dst_row, dst_col, target.upper()))
        return targets

    def _move_gives_check(self, from_fen, move):
        try:
            board, side = fen_to_board(from_fen)
            return is_in_check(apply_uci(board, move), opponent(side))
        except Exception:
            return False

    def _cycle_side_properties(self, positions, moves, start_index, side):
        cycle_moves = moves[start_index:]
        if not cycle_moves:
            return set()

        side_move_indices = [
            index for index, item in enumerate(cycle_moves, start_index)
            if item.get("side") == side
        ]
        if not side_move_indices:
            return set()

        properties = set()
        if all(self._move_gives_check(moves[index].get("from_fen", positions[index]), moves[index].get("move", "")) for index in side_move_indices):
            properties.add("long_check")

        any_check = any(
            self._move_gives_check(item.get("from_fen", positions[start_index + offset]), item.get("move", ""))
            for offset, item in enumerate(cycle_moves)
        )
        if any_check:
            return properties

        chased_piece_kind = None
        long_chase = True
        cycle_len = len(cycle_moves)
        for index in side_move_indices:
            next_index = start_index + ((index - start_index + 1) % cycle_len)
            opponent_move = moves[next_index]
            if opponent_move.get("side") != opponent(side):
                long_chase = False
                break

            targets = self._chased_targets_after_move(positions[index + 1], side)
            opp_src_row, opp_src_col, opp_dst_row, opp_dst_col = uci_to_coords(opponent_move.get("move", ""))
            matching = [target for target in targets if target[0] == opp_src_row and target[1] == opp_src_col]
            if not matching:
                long_chase = False
                break

            piece_kind = matching[0][2]
            if chased_piece_kind is None:
                chased_piece_kind = piece_kind
            elif chased_piece_kind != piece_kind:
                long_chase = False
                break

            escaped_targets = self._chased_targets_after_move(positions[next_index + 1], side)
            if any(target[0] == opp_dst_row and target[1] == opp_dst_col and target[2] == piece_kind for target in escaped_targets):
                long_chase = False
                break

        if long_chase and chased_piece_kind is not None:
            properties.add("long_chase")
        return properties

    def _candidate_repetition_violation(self, fen, move):
        if not self._config_bool("auto_change_repetition", default=True):
            return None
        try:
            board, side = fen_to_board(fen)
            to_fen = apply_uci_to_fen(fen, move)
        except Exception:
            return None

        positions, moves = self._position_sequence()
        if not positions:
            positions = [fen]
        elif self._repetition_position_key(positions[-1]) != self._repetition_position_key(fen):
            return None

        notation = move
        try:
            notation = move_to_chinese(board, move)
        except Exception:
            pass

        candidate_move = {
            "ply": len(moves) + 1,
            "side": side,
            "move": move,
            "notation": notation,
            "from_fen": fen,
            "to_fen": to_fen,
            "source": "me",
        }
        candidate_positions = positions + [to_fen]
        candidate_moves = moves + [candidate_move]
        final_key = self._repetition_position_key(to_fen)
        matching_indices = [
            index for index, item in enumerate(candidate_positions)
            if self._repetition_position_key(item) == final_key
        ]
        if len(matching_indices) < 3:
            return None

        start_index = matching_indices[-3]
        properties = self._cycle_side_properties(candidate_positions, candidate_moves, start_index, side)
        if "long_check" in properties:
            return "长将"
        if "long_chase" in properties:
            return "长捉"
        return None

    def _engine_candidate_moves(self, bestmove):
        moves = []
        for move in [bestmove]:
            if move and move != "(none)" and move not in moves:
                moves.append(move)
        for rank in sorted(getattr(self.eng, "pv_moves", {}) or {}):
            move = self.eng.pv_moves.get(rank)
            if move and move != "(none)" and move not in moves:
                moves.append(move)
        return moves

    def _pick_engine_move_avoiding_repetition(self, fen, bestmove):
        if not bestmove or bestmove == "(none)":
            return bestmove, None
        first_violation = None
        for move in self._engine_candidate_moves(bestmove):
            if not self._is_engine_move_legal_for_fen(fen, move):
                continue
            violation = self._candidate_repetition_violation(fen, move)
            if violation:
                if move == bestmove:
                    first_violation = violation
                dprint(f"[Repetition] skip {move}: would cause {violation}")
                continue
            if move != bestmove and first_violation:
                return move, first_violation
            return move, None
        return bestmove, first_violation

    def _format_move_line(self, moves, fen=None, flipped=False):
        if not fen:
            return [flip_move(move) if flipped else move for move in moves]
        try:
            board, _ = fen_to_board(fen)
        except Exception:
            return [flip_move(move) if flipped else move for move in moves]
        current_board = [row[:] for row in board]
        display_moves = []
        for move in moves:
            try:
                display_moves.append(move_to_chinese(current_board, move))
                current_board = apply_uci(current_board, move)
            except Exception:
                display_moves.append(flip_move(move) if flipped else move)
        return display_moves

    def _format_eval_score(self, score):
        if not score:
            return "-"
        parts = str(score).split()
        if len(parts) != 2:
            return str(score)
        kind, value = parts
        if kind == "cp":
            try:
                return f"{int(value) / 100:+.2f}"
            except ValueError:
                return score
        if kind == "mate":
            return f"杀棋 {value}"
        return score

    def _config_bool(self, key, default=False):
        value = getattr(self.vis, "config", {}).get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _config_float(self, key, default=0.0):
        value = getattr(self.vis, "config", {}).get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _config_int(self, key, default=0):
        value = getattr(self.vis, "config", {}).get(key, default)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    def _print_status_once(self, message):
        if self.last_status != message:
            print(message)
            self.last_status = message

    def _search_movetime(self):
        min_time = self._config_float("auto_move_search_time_min", 2.0)
        max_time = self._config_float("auto_move_search_time_max", 5.0)
        if max_time < min_time:
            min_time, max_time = max_time, min_time
        seconds = random.uniform(min_time, max_time)
        return max(1, int(round(seconds * 1000))), seconds

    def _search_depth(self):
        min_depth = self._config_int("engine_depth_min", 8)
        max_depth = self._config_int("engine_depth_max", 12)
        if max_depth < min_depth:
            min_depth, max_depth = max_depth, min_depth
        return max(1, random.randint(min_depth, max_depth))

    def _opening_book_path(self):
        return find_opening_book_path(getattr(self.vis, "config", {}))

    def _opening_book_priority(self):
        return normalize_book_priority(
            getattr(self.vis, "config", {}).get("book_move_priority", BOOK_PRIORITY_SCORE)
        )

    def _pick_opening_book_move(self, fen):
        if not self._config_bool("use_open_book", default=True):
            return None
        moves = query_xqbook_file(self._opening_book_path(), fen, self._opening_book_priority())
        for item in moves:
            move = item.get("uci")
            if self._is_engine_move_legal_for_fen(fen, move):
                return item
        return None

    def _wait_book_move_delay(self):
        _movetime, seconds = self._search_movetime()
        dprint(f"[Book] wait={seconds:.2f}s")
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self._should_cancel_search():
                raise InterruptedError("Book move cancelled")
            time.sleep(min(0.05, max(0.0, end_time - time.time())))

    def _try_opening_book_move(self, fen, flipped, allow_auto_move=True):
        item = self._pick_opening_book_move(fen)
        if not item:
            return False
        book_move = item["uci"]
        gui_move = flip_move(book_move) if flipped else book_move
        move_text = self._format_move_line([book_move], fen, flipped)[0]
        score = book_number(item.get("score"))
        win_rate = book_win_rate(item) * 100
        self._emit_move(gui_move, "库招")
        self._emit_analysis(
            f"命中库招：{move_text}\n"
            f"选招方式：{book_priority_label(self._opening_book_priority())}\n"
            f"分数：{score}    胜率：{win_rate:.1f}%"
        )
        self._wait_book_move_delay()
        moved = allow_auto_move and self._auto_move(fen, gui_move)
        if moved:
            self._record_move(fen, book_move, "me")
            try:
                self._set_expected_own_move_baseline(apply_uci_to_fen(fen, book_move))
            except Exception:
                self._set_expected_own_move_baseline(fen)
            self._print_status_once(f"Book move {gui_move}; waiting for turn to end...")
        else:
            self._set_board_change_baseline(fen)
            self._print_status_once(f"Book move {gui_move}; waiting for turn to end...")
        return True

    def _next_game_template_file(self):
        relative_path = os.path.join("templates", NEXT_GAME_BUTTON_TEMPLATE_NAME)
        for path in candidate_paths(relative_path):
            if os.path.exists(path):
                return path
        return None

    def _load_next_game_template(self):
        path = self._next_game_template_file()
        if not path:
            self._next_game_template = None
            self._next_game_template_path = None
            self._next_game_template_mtime = None
            return None

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None

        if (
            self._next_game_template is not None
            and self._next_game_template_path == path
            and self._next_game_template_mtime == mtime
        ):
            return self._next_game_template

        template = cv2.imread(path)
        if template is None or template.size == 0:
            return None
        self._next_game_template = template
        self._next_game_template_path = path
        self._next_game_template_mtime = mtime
        return template

    def _next_game_template_missing(self):
        now = time.time()
        if now - self._next_game_missing_logged_at > 5.0:
            self._next_game_missing_logged_at = now
            dprint("[Next game] button template missing. Click Recognize Next first.")

    def _next_game_screen_origin(self):
        screen_left = self._config_float(
            "next_game_screen_left",
            self._config_float("screen_left", 0.0),
        )
        screen_top = self._config_float(
            "next_game_screen_top",
            self._config_float("screen_top", 0.0),
        )
        return screen_left, screen_top

    def _match_next_game_template(self, screen_img, template, x1, y1, x2, y2, screen_left, screen_top):
        tmpl_h, tmpl_w = template.shape[:2]
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(screen_img.shape[1], int(round(x2)))
        y2 = min(screen_img.shape[0], int(round(y2)))
        roi = screen_img[y1:y2, x1:x2]
        if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
            return None

        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        threshold = self._config_float(
            "next_game_match_threshold",
            NEXT_GAME_MATCH_THRESHOLD_DEFAULT,
        )
        if score < threshold:
            return None
        local_x = x1 + max_loc[0] + tmpl_w / 2
        local_y = y1 + max_loc[1] + tmpl_h / 2
        return score, local_x + screen_left, local_y + screen_top

    def _next_game_search_window(self, screen_img, template):
        center_x = self._config_float("next_game_button_x", -1.0)
        center_y = self._config_float("next_game_button_y", -1.0)
        if center_x < 0 or center_y < 0:
            return None

        screen_left, screen_top = self._next_game_screen_origin()
        local_x = center_x - screen_left
        local_y = center_y - screen_top
        radius = self._config_float(
            "next_game_search_radius",
            NEXT_GAME_SEARCH_RADIUS_DEFAULT,
        )
        tmpl_h, tmpl_w = template.shape[:2]
        x1 = max(0, int(round(local_x - radius - tmpl_w / 2)))
        y1 = max(0, int(round(local_y - radius - tmpl_h / 2)))
        x2 = min(screen_img.shape[1], int(round(local_x + radius + tmpl_w / 2)))
        y2 = min(screen_img.shape[0], int(round(local_y + radius + tmpl_h / 2)))
        if x2 - x1 < tmpl_w or y2 - y1 < tmpl_h:
            return None
        return x1, y1, x2, y2, screen_left, screen_top

    def _locate_next_game_button(self, screen_img, template):
        window = self._next_game_search_window(screen_img, template)
        if window is not None:
            match = self._match_next_game_template(screen_img, template, *window)
            if match is not None:
                return match

        if not self._config_bool("next_game_fullscreen_scan", default=True):
            return None

        screen_left, screen_top = self._next_game_screen_origin()
        return self._match_next_game_template(
            screen_img,
            template,
            0,
            0,
            screen_img.shape[1],
            screen_img.shape[0],
            screen_left,
            screen_top,
        )

    def _click_next_game_button(self, x, y):
        previous_pos = self._mouse.position
        delay = float(self.vis.config.get("auto_move_click_delay", 0.08))
        try:
            self._set_mouse_position((x, y))
            time.sleep(delay)
            self._click_left()
        finally:
            if self._config_bool("auto_move_restore_mouse", default=False):
                self._set_mouse_position(previous_pos)

    def _maybe_start_next_game(self, screen_img):
        if not self._config_bool("auto_start_next_game", default=False):
            return False
        if not self._mouse_available:
            return False

        now = time.time()
        cooldown = self._config_float(
            "next_game_click_cooldown_seconds",
            NEXT_GAME_CLICK_COOLDOWN_DEFAULT,
        )
        if now - self.next_game_last_click_at < cooldown:
            return False
        scan_interval = self._config_float(
            "next_game_scan_interval_seconds",
            NEXT_GAME_SCAN_INTERVAL_DEFAULT,
        )
        if now - self.next_game_last_scan_at < scan_interval:
            return False
        self.next_game_last_scan_at = now

        template = self._load_next_game_template()
        if template is None:
            self._next_game_template_missing()
            return False

        match = self._locate_next_game_button(screen_img, template)
        if match is None:
            return False

        score, x, y = match
        dprint(f"[Next game] button detected score={score:.3f}; clicking.")
        self._click_next_game_button(x, y)
        self.next_game_last_click_at = now
        self._reset_state()
        self.waiting_for_new_game = True
        self._print_status_once("Clicked next game; waiting for new game...")
        return True

    def _fen_board_key(self, fen):
        return fen.strip().split()[0] if fen else ""

    def _is_initial_board_key(self, board_key):
        return board_key == self._fen_board_key(START_FEN)

    def _screen_player_side(self, flipped):
        return "black" if flipped else "red"

    def _piece_count_from_board_key(self, board_key):
        return sum(1 for ch in board_key if ch.isalpha())

    def _set_board_change_baseline(self, fen):
        key = self._fen_board_key(fen)
        self.board_change_baseline_fen = fen
        self.board_change_baseline_key = key
        self.board_change_baseline_piece_count = self._piece_count_from_board_key(key)
        self.board_change_initialized = True
        self.board_change_invalid_key = None
        self.board_change_invalid_count = 0
        self.board_change_strict_baseline = False

    def _set_expected_own_move_baseline(self, fen):
        self._set_board_change_baseline(fen)
        self.board_change_strict_baseline = True

    def _find_legal_transition(self, from_fen, current_key, sides=None):
        if not from_fen:
            return None
        try:
            board, baseline_side = fen_to_board(from_fen)
            side_order = sides if sides is not None else (
                baseline_side,
                opponent(baseline_side),
            )
            tried = []
            for side in side_order:
                if side in tried:
                    continue
                tried.append(side)
                for move in legal_moves(board, side):
                    next_board = apply_uci(board, move)
                    move_from_fen = board_to_fen(board, side)
                    next_fen = board_to_fen(next_board, opponent(side))
                    if self._fen_board_key(next_fen) == current_key:
                        return {
                            "from_fen": move_from_fen,
                            "to_fen": next_fen,
                            "move": move,
                            "side": side,
                        }
        except Exception as e:
            dprint(f"[Board change validation skipped] {e}")
            return None
        return None

    def _find_legal_transition_from_baseline(self, current_key):
        return self._find_legal_transition(self.board_change_baseline_fen, current_key)

    def _is_legal_next_board_key(self, current_key):
        return not self.board_change_baseline_fen or self._find_legal_transition_from_baseline(current_key) is not None

    def _record_move(self, from_fen, move, source):
        try:
            board, side = fen_to_board(from_fen)
            to_fen = apply_uci_to_fen(from_fen, move)
            notation = move_to_chinese(board, move)
        except Exception as e:
            dprint(f"[Record skipped] {source} {move}: {e}")
            return

        if self.record_moves and self.record_moves[-1].get("to_fen") == to_fen:
            return
        if self.record_start_fen is None:
            self.record_start_fen = from_fen
        self.record_moves.append({
            "ply": len(self.record_moves) + 1,
            "side": side,
            "move": move,
            "notation": notation,
            "from_fen": from_fen,
            "to_fen": to_fen,
            "source": source,
        })
        self.record_saved = False
        self._emit_record()

    def _record_transition(self, transition, source):
        if transition:
            self._record_move(transition["from_fen"], transition["move"], source)

    def _record_history_text(self):
        if not self.record_moves:
            return "还没有走法。"

        lines = []
        for index in range(0, len(self.record_moves), 2):
            red = self.record_moves[index]
            red_text = red.get("notation") or red.get("move", "")
            black_text = ""
            if index + 1 < len(self.record_moves):
                black = self.record_moves[index + 1]
                black_text = black.get("notation") or black.get("move", "")
            line = f"第{index // 2 + 1}回合  {red_text}"
            if black_text:
                line += f"    {black_text}"
            lines.append(line)
        return "\n".join(lines)

    def _record_dir(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_game_record(self, reason=""):
        if self.record_saved or not self.record_moves:
            return None
        path = self._write_record_moves(self.record_moves, self.record_start_fen, reason)
        if path:
            self.record_saved = True
        return path

    def _write_record_moves(self, moves, start_fen=None, reason=""):
        if not moves:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self._record_dir(), f"xiangqi-{stamp}.pgn")
        lines = [
            '[Event "象棋助手联机记录"]',
            f'[Date "{time.strftime("%Y.%m.%d")}"]',
            f'[Reason "{reason or "manual"}"]',
        ]
        if start_fen and start_fen != START_FEN:
            lines.append(f'[FEN "{start_fen}"]')
        lines.append("")
        move_parts = []
        for index in range(0, len(moves), 2):
            red = moves[index]["move"]
            black = moves[index + 1]["move"] if index + 1 < len(moves) else ""
            move_parts.append(f"{index // 2 + 1}. {red}" + (f" {black}" if black else ""))
        lines.append(" ".join(move_parts))
        lines.append("")
        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
        dprint(f"[Record] saved {path}")
        return path

    def save_current_record(self, reason="manual"):
        path = self._save_game_record(reason)
        if path:
            return path
        if not self.history_moves:
            return None
        moves = []
        for index, move in enumerate(self.history_moves):
            from_fen = self.history_fens[index] if index < len(self.history_fens) else ""
            try:
                board, side = fen_to_board(from_fen)
                notation = move_to_chinese(board, move)
                to_fen = apply_uci_to_fen(from_fen, move)
            except Exception:
                side = "red" if index % 2 == 0 else "black"
                notation = move
                to_fen = ""
            moves.append({
                "ply": index + 1,
                "side": side,
                "move": move,
                "notation": notation,
                "from_fen": from_fen,
                "to_fen": to_fen,
                "source": "local",
            })
        start_fen = self.history_fens[0] if self.history_fens else START_FEN
        return self._write_record_moves(moves, start_fen, reason)

    def _clear_game_record(self):
        self.record_start_fen = None
        self.record_moves = []
        self.record_saved = False
        self._emit_record()

    def _invalid_board_recovered(self, fen, current_key, min_stable):
        if self.board_change_invalid_key == current_key:
            self.board_change_invalid_count += 1
        else:
            self.board_change_invalid_key = current_key
            self.board_change_invalid_count = 1

        recover_frames = max(
            min_stable * 4,
            self._config_int("board_change_invalid_recover_frames", 12),
        )
        if self.stable_count >= min_stable and self.board_change_invalid_count >= recover_frames:
            self._set_board_change_baseline(fen)
            self._print_status_once("异常棋盘已稳定，重新等待对方走棋...")
            return True
        return False

    def _stable_frame_count_needed(self):
        return max(
            1,
            self._config_int(
                "board_stable_frames",
                BOARD_STABLE_FRAMES_DEFAULT,
            ),
        )

    def _update_stability(self, fen):
        if self.last_seen_fen == fen:
            self.stable_count += 1
        else:
            self.last_seen_fen = fen
            self.stable_count = 1

    def _read_board_position(self, screen_img, quiet_invalid=False):
        board_img = self.vis.detect_board(screen_img)
        if not self.vis.board_visible:
            return None

        mouse_pos = self._mouse.position if self._mouse_available else None
        grid = self.vis.get_board_state(board_img, mouse_screen_pos=mouse_pos)
        fen_unturned = self.vis.to_fen(grid, turn='w')
        if 'K' not in fen_unturned or 'k' not in fen_unturned:
            return None

        flipped = any('K' in grid[r] for r in range(5))
        bottom_turn = 'b' if flipped else 'w'
        process_grid = [row[::-1] for row in grid[::-1]] if flipped else grid
        fen = self.vis.to_fen(process_grid, turn=bottom_turn)
        try:
            validate_fen(fen)
        except ValueError as e:
            if not quiet_invalid:
                now = time.time()
                if not hasattr(self, '_last_invalid') or now - self._last_invalid > 2.0:
                    self._last_invalid = now
                    dprint(f"[Invalid FEN skipped] {e}")
            return None

        self._update_stability(fen)
        if self.board_callback is not None:
            self.board_callback([row[:] for row in grid])
        if self.fen_callback is not None:
            self.fen_callback(fen)
        return fen, grid, flipped

    def _board_cell_screen_pos(self, row, col):
        x_offset = float(self.vis.config.get("auto_move_click_offset_x", 0.0))
        y_offset = float(self.vis.config.get("auto_move_click_offset_y", 0.0))
        return self.vis.board_cell_screen_pos(row, col, x_offset, y_offset)

    def _set_mouse_position(self, pos):
        if not self._mouse_available:
            raise RuntimeError("Mouse control is not available.")
        self._mouse.position = (int(round(pos[0])), int(round(pos[1])))

    def _click_left(self):
        if hasattr(self._mouse, "press") and hasattr(self._mouse, "release"):
            self._mouse.press(Button.left)
            time.sleep(0.03)
            self._mouse.release(Button.left)
        else:
            self._mouse.click(Button.left)

    def _double_click_left(self):
        interval = float(self.vis.config.get("auto_move_double_click_interval", 0.08))
        self._click_left()
        time.sleep(interval)
        self._click_left()

    def _move_mouse_between(self, src, dst):
        duration = float(self.vis.config.get("auto_move_move_duration", 0.18))
        steps = max(1, int(self.vis.config.get("auto_move_move_steps", 12)))
        if duration <= 0 or steps <= 1:
            self._set_mouse_position(dst)
            return

        step_delay = duration / steps
        for i in range(1, steps + 1):
            t = i / steps
            x = src[0] + (dst[0] - src[0]) * t
            y = src[1] + (dst[1] - src[1]) * t
            self._set_mouse_position((x, y))
            time.sleep(step_delay)

    def _mouse_away_position(self):
        if "auto_move_mouse_away_x" in self.vis.config and "auto_move_mouse_away_y" in self.vis.config:
            return (
                float(self.vis.config["auto_move_mouse_away_x"]),
                float(self.vis.config["auto_move_mouse_away_y"]),
            )

        if sys.platform == "win32":
            try:
                import ctypes

                user32 = ctypes.windll.user32
                virtual_left = user32.GetSystemMetrics(76)
                virtual_top = user32.GetSystemMetrics(77)
                virtual_width = user32.GetSystemMetrics(78)
                virtual_height = user32.GetSystemMetrics(79)
                return (
                    virtual_left + virtual_width / 2,
                    virtual_top + virtual_height - 2,
                )
            except Exception:
                pass

        return (
            float(self.vis.config.get("screen_left", 0)) + 20.0,
            float(self.vis.config.get("screen_top", 0))
            + float(self.vis.config.get("screen_height", 1080))
            - 2.0,
        )

    def _auto_move(self, fen, gui_move):
        if not self._config_bool("auto_move", default=True):
            dprint(f"[Auto move skipped] disabled for {gui_move}")
            return False
        if not self._mouse_available:
            dprint(f"[Auto move skipped] mouse control unavailable for {gui_move}")
            return False

        try:
            src_row, src_col, dst_row, dst_col = uci_to_coords(gui_move)
        except ValueError as e:
            dprint(f"[Auto move skipped] {e}")
            return False

        src = self._board_cell_screen_pos(src_row, src_col)
        dst = self._board_cell_screen_pos(dst_row, dst_col)
        delay = float(self.vis.config.get("auto_move_click_delay", 0.08))
        previous_pos = self._mouse.position
        dprint(
            f"[Auto move] move={gui_move} "
            f"src=({src_row},{src_col})->({src[0]:.1f},{src[1]:.1f}) "
            f"dst=({dst_row},{dst_col})->({dst[0]:.1f},{dst[1]:.1f}) "
            f"delay={delay}"
        )

        try:
            self._set_mouse_position(src)
            time.sleep(delay)
            dprint("[Auto move] double-click source")
            self._double_click_left()
            time.sleep(delay)
            dprint("[Auto move] move to destination")
            self._move_mouse_between(src, dst)
            time.sleep(delay)
            dprint("[Auto move] click destination")
            self._click_left()
            away = self._mouse_away_position()
            dprint(f"[Auto move] move mouse away -> ({away[0]:.1f},{away[1]:.1f})")
            self._move_mouse_between(dst, away)
            dprint(f"[Auto move] done {gui_move}")
            return True
        finally:
            if self._config_bool("auto_move_restore_mouse", default=False):
                self._set_mouse_position(previous_pos)

    def _key_listener(self):
        if msvcrt is not None:
            while not self._stop_requested.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("r", "R"):
                        self._reset_requested.set()
                        dprint("[r] Reset and recalculate.", flush=True)
                    elif ch == "\x03":
                        os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.05)
            return

        if termios is None or tty is None:
            return

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # cbreak (not raw) keeps OPOST on so '\n' still does CR+LF in
            # the terminal — otherwise printed lines stair-step to the right.
            tty.setcbreak(fd)
            while True:
                ch = os.read(fd, 1)
                if ch in (b'r', b'R'):
                    self._reset_requested.set()
                    dprint("[r] Reset and recalculate.", flush=True)
                elif ch == b'\x03':  # Ctrl+C
                    os.kill(os.getpid(), signal.SIGINT)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def run(self, use_keyboard=True):
        keyboard_available = (termios is not None and tty is not None) or msvcrt is not None
        if use_keyboard and sys.stdin.isatty() and keyboard_available:
            t = threading.Thread(target=self._key_listener, daemon=True)
            t.start()
            dprint("Ready. Press 'r' to recalculate. Ctrl+C to exit.\n")
        else:
            dprint("Ready.\n")
        try:
            while not self._stop_requested.is_set():
                self._tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            dprint("\nExiting.")
        finally:
            self.eng.quit()

    def calculate_current_board(self):
        try:
            img = capture_screen(self.vis.config)
        except Exception as e:
            dprint(f"[Screen capture failed] {e}")
            self.vis.invalidate_homography()
            return False

        position = self._read_board_position(img)
        if position is None:
            dprint("No readable board found.")
            return False

        fen, _, flipped = position
        self._emit_move("", "")
        return self._calculate_and_print(fen, flipped, allow_auto_move=False)

    def _tick(self):
        if self._stop_requested.is_set():
            return
        if self._reload_requested.is_set():
            self._reload_requested.clear()
            try:
                self._reload()
            except Exception as e:
                dprint(f"[Reload failed] {e}")
                return
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._reset_state()
            self._recalc.set()

        try:
            try:
                img = capture_screen(self.vis.config)
            except Exception as e:
                now = time.time()
                if not hasattr(self, '_last_cap_err') or now - self._last_cap_err > 5.0:
                    self._last_cap_err = now
                    dprint(f"[Screen capture failed] {e}")
                self.vis.invalidate_homography()
                return
            if self._maybe_start_next_game(img):
                return

            position = self._read_board_position(
                img,
                quiet_invalid=self.waiting_for_new_game,
            )
            if position is None:
                if self.waiting_for_new_game:
                    self._print_status_once("Waiting for new game board...")
                elif self._manual_calculate_requested.is_set():
                    self._manual_calculate_requested.clear()
                    self._print_status_once("No readable board found.")
                return
            fen, _, flipped = position
            manual_calculate = self._manual_calculate_requested.is_set()

            if manual_calculate:
                self._manual_calculate_requested.clear()
                self._recalc.clear()
                self._emit_move("", "")
                moved_or_reported = self._calculate_and_print(
                    fen,
                    flipped,
                    allow_auto_move=False,
                )
                if moved_or_reported:
                    self.manual_result_hold_until = time.time() + 8.0
                return

            min_stable = self._stable_frame_count_needed()
            if self.waiting_for_new_game:
                if self.stable_count < min_stable:
                    self._print_status_once(
                        f"New game detected; waiting stable board "
                        f"{self.stable_count}/{min_stable}..."
                    )
                    return
                self.waiting_for_new_game = False
                self._print_status_once("New game ready; waiting for my turn...")

            if True:
                force = self._recalc.is_set()
                current_key = self._fen_board_key(fen)
                if not self.board_change_initialized:
                    if self.stable_count < min_stable:
                        self._print_status_once(
                            f"开始连线，等待当前局面稳定 {self.stable_count}/{min_stable}..."
                        )
                        return
                    transition = None
                    if not self._is_initial_board_key(current_key):
                        transition = self._find_legal_transition(
                            START_FEN,
                            current_key,
                            sides=("red",),
                        )
                    analysis_fen = fen
                    if transition is not None:
                        self._record_transition(transition, "opponent")
                        analysis_fen = transition["to_fen"]
                        self._print_status_once("已记录对方第一步，开始计算当前回合...")
                    else:
                        self._print_status_once("开始连线：按当前回合计算...")
                    self._recalc.clear()
                    moved_or_reported = self._calculate_and_print(analysis_fen, flipped)
                    return

                if not force and current_key == self.board_change_baseline_key:
                    if self._is_initial_board_key(current_key) and self._screen_player_side(flipped) == "black":
                        self._print_status_once("对方红方先走，等待红方走棋...")
                    else:
                        self._print_status_once("等待对方走棋...")
                    return

                if self.record_moves and self._is_initial_board_key(current_key):
                    self._reset_state()
                    self._set_board_change_baseline(fen)
                    self._print_status_once("新棋盘已识别，联机记录已清空。")
                    return

                current_piece_count = self._piece_count_from_board_key(current_key)
                if (
                    self.board_change_baseline_piece_count
                    and current_piece_count < self.board_change_baseline_piece_count - 1
                ):
                    if (
                        not self.board_change_strict_baseline
                        and self._invalid_board_recovered(fen, current_key, min_stable)
                    ):
                        return
                    self._print_status_once(
                        f"棋子数量异常，等待棋盘恢复 {current_piece_count}/{self.board_change_baseline_piece_count}..."
                    )
                    return

                transition = self._find_legal_transition_from_baseline(current_key)
                if not force and transition is None:
                    if (
                        not self.board_change_strict_baseline
                        and self._invalid_board_recovered(fen, current_key, min_stable)
                    ):
                        return
                    if self.board_change_strict_baseline:
                        self._print_status_once("等待对方走出下一步合法棋...")
                    else:
                        self._print_status_once("棋盘变化不是一步合法棋，等待识别恢复...")
                    return

                self.board_change_invalid_key = None
                self.board_change_invalid_count = 0
                self.board_change_strict_baseline = False

                if self.stable_count < min_stable:
                    self._print_status_once(
                        f"对方已走棋，等待棋盘稳定 {self.stable_count}/{min_stable}..."
                    )
                    return

                if transition is not None:
                    self._record_transition(transition, "opponent")
                self._recalc.clear()
                analysis_fen = transition["to_fen"] if transition is not None else fen
                moved_or_reported = self._calculate_and_print(analysis_fen, flipped)
                return

        except Exception as e:
            if DEBUG:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()

    def _calculate_and_print(self, fen, flipped, allow_auto_move=True):
        try:
            validate_fen(fen)
        except ValueError as e:
            dprint(f"[Invalid FEN, not sending to engine] {e}")
            return False

        try:
            if self._try_opening_book_move(fen, flipped, allow_auto_move=allow_auto_move):
                return True
        except InterruptedError:
            return False
        except Exception as e:
            dprint(f"[Book move failed] {e}")

        try:
            dprint("Calculating...", end='\r', flush=True)
            dprint(f"\n[Calculate] fen={fen}")
            if self.vis.config.get("engine_search_mode") == "depth":
                depth = self._search_depth()
                dprint(f"[Calculate] search_depth={depth}")
                self.eng.start_search(
                    fen,
                    depth=depth,
                    should_cancel=self._should_cancel_search,
                )
            else:
                movetime, search_seconds = self._search_movetime()
                dprint(f"[Calculate] search_time={search_seconds:.2f}s movetime={movetime}ms")
                self.eng.start_search(
                    fen,
                    movetime=movetime,
                    should_cancel=self._should_cancel_search,
                )
        except InterruptedError:
            self.eng.send_cmd("stop")
            return False
        except Exception as e:
            dprint(f"[Engine start failed] {e}")
            return False

        start = time.time()
        last_pv = None
        timeout = max(10.0, self._config_float("engine_timeout_seconds", 10.0))
        while self.eng.bestmove is None:
            if self._should_cancel_search():
                self.eng.send_cmd("stop")
                return False

            if time.time() - start > timeout:
                dprint("\n[Engine timeout] Stopping search.")
                self.eng.send_cmd("stop")
                self.eng.restart()
                return False

            if self.eng.process.poll() is not None:
                dprint("[Engine crashed] Restarting...")
                self.eng.restart()
                return False

            pv = self.eng.current_pv_move
            if pv:
                if not self._is_engine_move_legal_for_fen(fen, pv):
                    time.sleep(0.05)
                    continue
                gui_pv = flip_move(pv) if flipped else pv
                if gui_pv != last_pv:
                    last_pv = gui_pv
                    self._emit_move(gui_pv, "PV")
                    print(f"PV: {self._format_move_line([pv], fen, flipped)[0]}")
            self._emit_analysis(self._format_engine_analysis(fen, flipped))
            time.sleep(0.05)

        bm = self.eng.bestmove
        if not bm or bm == "(none)":
            dprint("[Best move] none")
            return False
        selected_bm, repetition_violation = self._pick_engine_move_avoiding_repetition(fen, bm)
        if selected_bm and selected_bm != bm:
            old_text = self._format_move_line([bm], fen, flipped)[0]
            new_text = self._format_move_line([selected_bm], fen, flipped)[0]
            message = f"检测到最佳招可能形成{repetition_violation}三循环，自动改用候选招：{old_text} -> {new_text}"
            dprint(f"[Repetition] {message}")
            current_analysis = self._format_engine_analysis(fen, flipped)
            self._emit_analysis((current_analysis + "\n" if current_analysis else "") + message)
            bm = selected_bm

        gui_move = flip_move(bm) if flipped else bm
        self._emit_move(gui_move, "最佳")
        self._emit_analysis(self._format_engine_analysis(fen, flipped))
        dprint(f"[Best move] engine={bm} gui={gui_move}")
        print(f"Best move: {self._format_move_line([bm], fen, flipped)[0]}")
        self.last_status = None
        moved = allow_auto_move and self._auto_move(fen, gui_move)
        if moved:
            self._record_move(fen, bm, "me")
            try:
                self._set_expected_own_move_baseline(apply_uci_to_fen(fen, bm))
            except Exception:
                self._set_expected_own_move_baseline(fen)
        else:
            self._set_board_change_baseline(fen)
        if moved:
            self._print_status_once(f"Played {gui_move}; waiting for turn to end...")
        else:
            self._print_status_once(f"Best move {gui_move}; waiting for turn to end...")
        return True


def _ensure_cooked_terminal():
    """Reset the controlling terminal to cooked mode.

    A previous run that died before its daemon key-listener could restore
    termios may have left the TTY in raw mode (OPOST cleared). Until we fix
    it, every '\\n' becomes a bare line-feed and printed lines stair-step
    to the right. Force OPOST/ONLCR + ICANON/ECHO/ISIG back on so output
    starts at column 0.
    """
    if termios is None:
        return
    if not sys.stdin.isatty():
        return
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        attrs[1] |= termios.OPOST | termios.ONLCR  # output processing
        attrs[3] |= termios.ICANON | termios.ECHO | termios.ISIG
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        pass


def launch_gui():
    from PyQt5.QtCore import QObject, QRectF, QSize, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QAbstractItemView,
        QHeaderView,
        QListWidget,
        QMainWindow,
        QMenu,
        QMenuBar,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    class GuiSignals(QObject):
        status = pyqtSignal(str)
        board = pyqtSignal(object)
        move = pyqtSignal(str, str)
        fen = pyqtSignal(str)
        analysis = pyqtSignal(str)
        engine_score = pyqtSignal(str, int)
        history_score = pyqtSignal(int, int, int)
        record = pyqtSignal(str)
        local_move = pyqtSignal(str)
        ai_done = pyqtSignal()
        score_analysis_done = pyqtSignal(int, int, int)
        stopped = pyqtSignal()
        calculation_done = pyqtSignal()
        calibration_done = pyqtSignal()
        recognition_done = pyqtSignal()

    class ScoreChartWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.score_records = []
            self.setMinimumHeight(100)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def set_score_records(self, records):
            cleaned = []
            for x_value, score in records:
                try:
                    cleaned.append((int(x_value), clamp_score_chart_value(score)))
                except (TypeError, ValueError):
                    continue
            if cleaned == self.score_records:
                return
            self.score_records = cleaned
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#ffffff"))

            left = 48
            right = 10
            top = 8
            bottom = 22
            plot_left = left
            plot_top = top
            plot_width = max(1, self.width() - left - right)
            plot_height = max(1, self.height() - top - bottom)
            plot_right = plot_left + plot_width
            plot_bottom = plot_top + plot_height

            painter.fillRect(
                QRectF(plot_left, plot_top, plot_width, plot_height),
                QColor("#f9f9f9"),
            )

            def y_for(score):
                return plot_top + (1000 - clamp_score_chart_value(score)) / 2000.0 * plot_height

            painter.setFont(QFont("Microsoft YaHei", 8))
            for tick in (-1000, -500, 0, 500, 1000):
                y = y_for(tick)
                painter.setPen(QPen(QColor("#dddddd"), 1))
                painter.drawLine(int(plot_left), int(y), int(plot_right), int(y))
                painter.setPen(QPen(QColor("#555555"), 1))
                painter.drawText(
                    QRectF(0, y - 8, left - 6, 16),
                    Qt.AlignRight | Qt.AlignVCenter,
                    str(tick),
                )

            painter.setPen(QPen(QColor("#8a8a8a"), 1))
            painter.drawLine(plot_left, plot_top, plot_left, plot_bottom)
            painter.drawLine(plot_left, plot_bottom, plot_right, plot_bottom)

            if self.score_records:
                min_x = min(x for x, _score in self.score_records)
                max_x = max(x for x, _score in self.score_records)

                def x_for(x_value):
                    if min_x == max_x:
                        return plot_left + plot_width / 2.0
                    return plot_left + (x_value - min_x) / (max_x - min_x) * plot_width

                points = [(x_for(x), y_for(score)) for x, score in self.score_records]
                painter.setPen(QPen(QColor("#f3622d"), 1))
                for index in range(1, len(points)):
                    x1, y1 = points[index - 1]
                    x2, y2 = points[index]
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            painter.end()

    class HelperWindow(QMainWindow):
        PIECE_LABELS = {
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
        BOARD_STYLES = {
            "classic": "assets/boardchess.jpg",
        }

        def __init__(self):
            super().__init__()
            self.app_worker = None
            self.worker_thread = None
            self.calculate_thread = None
            self.calibrate_thread = None
            self.recognize_thread = None
            self.closing = False
            self.signals = GuiSignals()
            self.signals.status.connect(self.set_status)
            self.signals.board.connect(self.update_board_display)
            self.signals.move.connect(self.update_move_display)
            self.signals.fen.connect(self.update_fen)
            self.signals.analysis.connect(self.update_analysis_display)
            self.signals.engine_score.connect(self.update_engine_score)
            self.signals.history_score.connect(self.update_history_score)
            self.signals.record.connect(self.update_record_display)
            self.signals.local_move.connect(self.apply_ai_move)
            self.signals.ai_done.connect(self.on_ai_done)
            self.signals.score_analysis_done.connect(self.on_score_analysis_done)
            self.signals.stopped.connect(self.on_stopped)
            self.signals.calculation_done.connect(self.on_calculation_done)
            self.signals.calibration_done.connect(self.on_calibration_done)
            self.signals.recognition_done.connect(self.on_recognition_done)

            self.setWindowTitle("象棋助手")
            self.setMinimumWidth(760)
            self.setMinimumHeight(580)
            self.resize(820, 580)
            self.current_grid = None
            self.current_move = ""
            self.current_move_label = ""
            self.current_fen = ""
            self.analysis_text = ""
            self.online_record_text = ""
            self.settings_dialog = None
            self.engine_settings_dialog = None
            self.analysis_dialog = None
            self.text_dialogs = []
            self.game_mode = "assistant"
            self.game_board = None
            self.game_side = "red"
            self.player_side = "red"
            self.board_flipped = False
            self.selected_square = None
            self.legal_target_moves = {}
            self.move_history = []
            self.history_fens = []
            self.history_moves = []
            self.history_scores = []
            self.history_index = 0
            self.ai_thread = None
            self.engine_analysis_thread = None
            self.engine_analysis_request_id = 0
            self.analysis_detail_enabled = False
            self.analysis_detail_fen = ""
            self.pending_ai_movetime = 1000
            self.pending_ai_depth = None
            self.pending_ai_fen = ""
            self.score_analysis_thread = None
            self.score_analysis_request_id = 0
            self.board_draw_rect = None
            self.board_style = "classic"
            self.show_board_numbers = False
            self.show_step_tip = True
            self.step_sound_enabled = False
            self.auto_move_enabled = True
            self.link_mode_active = False
            self.use_open_book = True
            self.book_move_priority = BOOK_PRIORITY_SCORE
            self.pending_book_fen = ""
            self.book_settings_dialog = None
            self.opening_book_path = self.find_opening_book_path()

            self.board_label = QLabel()
            self.board_label.setAlignment(Qt.AlignCenter)
            self.board_label.setMinimumSize(430, 480)
            self.board_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.board_label.setToolTip("读取屏幕后，这里会同步显示识别到的棋盘。")
            self.board_label.mousePressEvent = self.handle_board_click
            self.board_pixmap = self.load_board_pixmap()
            if self.board_pixmap.isNull():
                self.board_label.setText("棋盘图片未找到")
            else:
                self.update_board_display(None)

            self.status = QLabel("已停止")
            self.status.setAlignment(Qt.AlignCenter)
            self.analysis_detail_restart_timer = QTimer(self)
            self.analysis_detail_restart_timer.setSingleShot(True)
            self.analysis_detail_restart_timer.setInterval(250)
            self.analysis_detail_restart_timer.timeout.connect(self.start_analysis_detail_analysis)

            self.min_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.max_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.min_depth_spin = QSpinBox()
            self.min_depth_spin.setRange(1, 99)
            self.min_depth_spin.setValue(8)
            self.max_depth_spin = QSpinBox()
            self.max_depth_spin.setRange(1, 99)
            self.max_depth_spin.setValue(12)
            self.engine_search_mode_combo = QComboBox()
            self.engine_search_mode_combo.addItems(["随机时间", "随机深度"])
            self.engine_search_mode_combo.setFixedWidth(78)
            self.book_priority_combo = QComboBox()
            self.book_priority_combo.blockSignals(True)
            self.book_priority_combo.addItems(["分数优先", "胜率优先"])
            self.book_priority_combo.blockSignals(False)
            self.book_priority_combo.setToolTip("库里有多步时，选择按分数还是胜率优先。")
            self.book_priority_combo.currentTextChanged.connect(self.on_book_priority_changed)
            self.engine_threads_spin = QSpinBox()
            self.engine_threads_spin.setRange(1, 128)
            self.engine_threads_spin.setValue(1)
            self.engine_hash_spin = QSpinBox()
            self.engine_hash_spin.setRange(1, 4096)
            self.engine_hash_spin.setValue(128)
            self.min_time_spin.setFixedWidth(58)
            self.max_time_spin.setFixedWidth(58)
            self.min_depth_spin.setFixedWidth(42)
            self.max_depth_spin.setFixedWidth(42)
            self.calibrate_button = QPushButton("校准棋盘")
            self.auto_next_checkbox = QCheckBox("自动开始下一局")
            self.recognize_next_button = QPushButton("识别下一局按钮")
            self.min_time_spin.setToolTip("引擎最少思考几秒。")
            self.max_time_spin.setToolTip("引擎最多思考几秒。")
            self.min_depth_spin.setToolTip("引擎最少搜索深度。")
            self.max_depth_spin.setToolTip("引擎最多搜索深度。")
            self.engine_search_mode_combo.setToolTip("选择按随机时间还是随机深度搜索。")
            self.engine_threads_spin.setToolTip("Pikafish 使用多少个 CPU 核心，默认 1。")
            self.engine_hash_spin.setToolTip("Pikafish Hash，最高 4096 MB。")
            self.calibrate_button.setToolTip("重新校准棋盘位置。")
            self.auto_next_checkbox.setToolTip("一局结束后，如果识别到下一局按钮，就自动点击进入下一局。")
            self.recognize_next_button.setToolTip("在结算界面点击下一局按钮中心，让程序记住按钮样子。")

            self.calibrate_button.clicked.connect(self.calibrate_helper)
            self.recognize_next_button.clicked.connect(self.recognize_next_game)
            self.engine_search_mode_combo.currentTextChanged.connect(lambda _text: self.on_engine_settings_changed())
            self.min_time_spin.valueChanged.connect(lambda _value: self.on_engine_settings_changed())
            self.max_time_spin.valueChanged.connect(lambda _value: self.on_engine_settings_changed())
            self.min_depth_spin.valueChanged.connect(lambda _value: self.on_engine_settings_changed())
            self.max_depth_spin.valueChanged.connect(lambda _value: self.on_engine_settings_changed())
            self.engine_threads_spin.valueChanged.connect(self.on_engine_threads_spin_changed)
            self.engine_hash_spin.valueChanged.connect(self.on_engine_hash_spin_changed)

            move_time_row = QHBoxLayout()
            move_time_row.setContentsMargins(0, 0, 0, 0)
            move_time_row.setSpacing(4)
            move_time_row.addWidget(self.min_time_spin)
            move_time_row.addWidget(QLabel("-"))
            move_time_row.addWidget(self.max_time_spin)

            depth_range_row = QHBoxLayout()
            depth_range_row.setContentsMargins(0, 0, 0, 0)
            depth_range_row.setSpacing(4)
            depth_range_row.addWidget(self.min_depth_spin)
            depth_range_row.addWidget(QLabel("-"))
            depth_range_row.addWidget(self.max_depth_spin)

            self.engine_settings_panel = QWidget()
            engine_settings_layout = QVBoxLayout()
            engine_settings_layout.setContentsMargins(4, 0, 4, 2)
            engine_settings_layout.setSpacing(2)
            engine_range_row = QHBoxLayout()
            engine_range_row.setContentsMargins(0, 0, 0, 0)
            engine_range_row.setSpacing(4)
            engine_range_row.addWidget(self.engine_search_mode_combo)
            engine_range_row.addWidget(QLabel("时"))
            engine_range_row.addLayout(move_time_row)
            engine_range_row.addWidget(QLabel("深"))
            engine_range_row.addLayout(depth_range_row)
            engine_range_row.addStretch(1)
            engine_settings_layout.addLayout(engine_range_row)
            self.engine_settings_panel.setLayout(engine_settings_layout)

            link_settings = QGroupBox("连线设置")
            link_settings_form = QFormLayout()
            link_settings_form.addRow("棋盘位置", self.calibrate_button)
            link_settings_form.addRow(self.auto_next_checkbox)
            link_settings_form.addRow("下一局识别", self.recognize_next_button)
            link_settings.setLayout(link_settings_form)

            book_settings = QGroupBox("库招设置")
            book_settings_form = QFormLayout()
            book_settings_form.addRow("选招方式", self.book_priority_combo)
            book_settings.setLayout(book_settings_form)

            close_link_settings_button = QPushButton("保存并关闭")
            close_link_settings_button.clicked.connect(self.save_settings_and_close)
            settings_dialog_layout = QVBoxLayout()
            settings_dialog_layout.addWidget(link_settings)
            settings_dialog_layout.addWidget(close_link_settings_button)
            self.settings_dialog = QDialog(self)
            self.settings_dialog.setWindowTitle("连线设置")
            self.settings_dialog.setModal(False)
            self.settings_dialog.setLayout(settings_dialog_layout)

            close_book_settings_button = QPushButton("保存并关闭")
            close_book_settings_button.clicked.connect(self.save_settings_and_close)
            book_settings_dialog_layout = QVBoxLayout()
            book_settings_dialog_layout.addWidget(book_settings)
            book_settings_dialog_layout.addWidget(close_book_settings_button)
            self.book_settings_dialog = QDialog(self)
            self.book_settings_dialog.setWindowTitle("库招设置")
            self.book_settings_dialog.setModal(False)
            self.book_settings_dialog.setLayout(book_settings_dialog_layout)

            self.analysis_box = QPlainTextEdit()
            self.analysis_box.setReadOnly(True)
            self.analysis_box.setPlainText(self.analysis_text)
            analysis_dialog_layout = QVBoxLayout()
            analysis_dialog_layout.addWidget(self.analysis_box)
            close_analysis_button = QPushButton("关闭")
            close_analysis_button.clicked.connect(self.hide_analysis_dialog)
            analysis_dialog_layout.addWidget(close_analysis_button)
            self.analysis_dialog = QDialog(self)
            self.analysis_dialog.setWindowTitle("分析详情")
            self.analysis_dialog.setModal(False)
            self.analysis_dialog.resize(420, 260)
            self.analysis_dialog.setLayout(analysis_dialog_layout)

            board_content = QVBoxLayout()
            board_content.addWidget(self.board_label, 1)
            board_content.addWidget(self.status)

            self.side_panel = self.create_right_workspace()

            content = QHBoxLayout()
            content.addLayout(board_content, 1)
            content.addWidget(self.side_panel)

            central = QWidget()
            central.setLayout(content)
            self.setCentralWidget(central)
            self.setMenuBar(self.create_menu_bar())
            self.addToolBar(self.create_toolbar())
            self.load_settings()
            self.load_game_from_fen(START_FEN)
            self.game_mode = "setup"
            self.show_navigation_panel()
            self.set_status("摆谱：已载入初始棋盘，可点击棋子走子")
            QTimer.singleShot(0, self.enforce_startup_layout)
            QTimer.singleShot(120, self.enforce_startup_layout)

        def load_board_pixmap(self):
            path = self.BOARD_STYLES.get(self.board_style, self.BOARD_STYLES["classic"])
            pixmap = QPixmap(resolve_resource_path(path))
            return pixmap

        def create_icon(self, filename):
            path = resolve_resource_path(f"assets/public_ui/{filename}")
            icon = QIcon(path)
            return icon

        def create_toolbar(self):
            toolbar = QToolBar("工具")
            toolbar.setIconSize(QSize(22, 22))
            toolbar.setMovable(False)

            actions = [
                (self.create_icon("file-plus.png"), "新棋盘", self.start_new_board_mode),
                (self.create_icon("copy.png"), "复制 FEN", self.copy_fen),
                (self.create_icon("paste.png"), "粘贴 FEN", self.paste_fen_from_clipboard),
                (None, None, None),
                (self.create_icon("open.png"), "打开棋谱", self.import_game_record),
                (self.create_icon("save.png"), "保存棋谱", self.save_game_record),
                (None, None, None),
                (self.create_icon("back.png"), "悔棋", self.undo_local_move),
                (self.create_icon("reverse.png"), "翻转棋盘", self.reverse_board),
                (None, None, None),
                (None, "__engine_side_toggles__", None),
                (self.create_icon("lightbulb.png"), "分析详情", self.mode_analysis_detail),
                (self.create_icon("goim.png"), "立即计算", self.mode_single_calculation),
                (self.create_icon("bian.png"), "变招", self.show_opening_book_tab),
                (self.create_icon("cloud.png"), "启用库招", self.toggle_book_switch),
                (None, None, None),
            ]
            for icon, text, callback in actions:
                if text is None:
                    toolbar.addSeparator()
                    continue
                if text == "__engine_side_toggles__":
                    self.add_engine_side_actions(toolbar)
                    continue
                action = QAction(icon, text, self)
                action.setToolTip(text)
                if text == "分析详情":
                    self.analysis_detail_action = action
                    action.setCheckable(True)
                    action.setChecked(self.analysis_detail_enabled)
                    action.toggled.connect(self.toggle_analysis_detail)
                elif text == "启用库招":
                    self.book_switch_action = action
                    action.setCheckable(True)
                    action.setChecked(self.use_open_book)
                    action.toggled.connect(self.toggle_book_switch)
                else:
                    action.triggered.connect(callback)
                toolbar.addAction(action)
                if text == "立即计算":
                    self.calculate_button = action

            self.auto_move_button = QPushButton("自动走棋")
            self.auto_move_button.setIcon(self.create_icon("goim.png"))
            self.auto_move_button.setToolTip("开启自动走棋连线")
            self.auto_move_button.setCheckable(True)
            self.auto_move_button.clicked.connect(lambda checked=False: self.on_link_mode_button("auto", checked))
            toolbar.addWidget(self.auto_move_button)

            self.watch_button = QPushButton("观战")
            self.watch_button.setIcon(self.create_icon("link.png"))
            self.watch_button.setToolTip("开启观战连线，只分析不点击鼠标")
            self.watch_button.setCheckable(True)
            self.watch_button.clicked.connect(lambda checked=False: self.on_link_mode_button("watch", checked))
            toolbar.addWidget(self.watch_button)
            return toolbar

        def add_engine_side_actions(self, toolbar):
            self.engine_black_action = QAction(self.create_icon("robotblack.png"), "引擎执黑", self)
            self.engine_black_action.setToolTip("引擎执黑")
            self.engine_black_action.setCheckable(True)
            self.engine_black_action.toggled.connect(lambda checked: self.toggle_engine_side("black", checked))
            toolbar.addAction(self.engine_black_action)

            self.engine_red_action = QAction(self.create_icon("robotred.png"), "引擎执红", self)
            self.engine_red_action.setToolTip("引擎执红")
            self.engine_red_action.setCheckable(True)
            self.engine_red_action.toggled.connect(lambda checked: self.toggle_engine_side("red", checked))
            toolbar.addAction(self.engine_red_action)

            self.update_engine_side_actions()

        def make_table(self, headers):
            table = QTableWidget(0, len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            return table

        def make_icon_button(self, filename, tooltip=""):
            button = QPushButton()
            button.setIcon(self.create_icon(filename))
            button.setIconSize(QSize(18, 18))
            button.setFixedSize(24, 24)
            button.setFlat(True)
            if tooltip:
                button.setToolTip(tooltip)
            return button

        def create_right_workspace(self):
            panel = QWidget()
            panel.setMinimumWidth(380)
            panel.setMaximumWidth(540)
            panel_layout = QVBoxLayout()
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(0)

            self.side_panel_title = QLabel("导航")
            self.side_panel_title.hide()
            self.side_panel_text = QPlainTextEdit()
            self.side_panel_text.hide()

            engine_box = QFrame()
            engine_box.setFrameShape(QFrame.StyledPanel)
            engine_layout = QVBoxLayout()
            engine_layout.setContentsMargins(0, 0, 0, 0)
            engine_layout.setSpacing(0)
            engine_controls = QHBoxLayout()
            engine_controls.setContentsMargins(4, 2, 4, 2)
            engine_controls.setSpacing(4)
            self.engine_combo = QComboBox()
            self.engine_combo.addItems(["Pikafish"])
            self.engine_combo.setEditable(True)
            self.engine_combo.setFixedHeight(22)
            self.engine_combo.setFixedWidth(95)
            self.thread_combo = QComboBox()
            self.thread_combo.addItems([str(i) for i in range(1, 17)])
            self.thread_combo.setEditable(True)
            self.thread_combo.setFixedHeight(22)
            self.thread_combo.setFixedWidth(60)
            self.hash_combo = QComboBox()
            self.hash_combo.addItems(["16", "32", "64", "128", "256", "512", "1024", "2048", "4096"])
            self.hash_combo.setEditable(True)
            self.hash_combo.setFixedHeight(22)
            self.hash_combo.setFixedWidth(65)
            self.thread_combo.currentTextChanged.connect(self.on_thread_combo_changed)
            self.hash_combo.currentTextChanged.connect(self.on_hash_combo_changed)
            engine_controls.addWidget(self.engine_combo)
            engine_controls.addWidget(QLabel("线程"))
            engine_controls.addWidget(self.thread_combo)
            engine_controls.addWidget(QLabel("Hash"))
            engine_controls.addWidget(self.hash_combo)
            engine_controls.addStretch(1)
            engine_layout.addLayout(engine_controls)
            engine_layout.addWidget(self.engine_settings_panel)
            self.engine_output_list = QListWidget()
            self.engine_output_list.setFrameShape(QFrame.NoFrame)
            self.engine_output_list.setMinimumHeight(150)
            engine_layout.addWidget(self.engine_output_list, 1)
            engine_box.setLayout(engine_layout)

            self.right_tabs = QTabWidget()
            self.right_tabs.setTabPosition(QTabWidget.South)
            record_tab = QWidget()
            record_layout = QVBoxLayout()
            record_layout.setContentsMargins(0, 0, 0, 0)
            record_layout.setSpacing(0)
            self.nav_buttons_widget = QWidget()
            nav_buttons = QHBoxLayout()
            nav_buttons.setContentsMargins(0, 0, 0, 0)
            self.nav_first_button = QPushButton("|<")
            self.nav_prev_button = QPushButton("<")
            self.nav_next_button = QPushButton(">")
            self.nav_last_button = QPushButton(">|")
            self.nav_first_button.setToolTip("退到第一步")
            self.nav_prev_button.setToolTip("上一步")
            self.nav_next_button.setToolTip("下一步")
            self.nav_last_button.setToolTip("快进到最后一步")
            self.nav_first_button.clicked.connect(lambda: self.goto_history_index(0))
            self.nav_prev_button.clicked.connect(lambda: self.goto_history_index(self.history_index - 1))
            self.nav_next_button.clicked.connect(lambda: self.goto_history_index(self.history_index + 1))
            self.nav_last_button.clicked.connect(lambda: self.goto_history_index(len(self.history_fens) - 1))
            for button in (self.nav_first_button, self.nav_prev_button, self.nav_next_button, self.nav_last_button):
                nav_buttons.addWidget(button)
            self.nav_buttons_widget.setLayout(nav_buttons)
            self.nav_buttons_widget.hide()
            record_layout.addWidget(self.nav_buttons_widget)

            self.nav_controls_widget = QWidget()
            nav_controls = QVBoxLayout()
            nav_controls.setContentsMargins(0, 0, 0, 0)
            self.red_ai_checkbox = QCheckBox("红方电脑走棋")
            self.black_ai_checkbox = QCheckBox("黑方电脑走棋")
            self.red_ai_checkbox.setToolTip("勾选后，轮到红方时电脑自动走。")
            self.black_ai_checkbox.setToolTip("勾选后，轮到黑方时电脑自动走。")
            self.red_ai_checkbox.stateChanged.connect(lambda _state: self.on_computer_side_changed())
            self.black_ai_checkbox.stateChanged.connect(lambda _state: self.on_computer_side_changed())
            ai_row = QHBoxLayout()
            ai_row.addWidget(self.red_ai_checkbox)
            ai_row.addWidget(self.black_ai_checkbox)
            nav_controls.addLayout(ai_row)

            search_mode_row = QHBoxLayout()
            self.search_mode_combo = QComboBox()
            self.search_mode_combo.addItems(["时间优先", "深度优先"])
            self.search_mode_combo.setToolTip("时间优先按秒数计算；深度优先按搜索深度计算。")
            self.search_mode_combo.currentTextChanged.connect(lambda _text: self.refresh_navigation_text())
            search_mode_row.addWidget(QLabel("搜索"))
            search_mode_row.addWidget(self.search_mode_combo, 1)
            nav_controls.addLayout(search_mode_row)

            search_value_row = QHBoxLayout()
            self.ai_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.ai_time_spin.setValue(1.0)
            self.ai_time_spin.setToolTip("时间优先时，每步最多思考几秒。")
            self.ai_time_spin.valueChanged.connect(lambda _value: self.refresh_navigation_text())
            self.ai_depth_spin = QSpinBox()
            self.ai_depth_spin.setRange(1, 99)
            self.ai_depth_spin.setValue(8)
            self.ai_depth_spin.setToolTip("深度优先时，搜索到多少层。")
            self.ai_depth_spin.valueChanged.connect(lambda _value: self.refresh_navigation_text())
            search_value_row.addWidget(QLabel("时间"))
            search_value_row.addWidget(self.ai_time_spin)
            search_value_row.addWidget(QLabel("深度"))
            search_value_row.addWidget(self.ai_depth_spin)
            nav_controls.addLayout(search_value_row)
            self.nav_controls_widget.setLayout(nav_controls)
            self.nav_controls_widget.hide()
            record_layout.addWidget(self.nav_controls_widget)

            self.record_table = self.make_table(["序号", "着法", "分数"])
            self.record_table.cellDoubleClicked.connect(lambda row, _col: self.goto_history_index(row))
            self.record_table.setColumnWidth(0, 50)
            self.record_table.setColumnWidth(1, 95)
            self.record_table.setColumnWidth(2, 60)
            self.remark_text = QPlainTextEdit()
            self.remark_text.setReadOnly(False)
            self.remark_text.setPlaceholderText("手动输入注释...")
            self.subrecord_list = QListWidget()

            remark_pane = QWidget()
            remark_layout = QVBoxLayout()
            remark_layout.setContentsMargins(4, 4, 4, 1)
            remark_layout.setSpacing(2)
            remark_layout.addWidget(QLabel("注释:"))
            remark_layout.addWidget(self.remark_text)
            remark_pane.setLayout(remark_layout)

            variation_pane = QWidget()
            variation_layout = QVBoxLayout()
            variation_layout.setContentsMargins(4, 1, 4, 4)
            variation_layout.setSpacing(2)
            variation_layout.addWidget(QLabel("变招列表:"))
            variation_layout.addWidget(self.subrecord_list)
            variation_pane.setLayout(variation_layout)

            right_split = QSplitter(Qt.Vertical)
            right_split.addWidget(remark_pane)
            right_split.addWidget(variation_pane)
            right_split.setStretchFactor(0, 3)
            right_split.setStretchFactor(1, 2)

            record_split = QSplitter(Qt.Horizontal)
            record_split.addWidget(self.record_table)
            record_split.addWidget(right_split)
            record_split.setStretchFactor(0, 3)
            record_split.setStretchFactor(1, 2)
            record_layout.addWidget(record_split, 1)

            manual_toolbar = QFrame()
            manual_toolbar.setFixedHeight(28)
            manual_toolbar.setFrameShape(QFrame.StyledPanel)
            manual_layout = QHBoxLayout()
            manual_layout.setContentsMargins(4, 2, 4, 2)
            manual_layout.setSpacing(2)
            manual_layout.addStretch(1)
            self.manual_delete_button = self.make_icon_button("delete.png", "删除")
            self.manual_score_button = self.make_icon_button("score.png", "评分")
            self.manual_play_button = self.make_icon_button("play.png", "播放")
            self.manual_down_button = self.make_icon_button("down-arrow.png", "下移")
            self.manual_up_button = self.make_icon_button("up-arrow.png", "上移")
            self.nav_first_button = self.make_icon_button("arrow-double-left.png", "退到第一步")
            self.nav_prev_button = self.make_icon_button("arrow-left.png", "上一步")
            self.nav_next_button = self.make_icon_button("arrow-right.png", "下一步")
            self.nav_last_button = self.make_icon_button("arrow-double-right.png", "快进到最后一步")
            self.nav_first_button.clicked.connect(lambda: self.goto_history_index(0))
            self.nav_prev_button.clicked.connect(lambda: self.goto_history_index(self.history_index - 1))
            self.nav_next_button.clicked.connect(lambda: self.goto_history_index(self.history_index + 1))
            self.nav_last_button.clicked.connect(lambda: self.goto_history_index(len(self.history_fens) - 1))
            self.manual_score_button.clicked.connect(self.start_score_chart_analysis)
            for button in (
                self.manual_delete_button,
                self.manual_score_button,
                self.manual_play_button,
                self.manual_down_button,
                self.manual_up_button,
                self.nav_first_button,
                self.nav_prev_button,
                self.nav_next_button,
                self.nav_last_button,
            ):
                manual_layout.addWidget(button)
            manual_toolbar.setLayout(manual_layout)
            record_layout.addWidget(manual_toolbar)
            record_tab.setLayout(record_layout)

            chart_tab = QWidget()
            chart_layout = QVBoxLayout()
            chart_layout.setContentsMargins(0, 0, 0, 0)
            self.score_chart_widget = ScoreChartWidget()
            chart_layout.addWidget(self.score_chart_widget)
            chart_tab.setLayout(chart_layout)

            book_tab = QWidget()
            book_layout = QVBoxLayout()
            book_layout.setContentsMargins(0, 0, 0, 0)
            self.book_table = self.make_table(["招法", "分数", "胜率", "胜", "和", "负", "备注", "来源"])
            self.book_table.cellDoubleClicked.connect(self.book_table_double_clicked)
            book_layout.addWidget(self.book_table)
            book_tab.setLayout(book_layout)

            self.right_tabs.addTab(record_tab, "棋谱")
            self.right_tabs.addTab(chart_tab, "局势图")
            self.right_tabs.addTab(book_tab, "库招")

            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(engine_box)
            splitter.addWidget(self.right_tabs)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([390, 220])
            panel_layout.addWidget(splitter, 1)
            panel.setLayout(panel_layout)
            return panel

        def create_menu_bar(self):
            menu_bar = QMenuBar()

            file_menu = menu_bar.addMenu("文件(&F)")
            exit_action = QAction("退出(&X)", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            position_menu = menu_bar.addMenu("局面(&P)")
            new_position_action = QAction("新局面(&N)", self)
            new_position_action.triggered.connect(self.start_new_board_mode)
            position_menu.addAction(new_position_action)
            edit_position_action = QAction("编辑局面(&E)", self)
            edit_position_action.triggered.connect(self.start_position_setup_mode)
            position_menu.addAction(edit_position_action)
            copy_fen_action = QAction("复制局面 FEN(&C)", self)
            copy_fen_action.triggered.connect(self.copy_fen)
            self.copy_fen_action = copy_fen_action
            position_menu.addAction(copy_fen_action)
            paste_fen_action = QAction("粘贴局面 FEN(&V)", self)
            paste_fen_action.triggered.connect(self.paste_fen_from_clipboard)
            position_menu.addAction(paste_fen_action)
            position_menu.addSeparator()
            copy_image_action = QAction("复制局面图片(&I)", self)
            copy_image_action.triggered.connect(self.copy_board_image)
            position_menu.addAction(copy_image_action)
            paste_image_action = QAction("粘贴局面图片(&A)", self)
            paste_image_action.triggered.connect(self.paste_board_image)
            position_menu.addAction(paste_image_action)
            import_image_action = QAction("图片导入局面(&M)", self)
            import_image_action.triggered.connect(self.import_position_image)
            position_menu.addAction(import_image_action)
            export_image_action = QAction("局面导出图片(&O)", self)
            export_image_action.triggered.connect(self.export_board_image)
            position_menu.addAction(export_image_action)

            manual_menu = menu_bar.addMenu("棋谱(&R)")
            new_manual_action = QAction("新棋谱(&N)", self)
            new_manual_action.triggered.connect(self.start_new_board_mode)
            manual_menu.addAction(new_manual_action)
            open_manual_action = QAction("打开棋谱(&O)", self)
            open_manual_action.triggered.connect(self.import_game_record)
            manual_menu.addAction(open_manual_action)
            save_manual_action = QAction("保存棋谱(&S)", self)
            save_manual_action.triggered.connect(self.save_game_record)
            manual_menu.addAction(save_manual_action)
            save_as_manual_action = QAction("另存为棋谱(&A)", self)
            save_as_manual_action.triggered.connect(self.save_game_record)
            manual_menu.addAction(save_as_manual_action)
            manual_menu.addSeparator()
            self.chess_notation_action = QAction("棋谱管理(&M)", self)
            self.chess_notation_action.setCheckable(True)
            self.chess_notation_action.setChecked(True)
            self.chess_notation_action.triggered.connect(self.toggle_navigation_panel)
            manual_menu.addAction(self.chess_notation_action)
            self.show_tactic_action = QAction("显示分支棋步(&T)", self)
            self.show_tactic_action.setCheckable(True)
            self.show_tactic_action.setChecked(True)
            self.show_tactic_action.triggered.connect(self.show_opening_book_tab)
            manual_menu.addAction(self.show_tactic_action)

            book_menu = menu_bar.addMenu("开局库(&B)")
            import_book_action = QAction("导入库招(&I)", self)
            import_book_action.triggered.connect(self.choose_opening_book)
            book_menu.addAction(import_book_action)
            book_setting_action = QAction("库招设置(&S)", self)
            book_setting_action.triggered.connect(self.show_book_settings_dialog)
            book_menu.addAction(book_setting_action)
            self.book_enabled_menu_action = QAction("启用库招(&E)", self)
            self.book_enabled_menu_action.setCheckable(True)
            self.book_enabled_menu_action.setChecked(self.use_open_book)
            self.book_enabled_menu_action.toggled.connect(self.toggle_book_switch)
            book_menu.addAction(self.book_enabled_menu_action)
            local_book_action = QAction("本地库管理(&L)", self)
            local_book_action.triggered.connect(self.show_opening_book_tab)
            book_menu.addAction(local_book_action)

            self.calculate_button = None

            link_menu = menu_bar.addMenu("连线(&L)")
            settings_action = QAction("设置(&S)", self)
            settings_action.triggered.connect(self.show_settings_dialog)
            link_menu.addAction(settings_action)
            tips_action = QAction("说明(&I)", self)
            tips_action.triggered.connect(self.show_usage_tips)
            link_menu.addAction(tips_action)

            view_menu = menu_bar.addMenu("设置(&S)")
            self.step_tip_action = QAction("棋步提示(&P)", self)
            self.step_tip_action.setCheckable(True)
            self.step_tip_action.setChecked(self.show_step_tip)
            self.step_tip_action.triggered.connect(self.toggle_step_tip)
            view_menu.addAction(self.step_tip_action)
            self.step_sound_action = QAction("走棋音效(&U)", self)
            self.step_sound_action.setCheckable(True)
            self.step_sound_action.setChecked(self.step_sound_enabled)
            self.step_sound_action.triggered.connect(self.toggle_step_sound)
            view_menu.addAction(self.step_sound_action)
            self.show_number_action = QAction("显示线路(&N)", self)
            self.show_number_action.setCheckable(True)
            self.show_number_action.setChecked(self.show_board_numbers)
            self.show_number_action.triggered.connect(self.toggle_board_numbers)
            view_menu.addAction(self.show_number_action)
            self.top_window_action = QAction("窗口置顶(&T)", self)
            self.top_window_action.setCheckable(True)
            self.top_window_action.triggered.connect(self.toggle_top_window)
            view_menu.addAction(self.top_window_action)
            self.show_status_action = QAction("显示状态栏(&B)", self)
            self.show_status_action.setCheckable(True)
            self.show_status_action.setChecked(True)
            self.show_status_action.triggered.connect(lambda checked: self.status.setVisible(checked))
            view_menu.addAction(self.show_status_action)
            view_menu.addSeparator()
            board_style_menu = QMenu("棋盘样式", self)
            classic_action = QAction("原棋盘", self)
            classic_action.setCheckable(True)
            classic_action.setChecked(True)
            classic_action.triggered.connect(lambda: self.set_board_style("classic"))
            board_style_menu.addAction(classic_action)
            view_menu.addMenu(board_style_menu)
            board_size_menu = QMenu("棋盘大小", self)
            for title, size in (("小棋盘", (430, 480)), ("中棋盘", (520, 580)), ("大棋盘", (620, 690)), ("自适应", None)):
                action = QAction(title, self)
                action.triggered.connect(lambda _checked=False, s=size: self.set_board_view_size(s))
                board_size_menu.addAction(action)
            view_menu.addMenu(board_size_menu)

            help_menu = menu_bar.addMenu("帮助(&H)")
            home_action = QAction("主页(&H)", self)
            home_action.triggered.connect(lambda: self.show_text_dialog("主页", "https://github.com/sojourners/public-Xiangqi"))
            help_menu.addAction(home_action)
            instruction_action = QAction("使用说明(&I)", self)
            instruction_action.triggered.connect(lambda: self.show_text_dialog("使用说明", self.usage_manual_text()))
            help_menu.addAction(instruction_action)
            upgrade_action = QAction("查看更新(&U)", self)
            upgrade_action.triggered.connect(lambda: self.show_text_dialog("查看更新", "https://github.com/sojourners/public-Xiangqi/releases"))
            help_menu.addAction(upgrade_action)
            about_action = QAction("关于象棋助手(&A)", self)
            about_action.triggered.connect(lambda: self.show_text_dialog("关于", "象棋助手\n屏幕识别、Pikafish 分析、开局库和连线自动走棋"))
            help_menu.addAction(about_action)
            return menu_bar

        def set_board_style(self, style):
            if style != "classic":
                self.set_status("已保留当前项目原棋盘")
                return
            self.board_style = style
            self.board_pixmap = self.load_board_pixmap()
            self.update_board_display(self.current_grid)
            self.set_status("已切换棋盘样式")

        def toggle_board_numbers(self, checked):
            self.show_board_numbers = bool(checked)
            self.update_board_display(self.current_grid)
            self.set_status("已显示线路" if checked else "已隐藏线路")

        def reverse_board(self):
            self.board_flipped = not self.board_flipped
            self.selected_square = None
            self.legal_target_moves = {}
            self.update_board_display(self.current_grid)
            self.set_status("棋盘已翻转")

        def toggle_step_tip(self, checked):
            self.show_step_tip = bool(checked)
            self.update_board_display(self.current_grid)
            self.set_status("已开启棋步提示" if checked else "已关闭棋步提示")

        def toggle_step_sound(self, checked):
            self.step_sound_enabled = bool(checked)
            self.set_status("已开启走棋音效" if checked else "已关闭走棋音效")

        def toggle_top_window(self, checked):
            self.setWindowFlag(Qt.WindowStaysOnTopHint, bool(checked))
            self.show()
            self.set_status("已置顶窗口" if checked else "已取消置顶")

        def set_board_view_size(self, size):
            if size is None:
                self.board_label.setMinimumSize(430, 480)
                self.set_status("棋盘大小：自适应")
            else:
                width, height = size
                self.board_label.setMinimumSize(width, height)
                self.resize(max(self.width(), width + 390), max(self.height(), height + 90))
                self.set_status("棋盘大小已调整")
            self.update_board_display(self.current_grid)

        def on_thread_combo_changed(self, text):
            try:
                threads = max(1, min(128, int(float(text))))
            except (TypeError, ValueError):
                return
            was_blocked = self.engine_threads_spin.blockSignals(True)
            self.engine_threads_spin.setValue(threads)
            self.engine_threads_spin.blockSignals(was_blocked)
            self.on_engine_settings_changed(f"引擎线程数已设置为 {threads}")
            self.refresh_navigation_text()

        def on_engine_threads_spin_changed(self, value):
            self.sync_combo_text(self.thread_combo, str(value))
            self.on_engine_settings_changed(f"引擎线程数已设置为 {value}")
            self.refresh_navigation_text()

        def on_hash_combo_changed(self, text):
            try:
                hash_mb = max(1, min(4096, int(float(text))))
            except (TypeError, ValueError):
                return
            was_blocked = self.engine_hash_spin.blockSignals(True)
            self.engine_hash_spin.setValue(hash_mb)
            self.engine_hash_spin.blockSignals(was_blocked)
            self.on_engine_settings_changed(f"引擎 Hash 已设置为 {hash_mb} MB")
            self.refresh_navigation_text()

        def on_engine_hash_spin_changed(self, value):
            self.sync_combo_text(self.hash_combo, str(value))
            self.on_engine_settings_changed(f"引擎 Hash 已设置为 {value} MB")
            self.refresh_navigation_text()

        def sync_combo_text(self, combo, text):
            if combo.findText(text) < 0:
                combo.addItem(text)
            was_blocked = combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(was_blocked)

        def set_spin_value_silent(self, spin, value):
            was_blocked = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(was_blocked)

        def on_engine_settings_changed(self, status_text=None):
            if getattr(self, "_loading_settings", False) or getattr(self, "_saving_settings", False):
                return
            self.save_settings(silent=True)
            if status_text:
                self.set_status(status_text)
            self.refresh_navigation_text()

        def set_auto_move_enabled(self, enabled, save=False):
            self.auto_move_enabled = bool(enabled)
            self.update_link_mode_switches()
            if save:
                self.save_settings(silent=True)

        def on_link_mode_button(self, mode, checked):
            wants_auto_move = mode == "auto"
            if checked:
                self.link_mode_active = True
                self.set_auto_move_enabled(wants_auto_move, save=True)
                self.update_link_mode_switches(True, True)
                if self.helper_running():
                    self.set_status(f"{'自动走棋' if wants_auto_move else '观战'}已开启")
                    return
                self.start_helper()
                return

            if self.link_mode_active and self.auto_move_enabled == wants_auto_move:
                self.link_mode_active = False
                self.update_link_mode_switches(False, True)
                if self.helper_running() or self.app_worker is not None:
                    self.stop_helper()
                else:
                    self.set_status("连线已关闭")
                return

            self.update_link_mode_switches()

        def show_opening_book_tab(self, checked=True):
            self.side_panel_title.setText("开局库")
            self.side_panel.show()
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentIndex(2)
            self.load_opening_book_panel()

        def show_score_tab(self):
            self.side_panel_title.setText("局势图")
            self.side_panel.show()
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentIndex(1)
            self.refresh_right_workspace()
            self.set_status("已显示局势图")

        def toggle_book_switch(self, checked=None):
            self.use_open_book = (not self.use_open_book) if checked is None else bool(checked)
            if self.use_open_book:
                self.ensure_auto_move_enabled()
            self.update_book_switch_actions()
            self.refresh_navigation_text()
            self.refresh_right_workspace()
            try:
                update_config({"use_open_book": self.use_open_book})
            except Exception:
                pass
            self.set_status("已启用库招" if self.use_open_book else "已关闭库招")
            if self.use_open_book and self.is_computer_turn():
                self.request_ai_move()

        def update_book_switch_actions(self):
            for action_name in ("book_switch_action", "book_enabled_menu_action"):
                if not hasattr(self, action_name):
                    continue
                action = getattr(self, action_name)
                was_blocked = action.blockSignals(True)
                action.setChecked(self.use_open_book)
                action.blockSignals(was_blocked)

        def on_book_priority_changed(self, text):
            self.book_move_priority = normalize_book_priority(text)
            self.refresh_right_workspace()
            try:
                update_config({"book_move_priority": self.book_move_priority})
            except Exception:
                pass
            if self.use_open_book and self.is_computer_turn():
                self.request_ai_move()

        def show_book_settings_dialog(self):
            self.book_settings_dialog.show()
            self.book_settings_dialog.raise_()
            self.book_settings_dialog.activateWindow()

        def ensure_auto_move_enabled(self):
            if self.auto_move_enabled:
                return
            self.set_auto_move_enabled(True, save=True)

        def usage_manual_text(self):
            return (
                "1. 局面菜单可新建、编辑、复制/粘贴 FEN、导入/导出图片。\n"
                "2. 棋谱菜单可打开和保存棋谱，右侧棋谱页可复盘与悔棋。\n"
                "3. 右侧引擎栏可设置线程、Hash、随机搜索范围。\n"
                "4. 连线菜单保留原项目的校准、识别、自动走棋流程。"
            )

        def refresh_right_workspace(self):
            if not hasattr(self, "right_tabs"):
                return
            self.refresh_engine_output()
            self.refresh_record_table()
            self.refresh_book_table()
            self.refresh_score_chart()
            if self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())

        def refresh_score_chart(self):
            if hasattr(self, "score_chart_widget"):
                self.score_chart_widget.set_score_records(self.score_chart_records())

        def refresh_engine_output(self):
            if not hasattr(self, "engine_output_list"):
                return
            lines = [line for line in (self.analysis_text or "").splitlines() if line.strip()]
            self.engine_output_list.clear()
            for line in lines[:60]:
                self.engine_output_list.addItem(line)

        def ensure_history_scores(self):
            if not hasattr(self, "history_scores"):
                self.history_scores = []
            target = len(self.history_fens)
            if len(self.history_scores) < target:
                self.history_scores.extend([None] * (target - len(self.history_scores)))
            elif len(self.history_scores) > target:
                self.history_scores = self.history_scores[:target]

        def history_score_text(self, index):
            self.ensure_history_scores()
            if index < 0 or index >= len(self.history_scores):
                return ""
            score = self.history_scores[index]
            return "" if score is None else str(score)

        def refresh_record_table(self):
            if not hasattr(self, "record_table"):
                return
            self.ensure_history_scores()
            self.record_table.blockSignals(True)
            row_count = max(1, len(self.history_fens))
            self.record_table.setRowCount(row_count)
            for index in range(row_count):
                if index == 0:
                    move_text = "开始局面"
                else:
                    move = self.history_moves[index - 1] if index - 1 < len(self.history_moves) else ""
                    fen = self.history_fens[index - 1] if index - 1 < len(self.history_fens) else ""
                    move_text = self.format_chinese_move(fen, move) if move else ""
                if index == self.history_index:
                    move_text = "* " + move_text
                score = self.history_score_text(index)
                for col, value in enumerate((str(index), move_text, score)):
                    self.record_table.setItem(index, col, QTableWidgetItem(value))
            self.record_table.blockSignals(False)
            self.subrecord_list.clear()
            branches = self.query_xqbook(self.current_fen) if self.current_fen else []
            for item in branches[:30]:
                move_text = self.format_chinese_move(self.current_fen, item["uci"])
                self.subrecord_list.addItem(f"{move_text}  分数 {item['score']}")

        def refresh_book_table(self):
            if not hasattr(self, "book_table"):
                return
            branches = self.query_xqbook(self.current_fen) if self.current_fen else []
            self.book_table.blockSignals(True)
            self.book_table.setRowCount(len(branches))
            for row, item in enumerate(branches):
                total = max(1, int(item["win"] or 0) + int(item["draw"] or 0) + int(item["lost"] or 0))
                win_rate = f"{(int(item['win'] or 0) / total * 100):.1f}%"
                move_text = self.format_chinese_move(self.current_fen, item["uci"])
                values = (
                    move_text,
                    str(item["score"]),
                    win_rate,
                    str(item["win"]),
                    str(item["draw"]),
                    str(item["lost"]),
                    item["memo"],
                    "本地",
                )
                for col, value in enumerate(values):
                    table_item = QTableWidgetItem(value)
                    if col == 0:
                        table_item.setData(Qt.UserRole, item["uci"])
                    self.book_table.setItem(row, col, table_item)
            self.book_table.blockSignals(False)

        def book_table_double_clicked(self, row, _col):
            item = self.book_table.item(row, 0)
            move = item.data(Qt.UserRole) if item is not None else None
            if not move:
                return
            if self.game_board is not None and is_legal_move(self.game_board, self.game_side, move):
                self.play_local_move(move)
            else:
                self.update_move_display(move, "库招")
                self.set_status(f"库招：{self.format_chinese_move(self.current_fen, move)}")

        def show_analysis_dialog(self):
            self.show_analysis_detail_panel()

        def hide_analysis_dialog(self):
            self.analysis_dialog.hide()
            if hasattr(self, "pv_analysis_action"):
                self.pv_analysis_action.setChecked(False)

        def toggle_pv_analysis_window(self, checked):
            self.toggle_analysis_detail(checked)

        def toggle_analysis_detail(self, checked):
            self.analysis_detail_enabled = bool(checked)
            self.update_analysis_detail_action()
            if self.analysis_dialog.isVisible():
                self.analysis_dialog.hide()
            if self.analysis_detail_enabled:
                self.show_analysis_detail_panel()
                self.update_analysis_display("")
                self.start_analysis_detail_analysis()
                self.set_status("分析详情已开启")
            else:
                self.analysis_detail_restart_timer.stop()
                self.engine_analysis_request_id += 1
                self.analysis_detail_fen = ""
                self.refresh_right_workspace()
                self.set_status("分析详情已关闭")

        def update_analysis_detail_action(self):
            for action_name in ("analysis_detail_action", "pv_analysis_action"):
                if not hasattr(self, action_name):
                    continue
                action = getattr(self, action_name)
                was_blocked = action.blockSignals(True)
                action.setChecked(self.analysis_detail_enabled)
                action.blockSignals(was_blocked)

        def show_analysis_detail_panel(self):
            self.side_panel_title.setText("分析详情")
            self.side_panel.show()
            self.setMinimumWidth(880)
            self.setMinimumHeight(580)
            self.resize(max(self.width(), 980), max(self.height(), 580))
            self.refresh_right_workspace()

        def toggle_window_panel(self, name, checked):
            if checked:
                self.show_side_panel(name, f"{name}窗口已打开。\n后续可以在这里继续接入完整{name}内容。")
                self.set_status(f"已显示{name}")
            else:
                self.hide_side_panel_if_title(name)
                self.set_status(f"已隐藏{name}")

        def show_side_panel(self, title, text):
            self.side_panel_title.setText(title)
            if title != "局势图":
                self.side_panel_text.setPlainText(text)
            if hasattr(self, "right_tabs"):
                if title == "导航":
                    self.right_tabs.setCurrentIndex(0)
                elif title == "局势图":
                    self.right_tabs.setCurrentIndex(1)
                    self.refresh_score_chart()
                elif title == "开局库":
                    self.right_tabs.setCurrentIndex(2)
                else:
                    self.right_tabs.setCurrentIndex(0)
            self.refresh_navigation_buttons()
            self.side_panel.show()
            self.setMinimumWidth(880)
            self.setMinimumHeight(580)
            target_height = max(self.height(), 580)
            if self.width() < 980 or self.height() < target_height:
                self.resize(max(self.width(), 980), target_height)
            self.update_board_display(self.current_grid)
            self.refresh_right_workspace()

        def enforce_startup_layout(self):
            self.resize(max(self.width(), 820), max(self.height(), 580))
            self.update_board_display(self.current_grid)

        def hide_side_panel_if_title(self, title):
            if self.side_panel_title.text() == title:
                self.side_panel.hide()
                self.setMinimumWidth(520)

        def toggle_opening_book_panel(self, checked):
            if checked:
                self.show_side_panel("开局库", "")
                self.load_opening_book_panel()
                self.set_status("已显示开局库")
            else:
                self.hide_side_panel_if_title("开局库")
                self.set_status("已隐藏开局库")

        def toggle_score_chart_panel(self, checked):
            if checked:
                self.show_side_panel("局势图", "")
                self.set_status("已显示局势图")
            else:
                self.hide_side_panel_if_title("局势图")
                self.set_status("已隐藏局势图")

        def toggle_navigation_panel(self, checked):
            if checked:
                self.show_navigation_panel()
                self.set_status("已显示导航")
            else:
                self.hide_side_panel_if_title("导航")
                self.set_status("已隐藏导航")

        def choose_opening_book(self):
            start_dir = os.path.dirname(self.opening_book_path) if self.opening_book_path else r"C:\Users\Tom\Desktop"
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择开局库",
                start_dir,
                "开局库文件 (*.xqb);;所有文件 (*.*)",
            )
            if not path:
                return
            self.opening_book_path = path
            self.use_open_book = True
            self.update_book_switch_actions()
            try:
                update_config({
                    "opening_book_path": path,
                    "use_open_book": self.use_open_book,
                })
            except Exception:
                pass
            self.side_panel_title.setText("开局库")
            self.side_panel.show()
            if hasattr(self, "right_tabs"):
                self.right_tabs.setCurrentIndex(2)
            self.setMinimumWidth(720)
            if self.width() < 820:
                self.resize(820, self.height())
            self.load_opening_book_panel()
            self.set_status("已导入库招")

        def find_opening_book_path(self):
            return find_opening_book_path(load_config())

        def load_opening_book_panel(self):
            if not self.opening_book_path:
                self.side_panel_text.setPlainText(
                    "没有找到开局库文件。\n\n请把 book.xqb 放到程序文件夹，也就是 XiangqiHelper.exe 旁边。"
                )
                self.refresh_right_workspace()
                return

            text = [
                "开局库文件：",
                self.opening_book_path,
                "",
                f"启用库招：{'是' if self.use_open_book else '否'}",
                f"选招方式：{book_priority_label(self.book_move_priority)}",
                "",
            ]
            try:
                size_mb = os.path.getsize(self.opening_book_path) / (1024 * 1024)
                text.append(f"大小：{size_mb:.1f} MB")
            except OSError:
                pass

            try:
                uri = pathlib.Path(self.opening_book_path).as_uri() + "?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
                try:
                    row = conn.execute("select count(*) from book").fetchone()
                    count = row[0] if row else 0
                    text.append(f"记录数：{count:,}")
                finally:
                    conn.close()
            except Exception as e:
                text.append("")
                text.append(f"数据库读取失败：{e}")

            text.extend([
                "",
                "当前局面：",
                self.current_fen or "还没有读取到棋盘局面。",
                "",
            ])
            moves = self.query_xqbook(self.current_fen) if self.current_fen else []
            if moves:
                text.append("开局走法：")
                for index, item in enumerate(moves[:20], 1):
                    stats = f"胜{item['win']} 和{item['draw']} 负{item['lost']}"
                    memo = f"  {item['memo']}" if item["memo"] else ""
                    move_text = self.format_chinese_move(self.current_fen, item["uci"])
                    text.append(
                        f"{index}. {move_text}  分数 {item['score']}  {stats}{memo}"
                    )
            else:
                text.extend([
                    "开局走法：",
                    "当前局面在开局库里没有找到记录。",
                ])
            self.side_panel_text.setPlainText("\n".join(text))
            self.refresh_right_workspace()

        def xqbook_fen_to_key(self, fen):
            placement, side = fen.strip().split()[:2]
            rows = placement.count("/") + 1
            first_row = placement.split("/", 1)[0]
            cols = sum(int(ch) if ch.isdigit() else 1 for ch in first_row)
            size = rows * cols
            turn_is_red = side != "b"
            piece_codes = {
                "X": 0, "x": 0,
                "R": 1, "N": 2, "B": 3, "A": 4, "K": 5, "C": 6, "P": 7,
                "r": 9, "n": 10, "b": 11, "a": 12, "k": 13, "c": 14, "p": 15,
            }
            cells = [-1] * size
            index = 0
            for ch in placement:
                if ch == "/":
                    continue
                if ch.isdigit():
                    index += int(ch)
                    continue
                lookup = ch if turn_is_red else ch.swapcase()
                cells[index] = piece_codes.get(lookup, -1)
                index += 1

            mirror_ud = False
            if not turn_is_red:
                for row in range(rows // 2):
                    for col in range(cols):
                        a = row * cols + col
                        b = (rows - 1 - row) * cols + (cols - 1 - col)
                        cells[a], cells[b] = cells[b], cells[a]
                mirror_ud = True

            mirror_lr = False
            lr_done = False
            for row in range(rows):
                if lr_done:
                    break
                for col in range(cols // 2):
                    a = row * cols + col
                    b = row * cols + (cols - 1 - col)
                    if cells[a] != cells[b]:
                        mirror_lr = cells[b] > cells[a]
                        lr_done = True
                        break
            if mirror_lr:
                for row in range(rows):
                    for col in range(cols // 2):
                        a = row * cols + col
                        b = row * cols + (cols - 1 - col)
                        cells[a], cells[b] = cells[b], cells[a]

            out = bytearray()
            buffer = 0
            bits = 0
            buffer_bits = 32
            code_bits = 4
            for idx, value in enumerate(cells):
                if value == -1:
                    bits += 1
                else:
                    buffer |= 1 << (buffer_bits - bits - 1)
                    buffer |= value << (buffer_bits - bits - 1 - code_bits)
                    bits += 1 + code_bits
                next_value = cells[idx + 1] if idx + 1 < len(cells) else None
                need_bits = 1 if next_value == -1 else code_bits + 1
                if idx == len(cells) - 1 or buffer_bits - bits < need_bits:
                    threshold = 1 if idx == len(cells) - 1 else 8
                    while bits >= threshold:
                        out.append((buffer >> (buffer_bits - 8)) & 0xFF)
                        buffer = (buffer << 8) & 0xFFFFFFFF
                        bits -= 8
            return bytes(out), mirror_ud, mirror_lr, rows, cols

        def xqbook_mirror_move(self, move, mirror_ud, mirror_lr, rows, cols):
            from_row = move >> 12
            from_col = (move >> 8) & 0xF
            to_row = (move >> 4) & 0xF
            to_col = move & 0xF
            if mirror_ud:
                from_row = rows - 1 - from_row
                to_row = rows - 1 - to_row
                from_col = cols - 1 - from_col
                to_col = cols - 1 - to_col
            if mirror_lr:
                from_col = cols - 1 - from_col
                to_col = cols - 1 - to_col
            return from_row, from_col, to_row, to_col

        def query_xqbook(self, fen):
            return query_xqbook_file(self.opening_book_path, fen, self.book_move_priority)

        def score_chart_value(self, value):
            return clamp_score_chart_value(value)

        def score_chart_records(self):
            self.ensure_history_scores()
            records = []
            for index, score in enumerate(self.history_scores):
                if score is None:
                    continue
                records.append((index, self.score_chart_value(score)))
            return records

        def score_chart_text(self):
            lines = ["局势图 (-1000 ~ +1000)", ""]
            for index, score in self.score_chart_records():
                marker = " <- 当前" if index == self.history_index else ""
                lines.append(f"{index:>3}. {score:+d}{marker}")
            return "\n".join(lines).rstrip()

        def selected_score_chart_search(self):
            if hasattr(self, "search_mode_combo") and self.search_mode_combo.currentText() == "深度优先":
                return None, self.ai_depth_spin.value()
            if hasattr(self, "ai_time_spin"):
                return int(round(self.ai_time_spin.value() * 1000)), None
            return self.selected_ai_search()

        def cancel_score_chart_analysis(self):
            self.score_analysis_request_id += 1
            if hasattr(self, "manual_score_button"):
                self.manual_score_button.setEnabled(True)

        def start_score_chart_analysis(self, checked=False, start_index=1, only_missing=True, auto=False):
            if not self.history_fens:
                if not auto:
                    self.set_status("还没有可分析的棋谱")
                return
            self.ensure_history_scores()
            start_index = max(1, int(start_index or 1))
            targets = []
            for index, fen in enumerate(self.history_fens):
                if index < start_index:
                    continue
                if only_missing and self.history_scores[index] is not None:
                    continue
                targets.append((index, fen))
            if not targets:
                self.refresh_right_workspace()
                if not auto:
                    self.set_status("棋谱局势图分数已完整")
                return

            self.score_analysis_request_id += 1
            request_id = self.score_analysis_request_id
            movetime, depth = self.selected_score_chart_search()
            if hasattr(self, "manual_score_button"):
                self.manual_score_button.setEnabled(False)
            self.set_status(f"正在分析棋谱局势图：0/{len(targets)}")
            self.score_analysis_thread = threading.Thread(
                target=self.run_score_chart_analysis,
                args=(request_id, targets, movetime, depth),
                daemon=True,
            )
            self.score_analysis_thread.start()

        def score_analysis_cancelled(self, request_id):
            return self.closing or request_id != self.score_analysis_request_id

        def current_engine_chart_score(self, engine, fen):
            score = (getattr(engine, "pv_scores", {}) or {}).get(
                1,
                getattr(engine, "eval_score", ""),
            )
            score_info = java_oriented_engine_score(score, fen, False)
            if score_info is None:
                return None
            return int(score_info["chart_score"])

        def run_score_chart_analysis(self, request_id, targets, movetime, depth):
            engine = None
            completed = 0
            total = len(targets)
            try:
                engine = Engine()
                for position, (index, fen) in enumerate(targets, 1):
                    if self.score_analysis_cancelled(request_id):
                        break
                    self.signals.status.emit(f"正在分析棋谱局势图：{position}/{total}")
                    engine.start_search(
                        fen,
                        movetime=movetime,
                        depth=depth,
                        should_cancel=lambda rid=request_id: self.score_analysis_cancelled(rid),
                    )
                    if movetime is not None:
                        deadline = time.time() + max(2.0, movetime / 1000.0 + 2.0)
                    else:
                        deadline = time.time() + max(8.0, float(depth or 8) * 1.5)
                    last_score = None
                    while engine.bestmove is None and time.time() < deadline:
                        if self.score_analysis_cancelled(request_id):
                            engine.send_cmd("stop")
                            break
                        score = self.current_engine_chart_score(engine, fen)
                        if score is not None and score != last_score:
                            last_score = score
                            self.signals.history_score.emit(request_id, index, score)
                        time.sleep(0.12)
                    if self.score_analysis_cancelled(request_id):
                        break
                    score = self.current_engine_chart_score(engine, fen)
                    if score is not None:
                        self.signals.history_score.emit(request_id, index, score)
                    completed += 1
            except InterruptedError:
                if engine is not None:
                    engine.send_cmd("stop")
            except Exception as e:
                if not self.score_analysis_cancelled(request_id):
                    self.signals.status.emit(f"棋谱局势图分析失败：{e}")
            finally:
                if engine is not None:
                    try:
                        engine.quit()
                    except Exception:
                        pass
                self.signals.score_analysis_done.emit(request_id, completed, total)

        def format_chinese_move(self, fen, move):
            try:
                board, _ = fen_to_board(fen)
                return move_to_chinese(board, move)
            except Exception:
                return move

        def show_navigation_panel(self):
            self.show_side_panel("导航", self.navigation_text())

        def refresh_navigation_text(self):
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            self.refresh_right_workspace()

        def refresh_navigation_buttons(self):
            total = max(0, len(self.history_fens) - 1)
            can_go_back = self.history_index > 0
            can_go_forward = self.history_index < total
            self.nav_first_button.setEnabled(can_go_back)
            self.nav_prev_button.setEnabled(can_go_back)
            self.nav_next_button.setEnabled(can_go_forward)
            self.nav_last_button.setEnabled(can_go_forward)
            self.refresh_right_workspace()

        def navigation_text(self):
            lines = []
            total = max(0, len(self.history_fens) - 1)
            lines.append(f"当前步数：{self.history_index} / {total}")
            lines.append(f"当前走棋：{'红方' if self.game_side == 'red' else '黑方'}")
            search_value = self.engine_search_summary()
            auto_sides = []
            if self.red_ai_checkbox.isChecked():
                auto_sides.append("红方")
            if self.black_ai_checkbox.isChecked():
                auto_sides.append("黑方")
            lines.append(f"电脑走棋：{('、'.join(auto_sides) if auto_sides else '未开启')}    {search_value}")
            lines.append(
                f"库招：{'启用' if self.use_open_book else '关闭'}    {book_priority_label(self.book_move_priority)}"
            )
            lines.append("")
            lines.append("记录：")
            if self.online_record_text:
                lines.extend(self.online_record_text.splitlines())
            elif not self.history_moves:
                lines.append("还没有走法。")
            else:
                for round_index in range(0, len(self.history_moves), 2):
                    red_step = round_index + 1
                    black_step = round_index + 2
                    red_fen = self.history_fens[red_step - 1] if red_step - 1 < len(self.history_fens) else ""
                    red_move = self.format_chinese_move(red_fen, self.history_moves[round_index])
                    red_marker = " <- 当前" if red_step == self.history_index else ""
                    if black_step <= len(self.history_moves):
                        black_fen = self.history_fens[black_step - 1] if black_step - 1 < len(self.history_fens) else ""
                        black_move = self.format_chinese_move(black_fen, self.history_moves[round_index + 1])
                        black_marker = " <- 当前" if black_step == self.history_index else ""
                        lines.append(
                            f"第{round_index // 2 + 1}回合  {red_move}{red_marker}    {black_move}{black_marker}"
                        )
                    else:
                        lines.append(f"第{round_index // 2 + 1}回合  {red_move}{red_marker}")
            lines.append("")
            lines.append("分支走法：")
            branches = self.query_xqbook(self.current_fen) if self.current_fen else []
            if branches:
                for index, item in enumerate(branches[:20], 1):
                    stats = f"胜{item['win']} 和{item['draw']} 负{item['lost']}"
                    move_text = self.format_chinese_move(self.current_fen, item["uci"])
                    lines.append(f"{index}. {move_text}  分数 {item['score']}  {stats}")
            else:
                lines.append("当前局面没有开局库分支。")
            return "\n".join(lines)

        def goto_history_index(self, index):
            if not self.history_fens:
                return
            index = max(0, min(index, len(self.history_fens) - 1))
            self.history_index = index
            fen = self.history_fens[index]
            board, side = fen_to_board(fen)
            self.game_board = board
            self.game_side = side
            self.current_fen = fen
            self.current_grid = [row[:] for row in board]
            self.current_move = self.history_moves[index - 1] if index > 0 and index - 1 < len(self.history_moves) else ""
            self.selected_square = None
            self.legal_target_moves = {}
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            if self.side_panel.isVisible():
                if self.side_panel_title.text() == "导航":
                    self.side_panel_text.setPlainText(self.navigation_text())
                elif self.side_panel_title.text() == "开局库":
                    self.load_opening_book_panel()
                elif self.side_panel_title.text() == "局势图":
                    self.refresh_score_chart()
            self.set_status(f"已跳到第 {self.history_index} 步")
            self.schedule_analysis_detail_restart()

        def undo_local_move(self):
            if not self.history_fens or self.history_index <= 0:
                self.set_status("没有可悔的棋步")
                return
            self.goto_history_index(self.history_index - 1)

        def show_settings_dialog(self):
            self.settings_dialog.setWindowTitle("连线设置")
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()

        def show_engine_settings_dialog(self):
            self.engine_settings_dialog.show()
            self.engine_settings_dialog.raise_()
            self.engine_settings_dialog.activateWindow()

        def hide_settings_dialog(self):
            if self.settings_dialog is not None:
                self.settings_dialog.hide()
            if self.engine_settings_dialog is not None:
                self.engine_settings_dialog.hide()
            if self.book_settings_dialog is not None:
                self.book_settings_dialog.hide()

        def save_settings_and_close(self):
            self.save_settings(silent=True)
            self.hide_settings_dialog()
            self.set_status("设置已保存")

        def mode_single_calculation(self):
            self.set_auto_move_enabled(False, save=True)
            if self.current_fen:
                self.analyze_current_position("单次计算模式：正在分析当前局面")
            else:
                self.calculate_now()

        def mode_screen_monitor(self):
            self.link_mode_active = True
            self.set_auto_move_enabled(False, save=True)
            self.update_link_mode_switches(True, True)
            if not self.helper_running():
                self.start_helper()
            else:
                self.set_status("观战已经在运行")

        def mode_analysis_only(self):
            self.on_link_mode_button("watch", True)
            self.set_status("观战模式：不会自动点击棋子")

        def mode_auto_move(self):
            self.on_link_mode_button("auto", True)
            self.set_status("自动走棋模式：识别稳定后会自动点击推荐走法")

        def mode_analysis_detail(self, checked=True):
            self.toggle_analysis_detail(checked)

        def mode_setup(self):
            self.show_settings_dialog()
            self.set_status("校准设置模式：可校准棋盘和调整选项")

        def start_human_game(self):
            self.load_game_from_fen(START_FEN)
            self.game_mode = "human"
            self.player_side = "both"
            self.board_flipped = False
            self.update_board_display(self.current_grid)
            self.set_status("人人对弈：红方走")

        def start_ai_game(self, side):
            self.load_game_from_fen(START_FEN)
            self.game_mode = "ai"
            self.player_side = side
            self.update_board_display(self.current_grid)
            self.set_status(f"人机对弈：你执{'红' if side == 'red' else '黑'}")
            if self.game_side != self.player_side:
                self.request_ai_move()

        def toggle_engine_side(self, side, checked):
            if self.game_board is None:
                self.load_game_from_fen(self.current_fen or START_FEN)

            checkbox = self.red_ai_checkbox if side == "red" else self.black_ai_checkbox
            if checkbox.isChecked() != checked:
                checkbox.blockSignals(True)
                checkbox.setChecked(checked)
                checkbox.blockSignals(False)
            if checked:
                self.ensure_auto_move_enabled()

            self.game_mode = "setup"
            self.player_side = "both"
            self.update_engine_side_actions()
            self.refresh_navigation_text()

            side_text = "红" if side == "red" else "黑"
            state_text = "开启" if checked else "关闭"
            self.set_status(f"引擎执{side_text}已{state_text}")
            if checked and self.is_computer_turn():
                self.request_ai_move()

        def update_engine_side_actions(self):
            pairs = (
                ("engine_red_action", self.red_ai_checkbox.isChecked()),
                ("engine_black_action", self.black_ai_checkbox.isChecked()),
            )
            for action_name, checked in pairs:
                if not hasattr(self, action_name):
                    continue
                action = getattr(self, action_name)
                was_blocked = action.blockSignals(True)
                action.setChecked(checked)
                action.blockSignals(was_blocked)

        def start_trainer_mode(self):
            self.load_game_from_fen(self.current_fen or START_FEN)
            self.game_mode = "trainer"
            self.show_analysis_dialog()
            self.calculate_local_position("训练模式：先看提示，想好后再走")

        def start_position_setup_mode(self):
            self.load_empty_board()
            self.game_mode = "setup"
            self.show_navigation_panel()
            self.set_status("摆谱：空棋盘，右键格子添加或清空棋子")

        def start_new_board_mode(self):
            self.load_game_from_fen(START_FEN)
            self.game_mode = "setup"
            self.show_navigation_panel()
            self.set_status("新棋盘：已载入初始棋盘，可点击棋子走子")

        def import_game_record(self):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "导入棋谱",
                "",
                "棋谱文件 (*.pgn *.pgns *.txt);;所有文件 (*.*)",
            )
            if not path:
                return
            try:
                self.cancel_score_chart_analysis()
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
                parsed = parse_pgn(text)
                self.load_game_from_fen(parsed["fen"])
                self.game_mode = "record"
                self.history_fens = parsed.get("fens", [parsed["fen"]])
                self.history_moves = [item["uci"] for item in parsed.get("moves", [])]
                self.history_scores = [0] + [None] * max(0, len(self.history_fens) - 1)
                self.history_index = len(self.history_fens) - 1
                self.current_move = self.history_moves[-1] if self.history_moves else ""
                if self.history_fens:
                    self.current_fen = self.history_fens[self.history_index]
                    self.game_board, self.game_side = fen_to_board(self.current_fen)
                    self.current_grid = [row[:] for row in self.game_board]
                self.refresh_navigation_buttons()
                self.update_board_display(self.current_grid)
                self.show_text_dialog(
                    "导入棋谱",
                    f"已导入：{os.path.basename(path)}\n"
                    f"步数：{len(parsed['moves'])}\n"
                    f"当前局面：{parsed['fen']}",
                )
                self.set_status(f"已导入棋谱，共 {len(parsed['moves'])} 步")
                if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                    self.side_panel_text.setPlainText(self.navigation_text())
                self.start_score_chart_analysis(auto=True)
            except Exception as e:
                self.set_status(f"导入棋谱失败：{e}")

        def save_game_record(self):
            if self.app_worker is not None:
                path = self.app_worker.save_current_record("manual")
            else:
                helper = App.__new__(App)
                helper.record_saved = False
                helper.record_moves = []
                helper.record_start_fen = None
                helper.history_moves = self.history_moves
                helper.history_fens = self.history_fens
                path = App.save_current_record(helper, "manual")
            if path:
                self.set_status(f"棋谱已保存：{path}")
            else:
                self.set_status("没有可保存的棋谱")

        def import_fen_position(self):
            text, ok = QInputDialog.getText(
                self,
                "导入 FEN",
                "请输入 FEN：",
                text=self.current_fen or START_FEN,
            )
            if not ok:
                return
            fen = text.strip()
            if not fen:
                self.set_status("FEN 不能为空")
                return
            try:
                validate_fen(fen)
                self.load_game_from_fen(fen)
                self.game_mode = "setup"
                self.show_navigation_panel()
                self.set_status("已导入 FEN")
            except Exception as e:
                self.set_status(f"导入 FEN 失败：{e}")

        def start_pgn_mode(self):
            self.import_game_record()

        def start_online_mode(self):
            self.mode_screen_monitor()

        def load_game_from_fen(self, fen):
            self.cancel_score_chart_analysis()
            try:
                board, side = fen_to_board(fen)
            except Exception:
                board, side = fen_to_board(START_FEN)
            self.game_board = board
            self.game_side = side
            self.current_fen = board_to_fen(board, side)
            self.current_grid = [row[:] for row in board]
            self.current_move = ""
            self.selected_square = None
            self.legal_target_moves = {}
            self.move_history = []
            self.history_fens = [self.current_fen]
            self.history_moves = []
            self.history_scores = [0]
            self.history_index = 0
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            if self.side_panel.isVisible() and self.side_panel_title.text() == "开局库":
                self.load_opening_book_panel()
            self.schedule_analysis_detail_restart()

        def load_empty_board(self):
            self.cancel_score_chart_analysis()
            self.game_board = [[""] * 9 for _ in range(10)]
            self.game_side = "red"
            self.current_fen = board_to_fen(self.game_board, self.game_side)
            self.current_grid = [row[:] for row in self.game_board]
            self.current_move = ""
            self.selected_square = None
            self.legal_target_moves = {}
            self.move_history = []
            self.history_fens = [self.current_fen]
            self.history_moves = []
            self.history_scores = [0]
            self.history_index = 0
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            self.schedule_analysis_detail_restart()

        def display_square(self, row, col):
            if self.board_flipped:
                return 9 - row, 8 - col
            return row, col

        def logical_square(self, row, col):
            if self.board_flipped:
                return 9 - row, 8 - col
            return row, col

        def board_metrics(self, width=None, height=None):
            width = float(width if width is not None else self.board_pixmap.width())
            height = float(height if height is not None else self.board_pixmap.height())
            return {
                "origin_x": width / 18.0,
                "origin_y": height / 20.0,
                "step_x": width / 9.0,
                "step_y": height / 10.0,
                "piece": min(width / 9.0, height / 10.0),
            }

        def board_center(self, row, col, width=None, height=None):
            metrics = self.board_metrics(width, height)
            return (
                (col + 0.5) * metrics["step_x"],
                (row + 0.5) * metrics["step_y"],
            )

        def handle_board_click(self, event):
            if self.game_mode not in {"human", "ai", "trainer", "setup"} or self.game_board is None:
                return
            if self.is_computer_turn():
                self.set_status("请等电脑走完")
                return
            square = self.board_square_from_pos(event.pos().x(), event.pos().y())
            if square is None:
                return
            row, col = square
            if event.button() == Qt.RightButton and self.game_mode == "setup":
                self.show_piece_menu(row, col, event.globalPos())
                return
            if self.selected_square is None:
                self.select_game_square(row, col)
                return
            move = self.legal_target_moves.get((row, col))
            if move:
                self.play_local_move(move)
                return
            self.select_game_square(row, col)

        def show_piece_menu(self, row, col, global_pos):
            menu = QMenu(self)
            piece_groups = [
                ("红方", [("帅", "K"), ("仕", "A"), ("相", "B"), ("马", "N"), ("车", "R"), ("炮", "C"), ("兵", "P")]),
                ("黑方", [("将", "k"), ("士", "a"), ("象", "b"), ("马", "n"), ("车", "r"), ("炮", "c"), ("卒", "p")]),
            ]
            for group_name, pieces in piece_groups:
                submenu = menu.addMenu(group_name)
                for label, piece in pieces:
                    action = submenu.addAction(label)
                    action.triggered.connect(lambda _checked=False, p=piece: self.set_setup_piece(row, col, p))
            menu.addSeparator()
            clear_action = menu.addAction("清空")
            clear_action.triggered.connect(lambda _checked=False: self.set_setup_piece(row, col, ""))
            menu.exec_(global_pos)

        def set_setup_piece(self, row, col, piece):
            if self.game_board is None:
                return
            self.cancel_score_chart_analysis()
            self.game_board[row][col] = piece
            self.current_grid = [line[:] for line in self.game_board]
            self.current_fen = board_to_fen(self.game_board, self.game_side)
            self.current_move = ""
            self.selected_square = None
            self.legal_target_moves = {}
            self.history_fens = [self.current_fen]
            self.history_moves = []
            self.history_scores = [0]
            self.history_index = 0
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            text = "已清空" if not piece else f"已添加：{self.PIECE_LABELS.get(piece, piece)}"
            self.set_status(text)

        def board_square_from_pos(self, x, y):
            if not self.board_draw_rect:
                return None
            left, top, width, height = self.board_draw_rect
            if x < left or y < top or x >= left + width or y >= top + height:
                return None
            local_x = x - left
            local_y = y - top
            col = int(local_x / width * 9)
            row = int(local_y / height * 10)
            if 0 <= row < 10 and 0 <= col < 9:
                return self.logical_square(row, col)
            return None

        def select_game_square(self, row, col):
            piece = self.game_board[row][col]
            if piece_side(piece) != self.game_side:
                self.selected_square = None
                self.legal_target_moves = {}
                self.update_board_display(self.current_grid)
                return
            try:
                moves = legal_moves_for_piece(self.game_board, row, col)
            except Exception:
                moves = []
            self.selected_square = (row, col)
            self.legal_target_moves = {
                uci_to_coords(move)[2:4]: move
                for move in moves
            }
            self.update_board_display(self.current_grid)
            self.set_status(f"已选中：{self.PIECE_LABELS.get(piece, piece)}")

        def play_local_move(self, move):
            if not is_legal_move(self.game_board, self.game_side, move):
                self.set_status("这步不合法")
                return
            if self.history_fens and self.history_index < len(self.history_fens) - 1:
                self.history_fens = self.history_fens[: self.history_index + 1]
                self.history_moves = self.history_moves[: self.history_index]
                self.history_scores = self.history_scores[: self.history_index + 1]
            self.game_board = apply_uci(self.game_board, move)
            self.move_history.append(move)
            self.history_moves.append(move)
            self.game_side = opponent(self.game_side)
            self.current_fen = board_to_fen(self.game_board, self.game_side)
            self.history_fens.append(self.current_fen)
            self.history_scores.append(None)
            self.history_index = len(self.history_fens) - 1
            self.current_grid = [row[:] for row in self.game_board]
            self.current_move = move
            self.selected_square = None
            self.legal_target_moves = {}
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            self.set_status(f"{'红方' if self.game_side == 'red' else '黑方'}走")
            if self.side_panel.isVisible() and self.side_panel_title.text() == "开局库":
                self.load_opening_book_panel()
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "局势图":
                self.refresh_score_chart()
            if self.game_mode == "ai" and self.game_side != self.player_side:
                self.request_ai_move()
            elif self.is_computer_turn():
                self.request_ai_move()
            elif self.game_mode == "trainer":
                self.calculate_local_position("训练模式：继续找最佳走法")
            self.start_score_chart_analysis(auto=True)
            self.schedule_analysis_detail_restart()

        def on_computer_side_changed(self):
            self.update_engine_side_actions()
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            if self.is_computer_turn():
                self.request_ai_move()

        def is_computer_turn(self):
            if self.game_board is None:
                return False
            return (
                (self.game_side == "red" and self.red_ai_checkbox.isChecked())
                or (self.game_side == "black" and self.black_ai_checkbox.isChecked())
            )

        def selected_ai_search(self):
            if self.engine_search_mode_combo.currentText() == "随机深度":
                min_depth, max_depth = sorted((self.min_depth_spin.value(), self.max_depth_spin.value()))
                return None, random.randint(min_depth, max_depth)
            min_sec, max_sec = sorted((self.min_time_spin.value(), self.max_time_spin.value()))
            seconds = random.uniform(min_sec, max_sec)
            return int(round(seconds * 1000)), None

        def engine_search_summary(self):
            if self.engine_search_mode_combo.currentText() == "随机深度":
                min_depth, max_depth = sorted((self.min_depth_spin.value(), self.max_depth_spin.value()))
                return f"随机深度 {min_depth}-{max_depth}"
            min_sec, max_sec = sorted((self.min_time_spin.value(), self.max_time_spin.value()))
            return f"随机时间 {min_sec:.1f}-{max_sec:.1f} 秒"

        def selected_book_delay_ms(self):
            min_sec, max_sec = sorted((self.min_time_spin.value(), self.max_time_spin.value()))
            return max(1, int(round(random.uniform(min_sec, max_sec) * 1000)))

        def pick_opening_book_move(self, fen):
            if not self.use_open_book:
                return None
            for item in self.query_xqbook(fen):
                move = item.get("uci")
                if move and self.game_board is not None and is_legal_move(self.game_board, self.game_side, move):
                    return item
            return None

        def apply_book_move(self, fen, item):
            if self.pending_book_fen == fen:
                self.pending_book_fen = ""
            if self.current_fen != fen or not self.use_open_book or not self.is_computer_turn():
                return
            move = item.get("uci")
            if not move or self.game_board is None or not is_legal_move(self.game_board, self.game_side, move):
                self.request_ai_move()
                return
            try:
                move_text = move_to_chinese(self.game_board, move)
            except Exception:
                move_text = move
            score = book_number(item.get("score"))
            win_rate = book_win_rate(item) * 100
            self.update_analysis_display(
                f"命中库招：{move_text}\n"
                f"选招方式：{book_priority_label(self.book_move_priority)}\n"
                f"分数：{score}    胜率：{win_rate:.1f}%"
            )
            self.set_status(f"库招走棋：{move_text}")
            self.play_local_move(move)

        def request_ai_move(self):
            if self.ai_thread and self.ai_thread.is_alive():
                return
            book_item = self.pick_opening_book_move(self.current_fen)
            if book_item:
                if self.pending_book_fen == self.current_fen:
                    return
                delay_ms = self.selected_book_delay_ms()
                self.pending_book_fen = self.current_fen
                try:
                    move_text = move_to_chinese(self.game_board, book_item["uci"])
                except Exception:
                    move_text = book_item["uci"]
                self.set_status(f"命中库招：{move_text}，{delay_ms / 1000:.1f} 秒后走棋")
                QTimer.singleShot(delay_ms, lambda fen=self.current_fen, item=book_item: self.apply_book_move(fen, item))
                return
            self.set_status("AI 正在思考...")
            self.pending_ai_movetime, self.pending_ai_depth = self.selected_ai_search()
            self.pending_ai_fen = self.current_fen
            self.ai_thread = threading.Thread(target=self.run_ai_move, daemon=True)
            self.ai_thread.start()

        def run_ai_move(self):
            engine = None
            try:
                fen = self.current_fen
                side = self.game_side
                movetime = self.pending_ai_movetime
                depth = self.pending_ai_depth
                engine = Engine()
                engine.start_search(fen, movetime=movetime, depth=depth)
                deadline = time.time() + ((movetime / 1000.0 + 3.0) if movetime else max(8.0, depth * 1.5))
                last_text = ""
                while engine.bestmove is None and time.time() < deadline:
                    text = self.format_engine_snapshot(engine, fen)
                    if text and text != last_text:
                        last_text = text
                        self.signals.analysis.emit(text)
                    time.sleep(0.15)
                move = engine.bestmove
                if move and move != "(none)" and side == self.game_side:
                    text = self.format_engine_snapshot(engine, fen)
                    if text:
                        self.signals.analysis.emit(text + f"\n最佳：{self.format_engine_move(fen, move)}")
                    self.signals.move.emit(move, "AI")
                    try:
                        board, _ = fen_to_board(fen)
                        move_text = move_to_chinese(board, move)
                    except Exception:
                        move_text = move
                    self.signals.status.emit(f"AI 走棋：{move_text}")
                    self.signals.local_move.emit(move)
                else:
                    self.signals.status.emit("AI 没有找到走法")
            except Exception as e:
                self.signals.status.emit(f"AI 计算失败：{e}")
            finally:
                if engine is not None:
                    engine.quit()
                self.signals.ai_done.emit()

        def on_ai_done(self):
            if self.closing:
                return
            self.ai_thread = None
            finished_fen = self.pending_ai_fen
            if self.is_computer_turn():
                if finished_fen and self.current_fen == finished_fen:
                    return
                self.request_ai_move()

        def apply_ai_move(self, move):
            if self.pending_ai_fen and self.current_fen != self.pending_ai_fen:
                self.set_status("局面已变化，已忽略电脑旧走法")
                return
            if self.game_mode != "ai" and not self.is_computer_turn():
                self.set_status("电脑走棋已关闭")
                return
            if self.game_board is not None and is_legal_move(self.game_board, self.game_side, move):
                self.play_local_move(move)

        def calculate_local_position(self, status_text):
            self.show_analysis_detail_panel()
            self.analyze_current_position(status_text)

        def schedule_analysis_detail_restart(self):
            if not self.analysis_detail_enabled or self.closing:
                return
            self.analysis_detail_restart_timer.start()

        def start_analysis_detail_analysis(self):
            if not self.analysis_detail_enabled or self.closing:
                return
            if not self.current_fen:
                self.analysis_text = "还没有可分析的局面"
                self.update_analysis_display(self.analysis_text)
                return
            if (
                self.engine_analysis_thread
                and self.engine_analysis_thread.is_alive()
                and self.analysis_detail_fen == self.current_fen
            ):
                return
            self.engine_analysis_request_id += 1
            request_id = self.engine_analysis_request_id
            fen = self.current_fen
            self.analysis_detail_fen = fen
            self.update_analysis_display("")
            self.engine_analysis_thread = threading.Thread(
                target=self.run_engine_analysis,
                args=(fen, None, None, request_id, True),
                daemon=True,
            )
            self.engine_analysis_thread.start()

        def analyze_current_position(self, status_text="引擎正在分析当前局面..."):
            if not self.current_fen:
                self.set_status("还没有可分析的局面")
                return
            if self.analysis_detail_enabled:
                self.show_analysis_detail_panel()
                self.start_analysis_detail_analysis()
                self.set_status(status_text)
                return
            self.engine_analysis_request_id += 1
            request_id = self.engine_analysis_request_id
            fen = self.current_fen
            movetime, depth = self.selected_ai_search()
            self.update_analysis_display("")
            self.set_status(status_text)
            self.engine_analysis_thread = threading.Thread(
                target=self.run_engine_analysis,
                args=(fen, movetime, depth, request_id),
                daemon=True,
            )
            self.engine_analysis_thread.start()

        def run_engine_analysis(self, fen, movetime, depth, request_id, continuous=False):
            engine = None
            try:
                engine = Engine()
                engine.start_search(fen, movetime=movetime, depth=depth, infinite=continuous)
                if continuous:
                    last_text = ""
                    while (
                        request_id == self.engine_analysis_request_id
                        and self.analysis_detail_enabled
                        and self.current_fen == fen
                        and not self.closing
                    ):
                        text = self.format_engine_snapshot(engine, fen)
                        if text and text != last_text:
                            last_text = text
                            self.signals.analysis.emit(text)
                        time.sleep(0.15)
                    engine.send_cmd("stop")
                    return
                deadline = time.time() + ((movetime / 1000.0 + 2.0) if movetime else max(8.0, depth * 1.5))
                last_text = ""
                while time.time() < deadline and engine.bestmove is None:
                    if request_id != self.engine_analysis_request_id or self.closing:
                        return
                    text = self.format_engine_snapshot(engine, fen)
                    if text and text != last_text:
                        last_text = text
                        self.signals.analysis.emit(text)
                    time.sleep(0.15)
                if request_id != self.engine_analysis_request_id or self.closing:
                    return
                text = self.format_engine_snapshot(engine, fen)
                if engine.bestmove and engine.bestmove != "(none)":
                    best_text = self.format_engine_move(fen, engine.bestmove)
                    text = (text + "\n" if text else "") + f"最佳：{best_text}"
                    self.signals.move.emit(engine.bestmove, "最佳")
                self.signals.analysis.emit(text or "")
                self.signals.status.emit("引擎分析完成")
            except Exception as e:
                if request_id == self.engine_analysis_request_id:
                    self.signals.analysis.emit(f"引擎分析失败：{e}")
                    self.signals.status.emit(f"引擎分析失败：{e}")
            finally:
                if engine is not None:
                    engine.quit()

        def format_engine_snapshot(self, engine, fen):
            lines = []
            pv_lines = getattr(engine, "pv_lines", {}) or {}
            pv_line = getattr(engine, "current_pv_line", []) or []
            if not pv_lines and pv_line:
                pv_lines = {1: pv_line}
            scores = getattr(engine, "pv_scores", {}) or {}
            depths = getattr(engine, "pv_depths", {}) or {}
            times = getattr(engine, "pv_times", {}) or {}
            nps_values = getattr(engine, "pv_nps", {}) or {}
            for rank in sorted(pv_lines)[:5]:
                moves = pv_lines.get(rank) or []
                if not moves:
                    continue
                score = scores.get(rank, getattr(engine, "eval_score", "") if rank == 1 else "")
                score_info = java_oriented_engine_score(score, fen, self.board_flipped)
                if rank == 1 and score_info is not None:
                    self.signals.engine_score.emit(fen or "", int(score_info["chart_score"]))
                depth = depths.get(rank, getattr(engine, "current_depth", 0) if rank == 1 else 0)
                time_ms = times.get(rank, getattr(engine, "current_time_ms", 0) if rank == 1 else 0)
                nps = nps_values.get(rank, getattr(engine, "current_nps", 0) if rank == 1 else 0)
                display = self.format_engine_line(fen, moves[:8])
                body = "  ".join(display)
                if "null" in body:
                    continue
                lines.append(java_think_title(depth, rank, score_info, nps, time_ms))
                if body:
                    lines.append(body)
            if not lines:
                depth = getattr(engine, "current_depth", 0)
                score_info = java_oriented_engine_score(getattr(engine, "eval_score", ""), fen, self.board_flipped)
                if depth or score_info is not None:
                    lines.append(java_think_title(depth, 1, score_info, getattr(engine, "current_nps", 0), getattr(engine, "current_time_ms", 0)))
            return "\n".join(lines)

        def format_engine_score(self, score):
            score_info = java_oriented_engine_score(score, self.current_fen, self.board_flipped)
            if score_info is None:
                return "-"
            if score_info["kind"] == "mate":
                return f"{score_info['score']}步"
            return str(score_info["score"])

        def format_engine_move(self, fen, move):
            try:
                board, _side = fen_to_board(fen)
                return move_to_chinese(board, move)
            except Exception:
                return move

        def format_engine_line(self, fen, moves):
            try:
                board, _side = fen_to_board(fen)
            except Exception:
                return list(moves)
            current_board = [row[:] for row in board]
            display = []
            for move in moves:
                try:
                    display.append(move_to_chinese(current_board, move))
                    current_board = apply_uci(current_board, move)
                except Exception:
                    display.append(move)
            return display

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.update_board_display(self.current_grid)

        def update_fen(self, fen):
            self.current_fen = fen or ""
            if self.side_panel.isVisible() and self.side_panel_title.text() == "开局库":
                self.load_opening_book_panel()
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "局势图":
                self.refresh_score_chart()
            self.refresh_right_workspace()
            self.schedule_analysis_detail_restart()

        def update_analysis_display(self, text):
            self.analysis_text = text or ""
            if self.analysis_dialog.isVisible():
                self.analysis_box.setPlainText(self.analysis_text)
            self.refresh_right_workspace()

        def update_history_score(self, request_id, index, score):
            if request_id != self.score_analysis_request_id:
                return
            self.ensure_history_scores()
            if index < 0 or index >= len(self.history_scores):
                return
            try:
                score = int(score)
            except (TypeError, ValueError):
                return
            if self.history_scores[index] == score:
                return
            self.history_scores[index] = score
            self.refresh_right_workspace()

        def update_engine_score(self, fen, score):
            if fen and self.current_fen and fen.strip() != self.current_fen.strip():
                return
            self.ensure_history_scores()
            if self.history_index < 0 or self.history_index >= len(self.history_scores):
                return
            try:
                score = int(score)
            except (TypeError, ValueError):
                return
            if self.history_scores[self.history_index] == score:
                return
            self.history_scores[self.history_index] = score
            self.refresh_right_workspace()

        def on_score_analysis_done(self, request_id, completed, total):
            if request_id != self.score_analysis_request_id:
                return
            self.score_analysis_thread = None
            if hasattr(self, "manual_score_button"):
                self.manual_score_button.setEnabled(True)
            if self.closing:
                return
            if completed >= total:
                self.set_status(f"棋谱局势图分析完成：{completed}/{total}")
            elif completed:
                self.set_status(f"棋谱局势图分析已停止：{completed}/{total}")

        def update_record_display(self, text):
            self.online_record_text = text or ""
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            self.refresh_right_workspace()

        def copy_fen(self):
            if not self.current_fen:
                self.set_status("还没有可复制的 FEN，请先计算或开始识别。")
                return
            QApplication.clipboard().setText(self.current_fen)
            self.set_status("已复制当前 FEN")

        def paste_fen_from_clipboard(self):
            fen = QApplication.clipboard().text().strip()
            if not fen:
                self.import_fen_position()
                return
            try:
                validate_fen(fen)
                self.load_game_from_fen(fen)
                self.game_mode = "setup"
                self.show_navigation_panel()
                self.set_status("已粘贴局面 FEN")
            except Exception as e:
                self.set_status(f"粘贴 FEN 失败：{e}")

        def copy_board_image(self):
            pixmap = self.board_label.pixmap()
            if pixmap is None or pixmap.isNull():
                self.set_status("还没有可复制的局面图片")
                return
            QApplication.clipboard().setPixmap(pixmap)
            self.set_status("已复制局面图片")

        def paste_board_image(self):
            image = QApplication.clipboard().image()
            if image.isNull():
                self.set_status("剪贴板里没有局面图片")
                return
            self.show_text_dialog("粘贴局面图片", "已读取剪贴板图片。当前项目的图片识别仍使用校准后的模板识别，请优先用“图片导入局面”选择文件。")

        def export_board_image(self):
            pixmap = self.board_label.pixmap()
            if pixmap is None or pixmap.isNull():
                self.set_status("还没有可导出的局面图片")
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "导出局面图片",
                "xiangqi_position.png",
                "PNG 图片 (*.png);;所有文件 (*.*)",
            )
            if not path:
                return
            if not path.lower().endswith(".png"):
                path += ".png"
            if pixmap.save(path, "PNG"):
                self.set_status(f"局面图片已导出：{path}")
            else:
                self.set_status("局面图片导出失败")

        def import_position_image(self):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "图片导入局面",
                "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)",
            )
            if not path:
                return
            try:
                image = cv2.imread(path)
                if image is None:
                    raise ValueError("图片无法读取")
                vision = Vision()
                board_img = vision.detect_board(image)
                grid = vision.get_board_state(board_img, sticky=False)
                side_token = "w" if self.game_side == "red" else "b"
                fen = vision.to_fen(grid, side_token)
                validate_fen(fen)
                self.load_game_from_fen(fen)
                self.game_mode = "setup"
                self.show_navigation_panel()
                self.set_status("已从图片导入局面")
            except Exception as e:
                self.set_status(f"图片导入失败：{e}")

        def show_current_fen(self):
            if not self.current_fen:
                self.set_status("还没有 FEN，请先计算或开始识别。")
                return
            self.show_text_dialog("当前 FEN", self.current_fen)

        def copy_analysis_text(self):
            QApplication.clipboard().setText(self.analysis_text)
            self.set_status("已复制分析详情")

        def clear_move_marker(self):
            self.current_move = ""
            self.current_move_label = ""
            self.update_board_display(self.current_grid)
            self.set_status("已清除棋盘标记")

        def show_usage_tips(self):
            self.show_text_dialog(
                "连线说明",
                "1. 第一次使用先打开 连线 -> 设置 -> 校准棋盘。\n"
                "2. 想自动走棋点工具栏“自动走棋”，只看分析点“观战”。\n"
                "3. 计算时会显示实时分数和候选走法。\n"
                "4. 两个开关互斥，关闭当前开关会停止连线。",
            )

        def show_text_dialog(self, title, text):
            dialog = QDialog(self)
            dialog.setWindowTitle(title)
            dialog.setModal(False)
            layout = QVBoxLayout()
            box = QPlainTextEdit()
            box.setReadOnly(True)
            box.setPlainText(text)
            layout.addWidget(box)

            buttons = QHBoxLayout()
            copy_button = QPushButton("复制")
            close_button = QPushButton("关闭")
            copy_button.clicked.connect(lambda: QApplication.clipboard().setText(box.toPlainText()))
            copy_button.clicked.connect(lambda: self.set_status("已复制"))
            close_button.clicked.connect(dialog.close)
            buttons.addWidget(copy_button)
            buttons.addWidget(close_button)
            layout.addLayout(buttons)

            dialog.setLayout(layout)
            dialog.resize(460, 220)
            self.text_dialogs.append(dialog)
            dialog.destroyed.connect(lambda: self.text_dialogs.remove(dialog) if dialog in self.text_dialogs else None)
            dialog.show()

        def update_move_display(self, move, label):
            self.current_move = move or ""
            self.current_move_label = label or ""
            self.update_board_display(self.current_grid)

        def update_board_display(self, grid):
            if self.board_pixmap.isNull():
                return
            if grid is not None:
                self.current_grid = grid

            board = QPixmap(self.board_pixmap)
            draw_grid = self.current_grid
            if draw_grid:
                painter = QPainter(board)
                painter.setRenderHint(QPainter.Antialiasing, True)
                step_x = board.width() / 9.0
                step_y = board.height() / 10.0
                radius = int(min(step_x, step_y) * 0.36)
                font = QFont("Microsoft YaHei", max(14, int(radius * 0.95)))
                font.setBold(True)
                painter.setFont(font)

                if self.show_board_numbers:
                    number_font = QFont("Microsoft YaHei", max(10, int(radius * 0.36)))
                    painter.setFont(number_font)
                    painter.setPen(QPen(QColor(30, 30, 30, 190), 1))
                    black_numbers = ["１", "２", "３", "４", "５", "６", "７", "８", "９"]
                    red_numbers = ["九", "八", "七", "六", "五", "四", "三", "二", "一"]
                    top_y = max(2, int(step_y * 0.12))
                    bottom_y = board.height() - max(2, int(step_y * 0.30))
                    for index in range(9):
                        top_col = 8 - index if self.board_flipped else index
                        bottom_col = index if self.board_flipped else 8 - index
                        top_x = int(round((top_col + 0.5) * step_x - radius * 0.25))
                        bottom_x = int(round((bottom_col + 0.5) * step_x - radius * 0.25))
                        painter.drawText(top_x, top_y, black_numbers[index])
                        painter.drawText(bottom_x, bottom_y, red_numbers[index])
                    painter.setFont(font)

                for row_index, row in enumerate(draw_grid):
                    for col_index, piece in enumerate(row):
                        if not piece:
                            continue
                        label = self.PIECE_LABELS.get(piece, piece)
                        draw_row, draw_col = self.display_square(row_index, col_index)
                        cx = int(round((draw_col + 0.5) * step_x))
                        cy = int(round((draw_row + 0.5) * step_y))
                        color = QColor(190, 25, 25) if piece.isupper() else QColor(20, 20, 20)

                        painter.setPen(QPen(color, 3))
                        painter.setBrush(QBrush(QColor(250, 236, 204, 235)))
                        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
                        painter.setPen(QPen(color, 2))
                        painter.drawText(
                            cx - radius,
                            cy - radius,
                            radius * 2,
                            radius * 2,
                            Qt.AlignCenter,
                            label,
                        )

                if self.selected_square:
                    sel_row, sel_col = self.selected_square
                    draw_sel_row, draw_sel_col = self.display_square(sel_row, sel_col)
                    sel_x = int(round((draw_sel_col + 0.5) * step_x))
                    sel_y = int(round((draw_sel_row + 0.5) * step_y))
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(240, 160, 20), max(4, radius // 4)))
                    painter.drawEllipse(sel_x - radius - 4, sel_y - radius - 4, radius * 2 + 8, radius * 2 + 8)
                    painter.setBrush(QBrush(QColor(60, 150, 90, 175)))
                    painter.setPen(Qt.NoPen)
                    target_radius = max(6, radius // 3)
                    for target_row, target_col in self.legal_target_moves:
                        draw_target_row, draw_target_col = self.display_square(target_row, target_col)
                        target_x = int(round((draw_target_col + 0.5) * step_x))
                        target_y = int(round((draw_target_row + 0.5) * step_y))
                        painter.drawEllipse(
                            target_x - target_radius,
                            target_y - target_radius,
                            target_radius * 2,
                            target_radius * 2,
                        )

                if self.current_move and self.show_step_tip:
                    try:
                        src_row, src_col, dst_row, dst_col = uci_to_coords(self.current_move)
                    except ValueError:
                        src_row = src_col = dst_row = dst_col = None
                    if src_row is not None:
                        draw_src_row, draw_src_col = self.display_square(src_row, src_col)
                        draw_dst_row, draw_dst_col = self.display_square(dst_row, dst_col)
                        src_x = int(round((draw_src_col + 0.5) * step_x))
                        src_y = int(round((draw_src_row + 0.5) * step_y))
                        dst_x = int(round((draw_dst_col + 0.5) * step_x))
                        dst_y = int(round((draw_dst_row + 0.5) * step_y))
                        dx = dst_x - src_x
                        dy = dst_y - src_y
                        distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
                        ux = dx / distance
                        uy = dy / distance
                        edge_gap = min(radius, max(0, int(distance / 2) - 1))
                        start_x = int(round(src_x + ux * edge_gap))
                        start_y = int(round(src_y + uy * edge_gap))
                        end_x = int(round(dst_x - ux * edge_gap))
                        end_y = int(round(dst_y - uy * edge_gap))

                        mark = QColor(220, 30, 30)
                        line_width = max(5, radius // 4)
                        painter.setPen(QPen(mark, line_width))
                        painter.drawLine(start_x, start_y, end_x, end_y)

                        arrow_len = max(16, int(radius * 0.9))
                        arrow_w = max(9, int(radius * 0.55))
                        base_x = end_x - ux * arrow_len
                        base_y = end_y - uy * arrow_len
                        left_x = int(round(base_x - uy * arrow_w))
                        left_y = int(round(base_y + ux * arrow_w))
                        right_x = int(round(base_x + uy * arrow_w))
                        right_y = int(round(base_y - ux * arrow_w))
                        painter.drawLine(end_x, end_y, left_x, left_y)
                        painter.drawLine(end_x, end_y, right_x, right_y)

                painter.end()

            target_size = self.board_label.size()
            target_width = max(target_size.width(), self.board_label.minimumWidth())
            target_height = max(target_size.height(), self.board_label.minimumHeight())
            scaled = board.scaled(
                target_width,
                target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            left = (target_width - scaled.width()) // 2
            top = (target_height - scaled.height()) // 2
            self.board_draw_rect = (left, top, scaled.width(), scaled.height())
            self.board_label.setPixmap(scaled)

        def make_seconds_spin(self, minimum, maximum):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" 秒")
            return spin

        def config_float(self, config, key, default):
            try:
                return float(config.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def config_bool(self, config, key, default=False):
            value = config.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        def config_int(self, config, key, default=0):
            try:
                return int(float(config.get(key, default)))
            except (TypeError, ValueError):
                return int(default)

        def load_settings(self):
            config = load_config()
            self.board_style = "classic"
            self.board_pixmap = self.load_board_pixmap()
            self.show_board_numbers = self.config_bool(config, "show_board_numbers", False)
            self.show_step_tip = self.config_bool(config, "show_step_tip", True)
            self.step_sound_enabled = self.config_bool(config, "step_sound", False)
            self.use_open_book = self.config_bool(config, "use_open_book", True)
            self.book_move_priority = normalize_book_priority(
                config.get("book_move_priority", BOOK_PRIORITY_SCORE)
            )
            self.opening_book_path = find_opening_book_path(config)
            if hasattr(self, "show_number_action"):
                self.show_number_action.setChecked(self.show_board_numbers)
            if hasattr(self, "step_tip_action"):
                self.step_tip_action.setChecked(self.show_step_tip)
            if hasattr(self, "step_sound_action"):
                self.step_sound_action.setChecked(self.step_sound_enabled)
            self.update_book_switch_actions()
            if hasattr(self, "book_priority_combo"):
                self.book_priority_combo.blockSignals(True)
                self.book_priority_combo.setCurrentText(book_priority_label(self.book_move_priority))
                self.book_priority_combo.blockSignals(False)
            self.set_spin_value_silent(self.min_time_spin,
                self.config_float(config, "auto_move_search_time_min", 2.0)
            )
            self.set_spin_value_silent(self.max_time_spin,
                self.config_float(config, "auto_move_search_time_max", 5.0)
            )
            self.set_spin_value_silent(self.min_depth_spin,
                self.config_int(config, "engine_depth_min", 8)
            )
            self.set_spin_value_silent(self.max_depth_spin,
                self.config_int(config, "engine_depth_max", 12)
            )
            search_mode = config.get("engine_search_mode", "time")
            was_blocked = self.engine_search_mode_combo.blockSignals(True)
            self.engine_search_mode_combo.setCurrentText(
                "随机深度" if search_mode == "depth" else "随机时间"
            )
            self.engine_search_mode_combo.blockSignals(was_blocked)
            self.set_auto_move_enabled(
                self.config_bool(config, "auto_move", True),
                save=False,
            )
            threads = max(1, min(128, self.config_int(config, "engine_threads", 1)))
            self.set_spin_value_silent(self.engine_threads_spin, threads)
            if hasattr(self, "thread_combo"):
                self.sync_combo_text(self.thread_combo, str(threads))
            hash_mb = max(1, min(4096, self.config_int(config, "engine_hash", 128)))
            self.set_spin_value_silent(self.engine_hash_spin, hash_mb)
            if hasattr(self, "hash_combo"):
                self.sync_combo_text(self.hash_combo, str(hash_mb))
            self.auto_next_checkbox.setChecked(
                self.config_bool(config, "auto_start_next_game", False)
            )

        def save_settings(self, silent=False):
            if getattr(self, "_saving_settings", False):
                return
            self._saving_settings = True
            try:
                min_sec, max_sec = sorted((
                    self.min_time_spin.value(),
                    self.max_time_spin.value(),
                ))
                min_depth, max_depth = sorted((
                    self.min_depth_spin.value(),
                    self.max_depth_spin.value(),
                ))
                threads = max(1, min(128, self.engine_threads_spin.value()))
                hash_mb = max(1, min(4096, self.engine_hash_spin.value()))
                self.set_spin_value_silent(self.min_time_spin, min_sec)
                self.set_spin_value_silent(self.max_time_spin, max_sec)
                self.set_spin_value_silent(self.min_depth_spin, min_depth)
                self.set_spin_value_silent(self.max_depth_spin, max_depth)
                self.set_spin_value_silent(self.engine_threads_spin, threads)
                self.set_spin_value_silent(self.engine_hash_spin, hash_mb)
                self.sync_combo_text(self.thread_combo, str(threads))
                self.sync_combo_text(self.hash_combo, str(hash_mb))
                try:
                    update_config({
                        "auto_move_search_time_min": round(min_sec, 2),
                        "auto_move_search_time_max": round(max_sec, 2),
                        "engine_search_mode": "depth" if self.engine_search_mode_combo.currentText() == "随机深度" else "time",
                        "engine_depth_min": min_depth,
                        "engine_depth_max": max_depth,
                        "engine_threads": threads,
                        "engine_hash": hash_mb,
                        "auto_move": self.auto_move_enabled,
                        "auto_start_next_game": self.auto_next_checkbox.isChecked(),
                        "next_game_fullscreen_scan": True,
                        "show_board_numbers": self.show_board_numbers,
                        "show_step_tip": self.show_step_tip,
                        "step_sound": self.step_sound_enabled,
                        "use_open_book": self.use_open_book,
                        "book_move_priority": self.book_move_priority,
                        "opening_book_path": self.opening_book_path or "",
                    })
                except Exception as e:
                    self.set_status(f"设置保存失败：{e}")
                    return
                if self.app_worker is not None:
                    self.app_worker.request_reload()
                    if not silent:
                        self.set_status("设置已保存，正在重新读取...")
                elif not silent:
                    self.set_status("设置已保存")
            finally:
                self._saving_settings = False

        def set_status(self, text):
            self.status.setText(text)
            if hasattr(self, "timeShowLabel"):
                self.timeShowLabel.setText(time.strftime("%H:%M:%S"))
            self.refresh_right_workspace()

        def helper_running(self):
            return self.worker_thread is not None and self.worker_thread.is_alive()

        def update_link_mode_switches(self, active=None, enabled=True):
            if active is not None:
                self.link_mode_active = bool(active)
            if not hasattr(self, "auto_move_button") or not hasattr(self, "watch_button"):
                return
            pairs = (
                (self.auto_move_button, self.link_mode_active and self.auto_move_enabled),
                (self.watch_button, self.link_mode_active and not self.auto_move_enabled),
            )
            for button, checked in pairs:
                was_blocked = button.blockSignals(True)
                button.setChecked(bool(checked))
                button.setEnabled(bool(enabled))
                button.blockSignals(was_blocked)

        def start_helper(self):
            if self.worker_thread and self.worker_thread.is_alive():
                self.set_status("已经在运行")
                return
            if not calibration_exists():
                self.set_status("缺少 config.json，请先点击“校准棋盘”。")
                self.update_link_mode_switches(False, True)
                return

            self.link_mode_active = True
            self.save_settings(silent=True)
            self.update_link_mode_switches(True, True)
            self.set_status("正在启动...")
            self.update_record_display("")
            self.worker_thread = threading.Thread(target=self.run_worker, daemon=True)
            self.worker_thread.start()

        def stop_helper(self):
            if self.app_worker is None:
                self.set_status("还在启动中...")
                self.update_link_mode_switches(True, True)
                return
            self.update_link_mode_switches(False, False)
            self.set_status("正在停止...")
            self.app_worker.stop()

        def run_worker(self):
            try:
                self.app_worker = App(
                    exit_on_error=False,
                    hide_dock_icon=False,
                    board_callback=self.signals.board.emit,
                    move_callback=self.signals.move.emit,
                    fen_callback=self.signals.fen.emit,
                    analysis_callback=self.signals.analysis.emit,
                    record_callback=self.signals.record.emit,
                )
                if self.closing:
                    self.app_worker.stop()
                    self.app_worker.eng.quit()
                    return
                self.signals.status.emit("正在运行")
                self.app_worker.run(use_keyboard=False)
            except Exception as e:
                self.signals.status.emit(f"错误：{e}")
            finally:
                self.app_worker = None
                self.signals.stopped.emit()

        def reload_helper(self):
            if self.app_worker is None:
                self.set_status("请先开启“自动走棋”或“观战”")
                return
            self.app_worker.request_reload()
            self.set_status("正在重新读取...")

        def calculate_now(self):
            if self.calculate_thread and self.calculate_thread.is_alive():
                self.set_status("已经在计算")
                return
            if not calibration_exists():
                self.set_status("缺少 config.json，请先点击“校准棋盘”。")
                return
            if self.calculate_button is not None:
                self.calculate_button.setEnabled(False)
            self.current_move = ""
            self.current_move_label = ""
            self.update_board_display(self.current_grid)
            self.set_status("正在计算当前棋盘...")
            self.calculate_thread = threading.Thread(
                target=self.run_single_calculation,
                daemon=True,
            )
            self.calculate_thread.start()

        def run_single_calculation(self):
            app = None
            try:
                app = App(
                    exit_on_error=False,
                    hide_dock_icon=False,
                    board_callback=self.signals.board.emit,
                    move_callback=self.signals.move.emit,
                    fen_callback=self.signals.fen.emit,
                    analysis_callback=self.signals.analysis.emit,
                    record_callback=self.signals.record.emit,
                    enable_mouse=False,
                )
                if self.closing:
                    return
                ok = app.calculate_current_board()
                if not ok:
                    self.signals.status.emit("没有识别到可计算的棋盘")
            except Exception as e:
                self.signals.status.emit(f"计算失败：{e}")
            finally:
                if app is not None:
                    try:
                        app.eng.quit()
                    except Exception:
                        pass
                self.signals.calculation_done.emit()

        def calibrate_helper(self):
            if self.calibrate_thread and self.calibrate_thread.is_alive():
                self.set_status("校准窗口已经打开")
                return
            self.calibrate_button.setEnabled(False)
            self.set_status("正在打开校准窗口...")
            self.calibrate_thread = threading.Thread(target=self.run_calibration, daemon=True)
            self.calibrate_thread.start()

        def recognize_next_game(self):
            if self.recognize_thread and self.recognize_thread.is_alive():
                self.set_status("识别窗口已经打开")
                return
            self.recognize_next_button.setEnabled(False)
            self.set_status("正在打开下一局按钮识别窗口...")
            self.recognize_thread = threading.Thread(
                target=self.run_next_game_recognition,
                daemon=True,
            )
            self.recognize_thread.start()

        def run_calibration(self):
            try:
                if self.app_worker is not None:
                    self.app_worker.stop()
                if self.worker_thread is not None and self.worker_thread.is_alive():
                    self.worker_thread.join(timeout=3.0)

                result = subprocess.run(_calibration_command(), cwd=_app_dir(), check=False)
                if result.returncode == 0:
                    self.signals.status.emit("校准已保存")
                else:
                    self.signals.status.emit(f"校准窗口已关闭：{result.returncode}")
            except Exception as e:
                self.signals.status.emit(f"校准失败：{e}")
            finally:
                self.signals.calibration_done.emit()

        def run_next_game_recognition(self):
            try:
                if self.app_worker is not None:
                    self.app_worker.stop()
                if self.worker_thread is not None and self.worker_thread.is_alive():
                    self.worker_thread.join(timeout=3.0)

                result = subprocess.run(
                    _next_game_recognition_command(),
                    cwd=_app_dir(),
                    check=False,
                )
                if result.returncode == 0:
                    self.signals.status.emit("下一局按钮已保存")
                else:
                    self.signals.status.emit(f"识别窗口已关闭：{result.returncode}")
            except Exception as e:
                self.signals.status.emit(f"识别失败：{e}")
            finally:
                self.signals.recognition_done.emit()

        def on_calibration_done(self):
            self.calibrate_button.setEnabled(True)
            self.load_settings()

        def on_recognition_done(self):
            self.recognize_next_button.setEnabled(True)
            self.load_settings()

        def on_calculation_done(self):
            if self.calculate_button is not None:
                self.calculate_button.setEnabled(True)

        def on_stopped(self):
            self.update_link_mode_switches(False, True)
            if not self.status.text().startswith("错误："):
                self.set_status("已停止")

        def closeEvent(self, event):
            self.closing = True
            self.analysis_detail_restart_timer.stop()
            self.engine_analysis_request_id += 1
            self.score_analysis_request_id += 1
            if self.app_worker is not None:
                self.app_worker.stop()
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=2.0)
            if self.calibrate_thread is not None:
                self.calibrate_thread.join(timeout=1.0)
            if self.calculate_thread is not None:
                self.calculate_thread.join(timeout=1.0)
            if self.recognize_thread is not None:
                self.recognize_thread.join(timeout=1.0)
            if self.engine_analysis_thread is not None:
                self.engine_analysis_thread.join(timeout=1.0)
            if self.score_analysis_thread is not None:
                self.score_analysis_thread.join(timeout=1.0)
            event.accept()

    qt_app = QApplication(sys.argv)
    window = HelperWindow()
    window.show()
    return qt_app.exec_()


def run_cli():
    _ensure_cooked_terminal()
    fd = sys.stdin.fileno() if termios is not None and sys.stdin.isatty() else None
    saved = termios.tcgetattr(fd) if fd is not None else None
    try:
        App().run()
    finally:
        if saved is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except Exception:
                pass


def run_calibrate_mode():
    run_calibration_flow(prompt="--no-prompt" not in sys.argv)


def run_recognize_next_mode():
    run_next_game_recognition(prompt="--no-prompt" not in sys.argv)


def main():
    if "--calibrate" in sys.argv:
        run_calibrate_mode()
    elif "--recognize-next" in sys.argv:
        run_recognize_next_mode()
    elif "--cli" in sys.argv:
        run_cli()
    else:
        sys.exit(launch_gui())


if __name__ == '__main__':
    main()
