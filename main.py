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
        depth = getattr(self.eng, "current_depth", 0)
        score = getattr(self.eng, "eval_score", "")
        if depth or score:
            lines.append(f"深度：{depth or '-'}    分数：{self._format_eval_score(score)}")

        pv_line = getattr(self.eng, "current_pv_line", []) or []
        if pv_line and self._is_engine_move_legal_for_fen(fen, pv_line[0]):
            lines.append("主线：" + " ".join(self._format_move_line(pv_line, fen, flipped)))

        pv_lines = getattr(self.eng, "pv_lines", {}) or {}
        if pv_lines:
            lines.append("")
            lines.append("候选走法：")
            for rank in sorted(pv_lines)[:5]:
                moves = pv_lines.get(rank) or []
                if not moves:
                    continue
                if not self._is_engine_move_legal_for_fen(fen, moves[0]):
                    continue
                score_text = self._format_eval_score((getattr(self.eng, "pv_scores", {}) or {}).get(rank, ""))
                gui_moves = self._format_move_line(moves[:5], fen, flipped)
                lines.append(f"{rank}. {gui_moves[0]}    {score_text}    {' '.join(gui_moves)}")

        return "\n".join(lines) if lines else "正在等待引擎分析..."

    def _is_engine_move_legal_for_fen(self, fen, move):
        if not fen or not move:
            return False
        try:
            board, side = fen_to_board(fen)
            return is_legal_move(board, side, move)
        except Exception:
            return False

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
                    if self._screen_player_side(flipped) == "red":
                        if self.stable_count < min_stable:
                            self._print_status_once(
                                f"识别到我方红方，等待棋盘稳定 {self.stable_count}/{min_stable}..."
                            )
                            return
                        self._recalc.clear()
                        moved_or_reported = self._calculate_and_print(fen, flipped)
                        return
                    if (
                        not self._is_initial_board_key(current_key)
                        and self._screen_player_side(flipped) == "black"
                    ):
                        if self.stable_count < min_stable:
                            self._print_status_once(
                                f"识别到当前局面，等待棋盘稳定 {self.stable_count}/{min_stable}..."
                            )
                            return
                        transition = self._find_legal_transition(
                            START_FEN,
                            current_key,
                            sides=("red",),
                        )
                        if transition is not None:
                            self._record_transition(transition, "opponent")
                            self._recalc.clear()
                            self._print_status_once("已记录红方第一步，开始我方走棋...")
                            moved_or_reported = self._calculate_and_print(fen, flipped)
                            return
                        self._recalc.clear()
                        self._print_status_once("从当前局面开始，计算我方走法...")
                        moved_or_reported = self._calculate_and_print(fen, flipped)
                        return
                    self._set_board_change_baseline(fen)
                    if self._is_initial_board_key(current_key):
                        self._print_status_once("对方红方先走，等待红方走棋...")
                    else:
                        self._print_status_once("等待对方走棋...")
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
                moved_or_reported = self._calculate_and_print(fen, flipped)
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
            dprint("Calculating...", end='\r', flush=True)
            movetime, search_seconds = self._search_movetime()
            dprint(f"\n[Calculate] fen={fen}")
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
    from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMenu,
        QMenuBar,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    class GuiSignals(QObject):
        status = pyqtSignal(str)
        board = pyqtSignal(object)
        move = pyqtSignal(str, str)
        fen = pyqtSignal(str)
        analysis = pyqtSignal(str)
        record = pyqtSignal(str)
        local_move = pyqtSignal(str)
        ai_done = pyqtSignal()
        stopped = pyqtSignal()
        calculation_done = pyqtSignal()
        calibration_done = pyqtSignal()
        recognition_done = pyqtSignal()

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
            self.signals.record.connect(self.update_record_display)
            self.signals.local_move.connect(self.apply_ai_move)
            self.signals.ai_done.connect(self.on_ai_done)
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
            self.analysis_text = "还没有分析结果。"
            self.online_record_text = ""
            self.settings_dialog = None
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
            self.history_index = 0
            self.ai_thread = None
            self.pending_ai_movetime = 1000
            self.pending_ai_depth = None
            self.pending_ai_fen = ""
            self.board_draw_rect = None
            self.opening_book_path = self.find_opening_book_path()

            self.board_label = QLabel()
            self.board_label.setAlignment(Qt.AlignCenter)
            self.board_label.setMinimumSize(430, 480)
            self.board_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.board_label.setToolTip("读取屏幕后，这里会同步显示识别到的棋盘。")
            self.board_label.mousePressEvent = self.handle_board_click
            self.board_pixmap = QPixmap(resolve_resource_path("assets/boardchess.jpg"))
            if self.board_pixmap.isNull():
                self.board_label.setText("棋盘图片未找到")
            else:
                self.update_board_display(None)

            self.status = QLabel("已停止")
            self.status.setAlignment(Qt.AlignCenter)

            self.min_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.max_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.engine_threads_spin = QSpinBox()
            self.engine_threads_spin.setRange(1, 128)
            self.engine_threads_spin.setValue(1)
            self.auto_move_checkbox = QCheckBox("自动走棋")
            self.calibrate_button = QPushButton("校准棋盘")
            self.auto_next_checkbox = QCheckBox("自动开始下一局")
            self.recognize_next_button = QPushButton("识别下一局按钮")
            self.min_time_spin.setToolTip("引擎最少思考几秒。")
            self.max_time_spin.setToolTip("引擎最多思考几秒。")
            self.engine_threads_spin.setToolTip("Pikafish 使用多少个 CPU 核心，默认 1。")
            self.auto_move_checkbox.setToolTip("勾选后程序会自动点击推荐走法；不勾选时只计算走法，不点鼠标。")
            self.calibrate_button.setToolTip("重新校准棋盘位置。")
            self.auto_next_checkbox.setToolTip("一局结束后，如果识别到下一局按钮，就自动点击进入下一局。")
            self.recognize_next_button.setToolTip("在结算界面点击下一局按钮中心，让程序记住按钮样子。")

            self.calibrate_button.clicked.connect(self.calibrate_helper)
            self.recognize_next_button.clicked.connect(self.recognize_next_game)

            move_time_row = QHBoxLayout()
            move_time_row.addWidget(self.min_time_spin)
            move_time_row.addWidget(QLabel("到"))
            move_time_row.addWidget(self.max_time_spin)

            settings = QGroupBox("常用设置")
            settings_form = QFormLayout()
            settings_form.addRow("思考时间", move_time_row)
            settings_form.addRow("引擎核心数量", self.engine_threads_spin)
            settings_form.addRow(self.auto_move_checkbox)
            settings_form.addRow("棋盘位置", self.calibrate_button)
            settings_form.addRow(self.auto_next_checkbox)
            settings_form.addRow("下一局识别", self.recognize_next_button)
            settings.setLayout(settings_form)

            close_settings_button = QPushButton("保存并关闭")
            close_settings_button.clicked.connect(self.save_settings_and_close)
            settings_dialog_layout = QVBoxLayout()
            settings_dialog_layout.addWidget(settings)
            settings_dialog_layout.addWidget(close_settings_button)
            self.settings_dialog = QDialog(self)
            self.settings_dialog.setWindowTitle("连线设置")
            self.settings_dialog.setModal(False)
            self.settings_dialog.setLayout(settings_dialog_layout)

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

            self.side_panel_title = QLabel("导航")
            self.side_panel_title.setStyleSheet("font-weight: 600;")
            self.side_panel_text = QPlainTextEdit()
            self.side_panel_text.setReadOnly(True)
            self.side_panel_text.setMinimumWidth(260)
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
            for button in (
                self.nav_first_button,
                self.nav_prev_button,
                self.nav_next_button,
                self.nav_last_button,
            ):
                nav_buttons.addWidget(button)
            self.nav_buttons_widget.setLayout(nav_buttons)
            self.nav_buttons_widget.hide()
            self.nav_controls_widget = QWidget()
            nav_controls = QVBoxLayout()
            nav_controls.setContentsMargins(0, 0, 0, 0)
            self.red_ai_checkbox = QCheckBox("红方电脑走棋")
            self.black_ai_checkbox = QCheckBox("黑方电脑走棋")
            self.red_ai_checkbox.setToolTip("勾选后，轮到红方时电脑自动走。")
            self.black_ai_checkbox.setToolTip("勾选后，轮到黑方时电脑自动走。")
            self.red_ai_checkbox.stateChanged.connect(lambda _state: self.on_computer_side_changed())
            self.black_ai_checkbox.stateChanged.connect(lambda _state: self.on_computer_side_changed())
            nav_controls.addWidget(self.red_ai_checkbox)
            nav_controls.addWidget(self.black_ai_checkbox)

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
            side_panel_layout = QVBoxLayout()
            side_panel_layout.addWidget(self.side_panel_title)
            side_panel_layout.addWidget(self.nav_buttons_widget)
            side_panel_layout.addWidget(self.nav_controls_widget)
            side_panel_layout.addWidget(self.side_panel_text, 1)
            self.side_panel = QWidget()
            self.side_panel.setFixedWidth(270)
            self.side_panel.setLayout(side_panel_layout)
            self.side_panel.hide()

            content = QHBoxLayout()
            content.addLayout(board_content, 1)
            content.addWidget(self.side_panel)

            central = QWidget()
            central.setLayout(content)
            self.setCentralWidget(central)
            self.setMenuBar(self.create_menu_bar())
            self.load_settings()
            self.load_game_from_fen(START_FEN)
            self.game_mode = "setup"
            self.show_navigation_panel()
            self.set_status("摆谱：已载入初始棋盘，可点击棋子走子")
            QTimer.singleShot(0, self.enforce_startup_layout)
            QTimer.singleShot(120, self.enforce_startup_layout)

        def create_menu_bar(self):
            menu_bar = QMenuBar()

            game_menu = menu_bar.addMenu("棋局(&G)")
            import_record_action = QAction("导入棋谱(&I)", self)
            import_record_action.triggered.connect(self.import_game_record)
            game_menu.addAction(import_record_action)
            save_record_action = QAction("保存棋谱(&V)", self)
            save_record_action.triggered.connect(self.save_game_record)
            game_menu.addAction(save_record_action)
            import_fen_action = QAction("导入 FEN(&F)", self)
            import_fen_action.triggered.connect(self.import_fen_position)
            game_menu.addAction(import_fen_action)
            setup_record_action = QAction("摆谱(&S)", self)
            setup_record_action.triggered.connect(self.start_position_setup_mode)
            game_menu.addAction(setup_record_action)
            new_board_action = QAction("新棋盘(&N)", self)
            new_board_action.triggered.connect(self.start_new_board_mode)
            game_menu.addAction(new_board_action)
            game_menu.addSeparator()
            copy_fen_action = QAction("复制当前 FEN(&C)", self)
            copy_fen_action.triggered.connect(self.copy_fen)
            self.copy_fen_action = copy_fen_action
            game_menu.addAction(copy_fen_action)
            game_menu.addSeparator()
            exit_action = QAction("退出(&X)", self)
            exit_action.triggered.connect(self.close)
            game_menu.addAction(exit_action)

            self.calculate_button = None
            start_action = QAction("开始(&S)", self)
            start_action.setStatusTip("开始或停止屏幕监控。")
            start_action.triggered.connect(self.toggle_helper)
            self.start_button = start_action
            reload_action = QAction("重新读取(&R)", self)
            reload_action.setStatusTip("重新读取设置和校准数据。")
            reload_action.triggered.connect(self.reload_helper)
            self.reload_button = reload_action

            options_menu = menu_bar.addMenu("连线(&L)")
            options_menu.addAction(start_action)
            options_menu.addAction(reload_action)
            options_menu.addSeparator()
            settings_action = QAction("设置(&S)", self)
            settings_action.triggered.connect(self.show_settings_dialog)
            options_menu.addAction(settings_action)
            tips_action = QAction("使用提示(&T)", self)
            tips_action.triggered.connect(self.show_usage_tips)
            options_menu.addAction(tips_action)

            help_menu = menu_bar.addMenu("帮助(&H)")
            engine_action = QAction("检测引擎(&E)", self)
            engine_action.triggered.connect(self.check_engine_status)
            help_menu.addAction(engine_action)
            about_action = QAction("关于象棋助手(&A)", self)
            about_action.triggered.connect(lambda: self.set_status("象棋助手 - 屏幕识别与 Pikafish 分析"))
            help_menu.addAction(about_action)
            return menu_bar

        def show_analysis_dialog(self):
            self.analysis_box.setPlainText(self.analysis_text)
            self.analysis_dialog.show()
            self.analysis_dialog.raise_()
            self.analysis_dialog.activateWindow()
            if hasattr(self, "pv_analysis_action"):
                self.pv_analysis_action.setChecked(True)

        def hide_analysis_dialog(self):
            self.analysis_dialog.hide()
            if hasattr(self, "pv_analysis_action"):
                self.pv_analysis_action.setChecked(False)

        def toggle_pv_analysis_window(self, checked):
            if checked:
                self.analysis_box.setPlainText(self.analysis_text)
                self.analysis_dialog.show()
                self.analysis_dialog.raise_()
                self.analysis_dialog.activateWindow()
                self.set_status("已显示 PV分析")
            else:
                self.analysis_dialog.hide()
                self.set_status("已隐藏 PV分析")

        def toggle_window_panel(self, name, checked):
            if checked:
                self.show_side_panel(name, f"{name}窗口已打开。\n后续可以在这里继续接入完整{name}内容。")
                self.set_status(f"已显示{name}")
            else:
                self.hide_side_panel_if_title(name)
                self.set_status(f"已隐藏{name}")

        def show_side_panel(self, title, text):
            self.side_panel_title.setText(title)
            self.side_panel_text.setPlainText(text)
            self.nav_buttons_widget.setVisible(title == "导航")
            self.nav_controls_widget.setVisible(title == "导航")
            self.refresh_navigation_buttons()
            self.side_panel.show()
            self.setMinimumWidth(760)
            self.setMinimumHeight(580)
            target_height = max(self.height(), 580)
            if self.width() < 820 or self.height() < target_height:
                self.resize(max(self.width(), 820), target_height)
            self.update_board_display(self.current_grid)

        def enforce_startup_layout(self):
            self.resize(max(self.width(), 820), max(self.height(), 580))
            self.update_board_display(self.current_grid)

        def hide_side_panel_if_title(self, title):
            if self.side_panel_title.text() == title:
                self.side_panel.hide()
                self.setMinimumWidth(520)
                if self.width() > 560:
                    self.resize(520, self.height())

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
                self.show_side_panel("打分图", self.score_chart_text())
                self.set_status("已显示打分图")
            else:
                self.hide_side_panel_if_title("打分图")
                self.set_status("已隐藏打分图")

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
            if hasattr(self, "opening_book_action"):
                self.opening_book_action.setChecked(True)
            self.side_panel_title.setText("开局库")
            self.side_panel.show()
            self.setMinimumWidth(720)
            if self.width() < 820:
                self.resize(820, self.height())
            self.load_opening_book_panel()
            self.set_status("已选择开局库")

        def find_opening_book_path(self):
            for path in candidate_paths("book.xqb"):
                if os.path.exists(path):
                    return path
            return None

        def load_opening_book_panel(self):
            if not self.opening_book_path:
                self.side_panel_text.setPlainText(
                    "没有找到开局库文件。\n\n请把 book.xqb 放到程序文件夹，也就是 XiangqiHelper.exe 旁边。"
                )
                return

            text = [
                "开局库文件：",
                self.opening_book_path,
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
            if not self.opening_book_path or not fen:
                return []
            if not os.path.exists(self.opening_book_path):
                return []
            results = []
            try:
                key, mirror_ud, mirror_lr, rows, cols = self.xqbook_fen_to_key(fen)
                uri = pathlib.Path(self.opening_book_path).as_uri() + "?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
            except Exception:
                return []
            try:
                query = (
                    "select Id,Move,Score,Win,Draw,Lost,Valid,Memo "
                    "from book where key=? order by Score desc, Win desc, Id asc"
                )
                for row in conn.execute(query, (sqlite3.Binary(key),)):
                    _, raw_move, score, win, draw, lost, valid, memo = row
                    src_row, src_col, dst_row, dst_col = self.xqbook_mirror_move(
                        int(raw_move), mirror_ud, mirror_lr, rows, cols
                    )
                    results.append({
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
            return results

        def score_chart_text(self):
            lines = ["打分图", ""]
            if not self.history_fens:
                lines.append("还没有棋谱历史。")
                return "\n".join(lines)
            lines.append("当前先显示每步的开局库分数摘要；后续可以接入逐步引擎评分曲线。")
            lines.append("")
            for index, fen in enumerate(self.history_fens):
                moves = self.query_xqbook(fen)
                score = moves[0]["score"] if moves else 0
                bar_len = min(20, abs(int(score)) // 100)
                bar = ("+" if score >= 0 else "-") * bar_len
                marker = " <- 当前" if index == self.history_index else ""
                lines.append(f"{index:>3}. {score:>5} {bar}{marker}")
            return "\n".join(lines)

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

        def refresh_navigation_buttons(self):
            total = max(0, len(self.history_fens) - 1)
            can_go_back = self.history_index > 0
            can_go_forward = self.history_index < total
            self.nav_first_button.setEnabled(can_go_back)
            self.nav_prev_button.setEnabled(can_go_back)
            self.nav_next_button.setEnabled(can_go_forward)
            self.nav_last_button.setEnabled(can_go_forward)

        def navigation_text(self):
            lines = []
            total = max(0, len(self.history_fens) - 1)
            lines.append(f"当前步数：{self.history_index} / {total}")
            lines.append(f"当前走棋：{'红方' if self.game_side == 'red' else '黑方'}")
            search_value = (
                f"深度 {self.ai_depth_spin.value()}"
                if self.search_mode_combo.currentText() == "深度优先"
                else f"时间 {self.ai_time_spin.value():.1f} 秒"
            )
            auto_sides = []
            if self.red_ai_checkbox.isChecked():
                auto_sides.append("红方")
            if self.black_ai_checkbox.isChecked():
                auto_sides.append("黑方")
            lines.append(f"电脑走棋：{('、'.join(auto_sides) if auto_sides else '未开启')}    {search_value}")
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
                elif self.side_panel_title.text() == "打分图":
                    self.side_panel_text.setPlainText(self.score_chart_text())
            self.set_status(f"已跳到第 {self.history_index} 步")

        def show_settings_dialog(self):
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()

        def hide_settings_dialog(self):
            self.settings_dialog.hide()

        def save_settings_and_close(self):
            self.save_settings(silent=True)
            self.settings_dialog.hide()
            self.set_status("设置已保存")

        def mode_single_calculation(self):
            self.auto_move_checkbox.setChecked(False)
            self.save_settings(silent=True)
            self.calculate_now()
            self.set_status("单次计算模式：正在计算当前棋盘")

        def mode_screen_monitor(self):
            if not self.helper_running():
                self.start_helper()
            else:
                self.set_status("屏幕监控模式已经在运行")

        def mode_analysis_only(self):
            self.auto_move_checkbox.setChecked(False)
            self.save_settings(silent=True)
            if not self.helper_running():
                self.start_helper()
            self.set_status("只分析不走模式：不会自动点击棋子")

        def mode_auto_move(self):
            self.auto_move_checkbox.setChecked(True)
            self.save_settings(silent=True)
            if not self.helper_running():
                self.start_helper()
            self.set_status("自动走棋模式：识别稳定后会自动点击推荐走法")

        def mode_analysis_detail(self):
            self.show_analysis_dialog()
            if not self.current_move and not (self.calculate_thread and self.calculate_thread.is_alive()):
                self.calculate_now()
            self.set_status("分析详情模式：显示实时分数和候选走法")

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
            self.board_flipped = side == "black"
            self.update_board_display(self.current_grid)
            self.set_status(f"人机对弈：你执{'红' if side == 'red' else '黑'}")
            if self.game_side != self.player_side:
                self.request_ai_move()

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
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
                parsed = parse_pgn(text)
                self.load_game_from_fen(parsed["fen"])
                self.game_mode = "record"
                self.history_fens = parsed.get("fens", [parsed["fen"]])
                self.history_moves = [item["uci"] for item in parsed.get("moves", [])]
                self.history_index = len(self.history_fens) - 1
                self.current_move = self.history_moves[-1] if self.history_moves else ""
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
            self.show_text_dialog("PGN 复盘", "PGN 导入和前后步复盘会作为下一步继续移植。")
            self.set_status("PGN 复盘模式待继续移植")

        def start_online_mode(self):
            self.show_text_dialog("在线对弈", "在线对弈需要服务器登录、房间和匹配系统，后续继续移植。")
            self.set_status("在线对弈模式待继续移植")

        def load_game_from_fen(self, fen):
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
            self.history_index = 0
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            if self.side_panel.isVisible() and self.side_panel_title.text() == "开局库":
                self.load_opening_book_panel()

        def load_empty_board(self):
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
            self.history_index = 0
            self.update_board_display(self.current_grid)
            self.refresh_navigation_buttons()
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())

        def display_square(self, row, col):
            if self.board_flipped:
                return 9 - row, 8 - col
            return row, col

        def logical_square(self, row, col):
            if self.board_flipped:
                return 9 - row, 8 - col
            return row, col

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
            self.game_board[row][col] = piece
            self.current_grid = [line[:] for line in self.game_board]
            self.current_fen = board_to_fen(self.game_board, self.game_side)
            self.current_move = ""
            self.selected_square = None
            self.legal_target_moves = {}
            self.history_fens = [self.current_fen]
            self.history_moves = []
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
            col = int((x - left) / width * 9)
            row = int((y - top) / height * 10)
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
            self.game_board = apply_uci(self.game_board, move)
            self.move_history.append(move)
            self.history_moves.append(move)
            self.game_side = opponent(self.game_side)
            self.current_fen = board_to_fen(self.game_board, self.game_side)
            self.history_fens.append(self.current_fen)
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
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "打分图":
                self.side_panel_text.setPlainText(self.score_chart_text())
            if self.game_mode == "ai" and self.game_side != self.player_side:
                self.request_ai_move()
            elif self.is_computer_turn():
                self.request_ai_move()
            elif self.game_mode == "trainer":
                self.calculate_local_position("训练模式：继续找最佳走法")

        def on_computer_side_changed(self):
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
            if self.search_mode_combo.currentText() == "深度优先":
                return None, self.ai_depth_spin.value()
            return int(round(self.ai_time_spin.value() * 1000)), None

        def request_ai_move(self):
            if self.ai_thread and self.ai_thread.is_alive():
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
                while engine.bestmove is None and time.time() < deadline:
                    time.sleep(0.05)
                move = engine.bestmove
                if move and move != "(none)" and side == self.game_side:
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
            self.analysis_text = "正在分析当前棋盘..."
            self.show_analysis_dialog()
            self.set_status(status_text)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.update_board_display(self.current_grid)

        def update_fen(self, fen):
            self.current_fen = fen or ""
            if self.side_panel.isVisible() and self.side_panel_title.text() == "开局库":
                self.load_opening_book_panel()
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())
            elif self.side_panel.isVisible() and self.side_panel_title.text() == "打分图":
                self.side_panel_text.setPlainText(self.score_chart_text())

        def update_analysis_display(self, text):
            self.analysis_text = text or "还没有分析结果。"
            if self.analysis_dialog.isVisible():
                self.analysis_box.setPlainText(self.analysis_text)

        def update_record_display(self, text):
            self.online_record_text = text or ""
            if self.side_panel.isVisible() and self.side_panel_title.text() == "导航":
                self.side_panel_text.setPlainText(self.navigation_text())

        def copy_fen(self):
            if not self.current_fen:
                self.set_status("还没有可复制的 FEN，请先计算或开始识别。")
                return
            QApplication.clipboard().setText(self.current_fen)
            self.set_status("已复制当前 FEN")

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
                "使用提示",
                "1. 第一次使用先打开 连线 -> 设置 -> 校准棋盘。\n"
                "2. 想持续跟随屏幕棋盘，点 连线 -> 开始。\n"
                "3. 计算时会显示实时分数和候选走法。\n"
                "4. 自动走棋建议确认识别稳定以后再开启。",
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

                if self.current_move:
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
            self.min_time_spin.setValue(
                self.config_float(config, "auto_move_search_time_min", 2.0)
            )
            self.max_time_spin.setValue(
                self.config_float(config, "auto_move_search_time_max", 5.0)
            )
            self.auto_move_checkbox.setChecked(
                self.config_bool(config, "auto_move", True)
            )
            self.engine_threads_spin.setValue(
                self.config_int(config, "engine_threads", 1)
            )
            self.auto_next_checkbox.setChecked(
                self.config_bool(config, "auto_start_next_game", False)
            )

        def save_settings(self, silent=False):
            min_sec, max_sec = sorted((
                self.min_time_spin.value(),
                self.max_time_spin.value(),
            ))
            self.min_time_spin.setValue(min_sec)
            self.max_time_spin.setValue(max_sec)
            try:
                update_config({
                    "auto_move_search_time_min": round(min_sec, 2),
                    "auto_move_search_time_max": round(max_sec, 2),
                    "engine_threads": self.engine_threads_spin.value(),
                    "auto_move": self.auto_move_checkbox.isChecked(),
                    "auto_start_next_game": self.auto_next_checkbox.isChecked(),
                    "next_game_fullscreen_scan": True,
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

        def set_status(self, text):
            self.status.setText(text)

        def helper_running(self):
            return self.worker_thread is not None and self.worker_thread.is_alive()

        def toggle_helper(self):
            if self.helper_running():
                self.stop_helper()
            else:
                self.start_helper()

        def start_helper(self):
            if self.worker_thread and self.worker_thread.is_alive():
                self.set_status("已经在运行")
                return
            if not calibration_exists():
                self.set_status("缺少 config.json，请先点击“校准棋盘”。")
                self.start_button.setText("开始(&S)")
                self.start_button.setEnabled(True)
                return

            self.start_button.setText("停止(&S)")
            self.set_status("正在启动...")
            self.update_record_display("")
            self.worker_thread = threading.Thread(target=self.run_worker, daemon=True)
            self.worker_thread.start()

        def stop_helper(self):
            if self.app_worker is None:
                self.set_status("还在启动中...")
                return
            self.start_button.setEnabled(False)
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
                self.set_status("请先点击“开始”")
                return
            self.app_worker.request_reload()
            self.set_status("正在重新读取...")

        def check_engine_status(self):
            self.set_status("正在检测引擎...")
            threading.Thread(target=self.run_engine_check, daemon=True).start()

        def run_engine_check(self):
            engine = None
            try:
                engine = Engine()
                if engine.process and engine.process.poll() is None:
                    self.signals.status.emit("引擎正常，可以计算")
                else:
                    self.signals.status.emit("引擎没有正常运行")
            except Exception as e:
                self.signals.status.emit(f"引擎检测失败：{e}")
            finally:
                if engine is not None:
                    engine.quit()

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
            self.start_button.setEnabled(True)
            self.start_button.setText("开始(&S)")
            if not self.status.text().startswith("错误："):
                self.set_status("已停止")

        def closeEvent(self, event):
            self.closing = True
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
