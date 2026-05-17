import os
import sys
import time
import signal
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
from calibrate import run_turn_indicator_recognition
from calibrate import run_calibration as run_calibration_flow
from calibrate import update_config
from app_paths import candidate_paths
from xiangqi_core import (
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
TURN_GREEN_RATIO_DEFAULT = 0.015
TURN_GREEN_DROP_DEFAULT = 0.002
TURN_INDICATOR_STABLE_FRAMES_DEFAULT = 3
TURN_INACTIVE_STABLE_FRAMES_DEFAULT = 3


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


def _turn_indicator_recognition_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--recognize-turn", "--no-prompt"]
    return [sys.executable, os.path.join(_app_dir(), "main.py"), "--recognize-turn", "--no-prompt"]


class App:
    def __init__(self, exit_on_error=True, hide_dock_icon=True):
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
        self.last_turn_active = False
        self.turn_action_done = False
        self.turn_inactive_count = 0
        self.next_game_last_click_at = 0.0
        self.next_game_last_scan_at = 0.0
        self.waiting_for_new_game = False
        self.turn_indicator_last_ratio = None
        self.turn_indicator_missing_logged_at = 0.0
        self._next_game_template = None
        self._next_game_template_path = None
        self._next_game_template_mtime = None
        self._next_game_missing_logged_at = 0.0
        self._recalc = threading.Event()
        self._reset_requested = threading.Event()
        self._reload_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._mouse_available = True
        try:
            _load_mouse_controls(hide_dock_icon=hide_dock_icon)
            self._mouse = MouseController()
        except Exception as e:
            if self._config_bool("auto_move", default=True):
                raise RuntimeError(
                    "Mouse control is unavailable. On Linux, run under an X11 session "
                    "or set auto_move to false in config.json."
                ) from e
            self._mouse_available = False
            self._mouse = _NoMouseController()
            print(f"[Mouse] unavailable; continuing without auto move: {e}")

    def request_reload(self):
        self._reload_requested.set()
        self._reset_requested.set()

    def stop(self):
        self._stop_requested.set()
        self._reset_requested.set()
        try:
            self.eng.send_cmd("stop")
        except Exception:
            pass

    def _should_cancel_search(self):
        return self._reset_requested.is_set() or self._stop_requested.is_set()

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
        self.last_turn_active = False
        self.turn_action_done = False
        self.turn_inactive_count = 0
        self.waiting_for_new_game = False
        self.turn_indicator_last_ratio = None

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

    def _turn_indicator_region(self, screen_img):
        required = (
            "turn_indicator_x1",
            "turn_indicator_y1",
            "turn_indicator_x2",
            "turn_indicator_y2",
        )
        if not all(key in self.vis.config for key in required):
            return None

        screen_left = self._config_float(
            "turn_indicator_screen_left",
            self._config_float("screen_left", 0.0),
        )
        screen_top = self._config_float(
            "turn_indicator_screen_top",
            self._config_float("screen_top", 0.0),
        )
        x1 = self._config_float("turn_indicator_x1", 0.0) - screen_left
        y1 = self._config_float("turn_indicator_y1", 0.0) - screen_top
        x2 = self._config_float("turn_indicator_x2", 0.0) - screen_left
        y2 = self._config_float("turn_indicator_y2", 0.0) - screen_top
        x1, x2 = sorted((int(round(x1)), int(round(x2))))
        y1, y2 = sorted((int(round(y1)), int(round(y2))))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(screen_img.shape[1], x2)
        y2 = min(screen_img.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return x1, y1, x2, y2

    def _turn_indicator_missing(self):
        now = time.time()
        if now - self.turn_indicator_missing_logged_at > 5.0:
            self.turn_indicator_missing_logged_at = now
            dprint("[Turn avatar] area missing. Click Recognize Turn first.")

    def _turn_indicator_green_ratio(self, screen_img):
        region = self._turn_indicator_region(screen_img)
        if region is None:
            self._turn_indicator_missing()
            return None

        x1, y1, x2, y2 = region
        roi = screen_img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = (
            self._config_int("turn_green_h_min", 35),
            self._config_int("turn_green_s_min", 45),
            self._config_int("turn_green_v_min", 45),
        )
        upper = (
            self._config_int("turn_green_h_max", 95),
            255,
            255,
        )
        mask = cv2.inRange(hsv, lower, upper)
        return float(cv2.countNonZero(mask)) / float(mask.size)

    def _own_turn_indicator_active(self, screen_img):
        if not self._config_bool("auto_turn_indicator", default=True):
            return False

        ratio = self._turn_indicator_green_ratio(screen_img)
        if ratio is None:
            return False

        previous = self.turn_indicator_last_ratio
        self.turn_indicator_last_ratio = ratio

        threshold = self._config_float("turn_green_min_ratio", TURN_GREEN_RATIO_DEFAULT)
        if ratio < threshold:
            return False

        if self._config_bool("turn_indicator_trigger_on_visible", default=True):
            return True

        drop = self._config_float("turn_green_drop_ratio", TURN_GREEN_DROP_DEFAULT)
        if previous is None or previous < threshold:
            return True
        return ratio <= previous - drop

    def _stable_frame_count_needed(self):
        return max(
            1,
            self._config_int(
                "turn_indicator_min_stable_frames",
                TURN_INDICATOR_STABLE_FRAMES_DEFAULT,
            ),
        )

    def _turn_inactive_frame_count_needed(self):
        return max(
            1,
            self._config_int(
                "turn_indicator_inactive_frames",
                TURN_INACTIVE_STABLE_FRAMES_DEFAULT,
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

        x_offset = float(self.vis.config.get("auto_move_mouse_away_offset_x", 80.0))
        y_offset = float(self.vis.config.get("auto_move_mouse_away_offset_y", 80.0))
        return (
            float(self.vis.config["bottom_right_x"])
            + float(self.vis.config.get("screen_left", 0))
            + x_offset,
            float(self.vis.config["bottom_right_y"])
            + float(self.vis.config.get("screen_top", 0))
            + y_offset,
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
                return
            fen, _, flipped = position

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

            if not self._own_turn_indicator_active(img):
                if self.last_turn_active:
                    self.turn_inactive_count += 1
                    needed = self._turn_inactive_frame_count_needed()
                    if self.turn_inactive_count < needed:
                        if self.turn_action_done:
                            self._print_status_once(
                                "Move already played; waiting for turn to end..."
                            )
                        else:
                            self._print_status_once(
                                f"Confirming turn end {self.turn_inactive_count}/{needed}..."
                            )
                        return
                    dprint("[Turn avatar] turn ended; ready for next turn.")
                self.last_turn_active = False
                self.turn_action_done = False
                self.turn_inactive_count = 0
                self._print_status_once("Waiting for my turn...")
                return

            if not self.last_turn_active:
                dprint("[Turn avatar] my turn detected.")
            self.last_turn_active = True
            self.turn_inactive_count = 0

            force = self._recalc.is_set()
            if self.turn_action_done and not force:
                self._print_status_once("Move already played; waiting for turn to end...")
                return

            if self.stable_count < min_stable:
                self._print_status_once(
                    f"My turn detected; waiting stable board {self.stable_count}/{min_stable}..."
                )
                return

            self._recalc.clear()
            moved_or_reported = self._calculate_and_print(fen, flipped)
            if moved_or_reported:
                self.turn_action_done = True

        except Exception as e:
            if DEBUG:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()

    def _calculate_and_print(self, fen, flipped):
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
                gui_pv = flip_move(pv) if flipped else pv
                if gui_pv != last_pv:
                    last_pv = gui_pv
            time.sleep(0.05)

        bm = self.eng.bestmove
        if not bm or bm == "(none)":
            dprint("[Best move] none")
            return False

        gui_move = flip_move(bm) if flipped else bm
        dprint(f"[Best move] engine={bm} gui={gui_move}")
        print(f"Best move: {gui_move}")
        self.last_status = None
        moved = self._auto_move(fen, gui_move)
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
    from PyQt5.QtCore import QObject, Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class GuiSignals(QObject):
        status = pyqtSignal(str)
        stopped = pyqtSignal()
        calibration_done = pyqtSignal()
        recognition_done = pyqtSignal()

    class HelperWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.app_worker = None
            self.worker_thread = None
            self.calibrate_thread = None
            self.recognize_thread = None
            self.closing = False
            self.signals = GuiSignals()
            self.signals.status.connect(self.set_status)
            self.signals.stopped.connect(self.on_stopped)
            self.signals.calibration_done.connect(self.on_calibration_done)
            self.signals.recognition_done.connect(self.on_recognition_done)

            self.setWindowTitle("Xiangqi Helper")
            self.setMinimumWidth(420)

            self.title = QLabel("Xiangqi Helper")
            self.title.setAlignment(Qt.AlignCenter)
            self.title.setStyleSheet("font-size: 20px; font-weight: 600;")

            self.status = QLabel("Stopped")
            self.status.setAlignment(Qt.AlignCenter)

            self.start_button = QPushButton("Start")
            self.reload_button = QPushButton("Reload")
            self.calibrate_button = QPushButton("Calibrate")
            self.exit_button = QPushButton("Exit")

            self.min_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.max_time_spin = self.make_seconds_spin(0.1, 60.0)
            self.turn_indicator_checkbox = QCheckBox("Use turn avatar")
            self.recognize_turn_button = QPushButton("Recognize Turn")
            self.auto_next_checkbox = QCheckBox("Auto start next game")
            self.active_next_scan_checkbox = QCheckBox("Active next screen scan")
            self.recognize_next_button = QPushButton("Recognize Next")
            self.save_settings_button = QPushButton("Save Settings")

            self.start_button.clicked.connect(self.toggle_helper)
            self.reload_button.clicked.connect(self.reload_helper)
            self.calibrate_button.clicked.connect(self.calibrate_helper)
            self.exit_button.clicked.connect(self.close)
            self.recognize_turn_button.clicked.connect(self.recognize_turn_indicator)
            self.recognize_next_button.clicked.connect(self.recognize_next_game)
            self.save_settings_button.clicked.connect(self.save_settings)

            buttons = QHBoxLayout()
            buttons.addWidget(self.start_button)
            buttons.addWidget(self.reload_button)
            buttons.addWidget(self.calibrate_button)
            buttons.addWidget(self.exit_button)

            move_time_row = QHBoxLayout()
            move_time_row.addWidget(self.min_time_spin)
            move_time_row.addWidget(QLabel("to"))
            move_time_row.addWidget(self.max_time_spin)

            settings = QGroupBox("Config")
            settings_form = QFormLayout()
            settings_form.addRow("Move time", move_time_row)
            settings_form.addRow(self.turn_indicator_checkbox)
            settings_form.addRow("Turn avatar", self.recognize_turn_button)
            settings_form.addRow(self.auto_next_checkbox)
            settings_form.addRow(self.active_next_scan_checkbox)
            settings_form.addRow("Next button", self.recognize_next_button)
            settings_form.addRow("", self.save_settings_button)
            settings.setLayout(settings_form)

            layout = QVBoxLayout()
            layout.addWidget(self.title)
            layout.addWidget(self.status)
            layout.addLayout(buttons)
            layout.addWidget(settings)
            self.setLayout(layout)
            self.load_settings()

        def make_seconds_spin(self, minimum, maximum):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" sec")
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

        def load_settings(self):
            config = load_config()
            self.min_time_spin.setValue(
                self.config_float(config, "auto_move_search_time_min", 2.0)
            )
            self.max_time_spin.setValue(
                self.config_float(config, "auto_move_search_time_max", 5.0)
            )
            self.turn_indicator_checkbox.setChecked(
                self.config_bool(config, "auto_turn_indicator", True)
            )
            self.auto_next_checkbox.setChecked(
                self.config_bool(config, "auto_start_next_game", False)
            )
            self.active_next_scan_checkbox.setChecked(
                self.config_bool(config, "next_game_fullscreen_scan", True)
            )

        def save_settings(self):
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
                    "auto_turn_indicator": self.turn_indicator_checkbox.isChecked(),
                    "auto_start_next_game": self.auto_next_checkbox.isChecked(),
                    "next_game_fullscreen_scan": self.active_next_scan_checkbox.isChecked(),
                })
            except Exception as e:
                self.set_status(f"Settings error: {e}")
                return
            if self.app_worker is not None:
                self.app_worker.request_reload()
                self.set_status("Settings saved. Reloading...")
            else:
                self.set_status("Settings saved")

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
                self.set_status("Already running")
                return
            if not calibration_exists():
                self.set_status("config.json missing. Click Calibrate first.")
                self.start_button.setText("Start")
                self.start_button.setEnabled(True)
                return

            self.start_button.setText("Stop")
            self.set_status("Starting...")
            self.worker_thread = threading.Thread(target=self.run_worker, daemon=True)
            self.worker_thread.start()

        def stop_helper(self):
            if self.app_worker is None:
                self.set_status("Still starting...")
                return
            self.start_button.setEnabled(False)
            self.set_status("Stopping...")
            self.app_worker.stop()

        def run_worker(self):
            try:
                self.app_worker = App(exit_on_error=False, hide_dock_icon=False)
                if self.closing:
                    self.app_worker.stop()
                    self.app_worker.eng.quit()
                    return
                self.signals.status.emit("Running")
                self.app_worker.run(use_keyboard=False)
            except Exception as e:
                self.signals.status.emit(f"Error: {e}")
            finally:
                self.app_worker = None
                self.signals.stopped.emit()

        def reload_helper(self):
            if self.app_worker is None:
                self.set_status("Start helper first")
                return
            self.app_worker.request_reload()
            self.set_status("Reloading...")

        def calibrate_helper(self):
            if self.calibrate_thread and self.calibrate_thread.is_alive():
                self.set_status("Calibration already running")
                return
            self.calibrate_button.setEnabled(False)
            self.set_status("Opening calibration...")
            self.calibrate_thread = threading.Thread(target=self.run_calibration, daemon=True)
            self.calibrate_thread.start()

        def recognize_next_game(self):
            if self.recognize_thread and self.recognize_thread.is_alive():
                self.set_status("Recognition already running")
                return
            self.recognize_turn_button.setEnabled(False)
            self.recognize_next_button.setEnabled(False)
            self.set_status("Opening next button recognition...")
            self.recognize_thread = threading.Thread(
                target=self.run_next_game_recognition,
                daemon=True,
            )
            self.recognize_thread.start()

        def recognize_turn_indicator(self):
            if self.recognize_thread and self.recognize_thread.is_alive():
                self.set_status("Recognition already running")
                return
            self.recognize_turn_button.setEnabled(False)
            self.recognize_next_button.setEnabled(False)
            self.set_status("Opening turn avatar recognition...")
            self.recognize_thread = threading.Thread(
                target=self.run_turn_indicator_recognition,
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
                    self.signals.status.emit("Calibration saved")
                else:
                    self.signals.status.emit(f"Calibration exited: {result.returncode}")
            except Exception as e:
                self.signals.status.emit(f"Calibration error: {e}")
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
                    self.signals.status.emit("Next button saved")
                else:
                    self.signals.status.emit(f"Recognition exited: {result.returncode}")
            except Exception as e:
                self.signals.status.emit(f"Recognition error: {e}")
            finally:
                self.signals.recognition_done.emit()

        def run_turn_indicator_recognition(self):
            try:
                if self.app_worker is not None:
                    self.app_worker.stop()
                if self.worker_thread is not None and self.worker_thread.is_alive():
                    self.worker_thread.join(timeout=3.0)

                result = subprocess.run(
                    _turn_indicator_recognition_command(),
                    cwd=_app_dir(),
                    check=False,
                )
                if result.returncode == 0:
                    self.signals.status.emit("Turn avatar saved")
                else:
                    self.signals.status.emit(f"Recognition exited: {result.returncode}")
            except Exception as e:
                self.signals.status.emit(f"Recognition error: {e}")
            finally:
                self.signals.recognition_done.emit()

        def on_calibration_done(self):
            self.calibrate_button.setEnabled(True)
            self.load_settings()

        def on_recognition_done(self):
            self.recognize_turn_button.setEnabled(True)
            self.recognize_next_button.setEnabled(True)
            self.load_settings()

        def on_stopped(self):
            self.start_button.setEnabled(True)
            self.start_button.setText("Start")
            if not self.status.text().startswith("Error:"):
                self.set_status("Stopped")

        def closeEvent(self, event):
            self.closing = True
            if self.app_worker is not None:
                self.app_worker.stop()
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=2.0)
            if self.calibrate_thread is not None:
                self.calibrate_thread.join(timeout=1.0)
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


def run_recognize_turn_mode():
    run_turn_indicator_recognition(prompt="--no-prompt" not in sys.argv)


def main():
    if "--calibrate" in sys.argv:
        run_calibrate_mode()
    elif "--recognize-next" in sys.argv:
        run_recognize_next_mode()
    elif "--recognize-turn" in sys.argv:
        run_recognize_turn_mode()
    elif "--cli" in sys.argv:
        run_cli()
    else:
        sys.exit(launch_gui())


if __name__ == '__main__':
    main()
