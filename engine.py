import subprocess
import threading
import time
import os
import platform
import sys
import stat

from app_paths import candidate_paths, resolve_resource_path
from calibrate import load_config


DEFAULT_ENGINE_PATH = None
FALLBACK_ENGINE_PATH = "./pikafish.exe" if os.name == "nt" else "./pikafish"
DEFAULT_ENGINE_DIR = "./engines"
DEFAULT_NNUE_PATH = "./pikafish.nnue"


def _resolve_project_path(path):
    return resolve_resource_path(path)


def _project_path_candidates(path):
    return candidate_paths(path)


def _windows_cpu_flags():
    if os.name != "nt":
        return set()

    try:
        import ctypes
    except Exception:
        return set()

    features = {
        37: "sse41",
        38: "sse42",
        39: "avx",
        40: "avx2",
        41: "avx512f",
    }
    flags = set()
    kernel32 = ctypes.windll.kernel32
    for feature_id, name in features.items():
        try:
            if kernel32.IsProcessorFeaturePresent(feature_id):
                flags.add(name)
        except Exception:
            pass

    if "sse42" in flags:
        flags.add("popcnt")
    return flags


def _cpu_flags():
    if platform.machine().lower() in ("arm64", "aarch64"):
        return set()

    flags = set()
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", errors="ignore") as f:
                text = f.read().lower().replace("_", "")
            for line in text.splitlines():
                if line.startswith(("flags", "features")):
                    flags.update(line.split(":", 1)[1].split())
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            text = subprocess.check_output(
                ["sysctl", "-a"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).lower().replace("_", "")
            for line in text.splitlines():
                if "machdep.cpu.features" in line or "machdep.cpu.leaf7features" in line:
                    flags.update(line.split(":", 1)[1].split())
        except Exception:
            pass
    elif os.name == "nt":
        flags.update(_windows_cpu_flags())

    normalized = set(flags)
    if "sse4.1" in flags:
        normalized.add("sse41")
    if "avx512vnni" in flags:
        normalized.add("vnni512")
    return normalized


def _platform_engine_dir_name():
    if sys.platform.startswith("linux"):
        return "Linux"
    if sys.platform == "darwin":
        return "MacOS"
    if os.name == "nt":
        return "Windows"
    return ""


def _score_engine(path, flags):
    text = str(path).lower().replace("_", "-")
    machine = platform.machine().lower()
    score = 0

    if _platform_engine_dir_name().lower() not in text:
        score -= 20

    if machine in ("arm64", "aarch64"):
        if any(key in text for key in ("apple-silicon", "arm64", "aarch64", "armv8")):
            score += 100
        if "x86" in text or "avx" in text or "sse" in text or "bmi" in text:
            return None
        return score

    if os.name == "nt" and not text.endswith(".exe"):
        return None
    if os.name != "nt" and text.endswith(".exe"):
        return None

    known = bool(flags)
    if not known:
        if "vnni512" in text:
            return score + 100
        if "avx512" in text:
            return score + 90
        if "bmi2" in text:
            return score + 80
        if "avx2" in text:
            return score + 70
        if "modern" in text or "avxvnni" in text:
            return score + 60
        if "sse41-popcnt" in text or "sse4" in text:
            return score + 40
        if "ssse3" in text:
            return score + 30
        return score + 20

    if "vnni512" in text:
        if known and "avx512f" not in flags:
            return None
        return score + 100
    if "avx512icl" in text:
        if known and "avx512f" not in flags:
            return None
        return score + 95
    if "avx512" in text:
        if known and "avx512f" not in flags:
            return None
        return score + 90
    if "bmi2" in text:
        if known and "bmi2" not in flags:
            return None
        return score + 80
    if "avx2" in text:
        if known and "avx2" not in flags:
            return None
        return score + 70
    if "modern" in text or "avxvnni" in text:
        if known and "avxvnni" in text and "avxvnni" not in flags:
            return None
        return score + 60
    if "sse41-popcnt" in text or "sse4" in text:
        if known and not (("sse41" in flags or "sse4.1" in flags) and "popcnt" in flags):
            return None
        return score + 40
    if "ssse3" in text:
        return score + 30

    return score + 20


class Engine:
    def __init__(self, engine_path=DEFAULT_ENGINE_PATH, nnue_path=DEFAULT_NNUE_PATH):
        self.engine_path = engine_path
        self.nnue_path = nnue_path
        self.process = None
        self._start_process()

    def _engine_candidates(self):
        if self.engine_path:
            return [_resolve_project_path(self.engine_path)]

        paths = []
        for engine_dir in _project_path_candidates(DEFAULT_ENGINE_DIR):
            if not os.path.isdir(engine_dir):
                continue
            platform_dir = os.path.join(engine_dir, _platform_engine_dir_name())
            search_root = platform_dir if os.path.isdir(platform_dir) else engine_dir
            for root, _, files in os.walk(search_root):
                for name in files:
                    if name.lower().startswith("pikafish"):
                        paths.append(os.path.join(root, name))

        for root in _project_path_candidates("."):
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                if name.lower().startswith("pikafish"):
                    paths.append(os.path.join(root, name))

        for fallback in _project_path_candidates(FALLBACK_ENGINE_PATH):
            if os.path.exists(fallback):
                paths.append(fallback)

        flags = _cpu_flags()
        scored = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            score = _score_engine(path, flags)
            if score is not None:
                scored.append((score, path))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in scored]

    def _start_process(self):
        self.abs_nnue_path = None
        if self.nnue_path:
            self.abs_nnue_path = _resolve_project_path(self.nnue_path)
            if not os.path.exists(self.abs_nnue_path):
                raise FileNotFoundError(f"NNUE file not found at {self.abs_nnue_path}.")

        candidates = self._engine_candidates()
        if not candidates:
            raise FileNotFoundError(
                "No Pikafish engine found. Expected an engines/ folder "
                f"or {FALLBACK_ENGINE_PATH}."
            )

        errors = []
        for abs_path in candidates:
            if not os.path.exists(abs_path):
                errors.append(f"{abs_path}: not found")
                continue
            try:
                self._launch_process(abs_path)
                print(f"[Engine] using {abs_path}")
                return
            except Exception as e:
                errors.append(f"{abs_path}: {e}")
                try:
                    if self.process and self.process.poll() is None:
                        self.process.terminate()
                        self.process.wait(timeout=1.0)
                except Exception:
                    pass

        raise RuntimeError("Failed to start any Pikafish engine:\n" + "\n".join(errors))

    def _launch_process(self, abs_path):
        if os.name != "nt":
            try:
                mode = os.stat(abs_path).st_mode
                os.chmod(abs_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass

        kwargs = {}
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            [abs_path],
            cwd=os.path.dirname(abs_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **kwargs,
        )
        
        self.bestmove = None
        self.eval_score = ""
        self.current_depth = 0
        self.current_pv_move = None
        self.current_pv_line = []
        # multipv index (1..N) -> first move of that line, in engine coords
        self.pv_moves = {}
        self.pv_lines = {}
        self.pv_scores = {}
        self._ready_event = threading.Event()

        # Start a thread to read engine output continuously
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

        self.send_cmd("uci")
        time.sleep(0.5)
        try:
            threads = max(1, int(float(load_config().get("engine_threads", 1))))
        except Exception:
            threads = 1
        self.send_cmd(f"setoption name Threads value {threads}")
        if self.abs_nnue_path:
            self.send_cmd(f"setoption name EvalFile value {self.abs_nnue_path}")
        # Ask the engine for several lines so callers can pick an
        # alternative when a position recurs and the #1 move would loop.
        self.send_cmd("setoption name MultiPV value 5")
        self.wait_for_ready(timeout=4.0)

    def restart(self):
        print("\n[Engine] Restarting engine...")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self._start_process()

    def _reader(self):
        while True:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                # print(f"[RAW ENGINE] {line}")
                
                if line == "readyok":
                    self._ready_event.set()

                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        self.bestmove = parts[1]
                        
                if line.startswith("info"):
                    parts = line.split()
                    mpv_idx = 1
                    if "multipv" in parts:
                        try:
                            mpv_idx = int(parts[parts.index("multipv") + 1])
                        except (ValueError, IndexError):
                            mpv_idx = 1

                    if "depth" in parts:
                        try:
                            depth = int(parts[parts.index("depth") + 1])
                            if mpv_idx == 1:
                                self.current_depth = depth
                        except (ValueError, IndexError):
                            pass

                    # Extract score if present
                    if "score" in line:
                        try:
                            idx = parts.index("score")
                            if idx + 2 < len(parts):
                                score = f"{parts[idx+1]} {parts[idx+2]}"
                                self.pv_scores[mpv_idx] = score
                                if mpv_idx == 1:
                                    self.eval_score = score
                        except ValueError:
                            pass
                            
                    # Extract PV (Principal Variation) best move so far
                    if " pv " in line:
                        pv_str = line.split(" pv ")[1]
                        pv_line = pv_str.split()
                        pv_first = pv_line[0] if pv_line else None

                        if pv_first:
                            self.pv_moves[mpv_idx] = pv_first
                            self.pv_lines[mpv_idx] = pv_line[:8]
                            if mpv_idx == 1:
                                self.current_pv_move = pv_first
                                self.current_pv_line = pv_line[:8]
            except Exception as e:
                print(f"\n[Engine Reader Error] {e}")

    def send_cmd(self, cmd):
        if self.process and self.process.poll() is None:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def wait_for_ready(self, timeout=3.0, should_cancel=None):
        self._ready_event.clear()
        self.send_cmd("isready")
        deadline = time.time() + timeout
        while not self._ready_event.is_set():
            if should_cancel and should_cancel():
                raise InterruptedError("Engine wait cancelled")
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._ready_event.wait(min(0.05, remaining))
        if not self._ready_event.is_set():
            raise TimeoutError("Engine did not respond with readyok")

    def start_search(self, fen, movetime=None, depth=None, should_cancel=None):
        if self.process.poll() is not None:
            self.restart()

        if should_cancel and should_cancel():
            raise InterruptedError("Engine search cancelled")

        self.send_cmd("stop")
        self.wait_for_ready(should_cancel=should_cancel)

        if should_cancel and should_cancel():
            raise InterruptedError("Engine search cancelled")

        self.bestmove = None
        self.eval_score = ""
        self.current_depth = 0
        self.current_pv_move = None
        self.current_pv_line = []
        self.pv_moves = {}
        self.pv_lines = {}
        self.pv_scores = {}
        self.send_cmd("ucinewgame")
        self.wait_for_ready(should_cancel=should_cancel)

        if should_cancel and should_cancel():
            raise InterruptedError("Engine search cancelled")

        self.send_cmd(f"position fen {fen}")
        if movetime is not None:
            self.send_cmd(f"go movetime {movetime}")
        elif depth is not None:
            self.send_cmd(f"go depth {depth}")
        else:
            self.send_cmd("go movetime 250")
        
    def get_best_move(self, fen, depth=30):
        self.start_search(fen, depth=depth)
        while self.bestmove is None:
            time.sleep(0.1)
        return self.bestmove

    def quit(self):
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                self.send_cmd("quit")
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
