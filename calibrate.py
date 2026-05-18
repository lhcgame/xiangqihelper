import cv2
import json
import mss
import numpy as np
import os
import sys

from app_paths import candidate_paths, resolve_resource_path, resolve_user_path

TEMPLATE_DIR_NAME = "templates"
CONFIG_FILE_NAME = "config.json"
NEXT_GAME_BUTTON_TEMPLATE_NAME = "next_game_button.png"
PATCH_SIZE = 48
CALIBRATE_WINDOW_MAX_W = 2600
CALIBRATE_WINDOW_MAX_H = 1600
NEXT_GAME_TEMPLATE_HALF_W = 110
NEXT_GAME_TEMPLATE_HALF_H = 45
REQUIRED_CALIBRATION_KEYS = (
    "top_left_x",
    "top_left_y",
    "bottom_right_x",
    "bottom_right_y",
    "dx",
    "dy",
)

PIECE_NAMES = [
    "R_King", "R_Advisor", "R_Elephant", "R_Horse", "R_Chariot", "R_Cannon", "R_Pawn",
    "B_King", "B_Advisor", "B_Elephant", "B_Horse", "B_Chariot", "B_Cannon", "B_Pawn",
]


def _starting_layout():
    layout = {}
    black_back_rank = [
        "B_Chariot", "B_Horse", "B_Elephant", "B_Advisor", "B_King",
        "B_Advisor", "B_Elephant", "B_Horse", "B_Chariot",
    ]
    red_back_rank = [
        "R_Chariot", "R_Horse", "R_Elephant", "R_Advisor", "R_King",
        "R_Advisor", "R_Elephant", "R_Horse", "R_Chariot",
    ]

    for file_index, name in enumerate(black_back_rank):
        layout[(file_index, 0)] = name
    for file_index, name in enumerate(red_back_rank):
        layout[(file_index, 9)] = name

    layout[(1, 2)] = "B_Cannon"
    layout[(7, 2)] = "B_Cannon"
    layout[(1, 7)] = "R_Cannon"
    layout[(7, 7)] = "R_Cannon"

    for file_index in (0, 2, 4, 6, 8):
        layout[(file_index, 3)] = "B_Pawn"
        layout[(file_index, 6)] = "R_Pawn"

    return layout


STARTING_LAYOUT = _starting_layout()


def _config_read_path():
    return resolve_resource_path(CONFIG_FILE_NAME)


def config_paths():
    return candidate_paths(CONFIG_FILE_NAME)


def has_calibration(config):
    return all(key in config for key in REQUIRED_CALIBRATION_KEYS)


def calibration_exists():
    return has_calibration(_load_existing_config())


def load_config():
    return _load_existing_config()


def save_config(config):
    with open(_config_write_path(), "w") as f:
        json.dump(config, f, indent=4)


def update_config(values):
    config = load_config()
    config.update(values)
    save_config(config)
    return config


def _config_write_path():
    return resolve_user_path(CONFIG_FILE_NAME)


def _template_write_dir():
    return resolve_user_path(TEMPLATE_DIR_NAME)


def _next_game_template_write_path():
    return os.path.join(_template_write_dir(), NEXT_GAME_BUTTON_TEMPLATE_NAME)


def _select_monitor(monitors, config):
    default_index = 1 if len(monitors) > 1 else 0
    try:
        requested = int(config.get("screen_monitor", default_index))
    except (TypeError, ValueError):
        requested = default_index

    if 0 <= requested < len(monitors):
        return requested, monitors[requested]
    return default_index, monitors[default_index]


def capture_screen(config=None, return_monitor=False):
    if config is None:
        config = _load_existing_config()
    with mss.mss() as sct:
        monitors = sct.monitors
        monitor_index, monitor = _select_monitor(monitors, config)
        shot = sct.grab(monitor)
    image = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    if return_monitor:
        info = dict(monitor)
        info["index"] = monitor_index
        return image, info
    return image


def _fit_to_display(img, max_w=CALIBRATE_WINDOW_MAX_W, max_h=CALIBRATE_WINDOW_MAX_H):
    height, width = img.shape[:2]
    scale = min(max_w / width, max_h / height, 1.0)
    if scale == 1.0:
        return img.copy(), 1.0
    size = (int(width * scale), int(height * scale))
    return cv2.resize(img, size), scale


def _draw_grid(canvas, pt1, pt2, scale):
    x1, y1 = pt1
    x2, y2 = pt2
    for file_index in range(9):
        x = int(round((x1 + (x2 - x1) * file_index / 8.0) * scale))
        cv2.line(
            canvas,
            (x, int(round(y1 * scale))),
            (x, int(round(y2 * scale))),
            (0, 255, 0),
            1,
        )
    for rank_index in range(10):
        y = int(round((y1 + (y2 - y1) * rank_index / 9.0) * scale))
        cv2.line(
            canvas,
            (int(round(x1 * scale)), y),
            (int(round(x2 * scale)), y),
            (0, 255, 0),
            1,
        )


def _draw_message(canvas, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    margin = 14
    line_height = 32
    box_height = margin * 2 + line_height * len(lines)

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (canvas.shape[1], box_height),
        (255, 255, 255),
        -1,
    )
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    for index, line in enumerate(lines):
        y = margin + 24 + index * line_height
        cv2.putText(
            canvas,
            line,
            (margin, y),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )


def pick_corners(full_img):
    display, scale = _fit_to_display(full_img)
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / scale, y / scale))
            print(f"  point {len(points)}: ({points[-1][0]:.1f}, {points[-1][1]:.1f})")

    window = "Calibrate"
    window_resized = False
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])
    cv2.moveWindow(window, 20, 20)
    cv2.setMouseCallback(window, on_mouse)

    print("Click the top-left grid intersection, then the bottom-right grid intersection.")
    print("Press ENTER to save, C to clear, or ESC to cancel.")

    while True:
        canvas = display.copy()
        if len(points) == 2:
            _draw_grid(canvas, points[0], points[1], scale)

        for index, (x, y) in enumerate(points, start=1):
            pos = (int(round(x * scale)), int(round(y * scale)))
            cv2.circle(canvas, pos, 6, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(index),
                (pos[0] + 8, pos[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        _draw_message(canvas, [
            "Click the TOP-LEFT grid intersection, then the BOTTOM-RIGHT grid intersection.",
            "Press ENTER to save. Press C to clear. Press ESC to cancel.",
        ])
        cv2.imshow(window, canvas)
        if not window_resized:
            cv2.resizeWindow(window, display.shape[1], display.shape[0])
            window_resized = True

        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) == 2:
            break
        if key == ord("c"):
            points.clear()
        if key == 27:
            cv2.destroyWindow(window)
            sys.exit("Calibration cancelled.")

    cv2.destroyWindow(window)
    return points[0], points[1]


def pick_point(full_img, window, lines):
    display, scale = _fit_to_display(full_img)
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points[:] = [(x / scale, y / scale)]
            print(f"  point: ({points[0][0]:.1f}, {points[0][1]:.1f})")

    window_resized = False
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])
    cv2.moveWindow(window, 20, 20)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = display.copy()
        if points:
            x, y = points[0]
            pos = (int(round(x * scale)), int(round(y * scale)))
            cv2.circle(canvas, pos, 8, (0, 0, 255), -1)
            cv2.line(canvas, (pos[0] - 18, pos[1]), (pos[0] + 18, pos[1]), (0, 0, 255), 2)
            cv2.line(canvas, (pos[0], pos[1] - 18), (pos[0], pos[1] + 18), (0, 0, 255), 2)

        _draw_message(canvas, lines)
        cv2.imshow(window, canvas)
        if not window_resized:
            cv2.resizeWindow(window, display.shape[1], display.shape[0])
            window_resized = True

        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and points:
            break
        if key == ord("c"):
            points.clear()
        if key == 27:
            cv2.destroyWindow(window)
            sys.exit("Recognition cancelled.")

    cv2.destroyWindow(window)
    return points[0]


def pick_rectangle(full_img, window, lines):
    display, scale = _fit_to_display(full_img)
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / scale, y / scale))
            print(f"  point {len(points)}: ({points[-1][0]:.1f}, {points[-1][1]:.1f})")

    window_resized = False
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])
    cv2.moveWindow(window, 20, 20)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = display.copy()
        for index, (x, y) in enumerate(points, start=1):
            pos = (int(round(x * scale)), int(round(y * scale)))
            cv2.circle(canvas, pos, 7, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(index),
                (pos[0] + 8, pos[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) == 2:
            x1 = int(round(min(points[0][0], points[1][0]) * scale))
            y1 = int(round(min(points[0][1], points[1][1]) * scale))
            x2 = int(round(max(points[0][0], points[1][0]) * scale))
            y2 = int(round(max(points[0][1], points[1][1]) * scale))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _draw_message(canvas, lines)
        cv2.imshow(window, canvas)
        if not window_resized:
            cv2.resizeWindow(window, display.shape[1], display.shape[0])
            window_resized = True

        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) == 2:
            break
        if key == ord("c"):
            points.clear()
        if key == 27:
            cv2.destroyWindow(window)
            sys.exit("Recognition cancelled.")

    cv2.destroyWindow(window)
    x1 = min(points[0][0], points[1][0])
    y1 = min(points[0][1], points[1][1])
    x2 = max(points[0][0], points[1][0])
    y2 = max(points[0][1], points[1][1])
    if x2 - x1 < 10 or y2 - y1 < 10:
        sys.exit("Selected area is too small.")
    return x1, y1, x2, y2


def _crop_button_template(full_img, point):
    x, y = point
    height, width = full_img.shape[:2]
    x1 = max(0, int(round(x - NEXT_GAME_TEMPLATE_HALF_W)))
    y1 = max(0, int(round(y - NEXT_GAME_TEMPLATE_HALF_H)))
    x2 = min(width, int(round(x + NEXT_GAME_TEMPLATE_HALF_W)))
    y2 = min(height, int(round(y + NEXT_GAME_TEMPLATE_HALF_H)))
    crop = full_img[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        sys.exit("Button template is too small. Click the center of the Next Game button.")
    return crop


def save_next_game_button(full_img, point, monitor_info=None):
    os.makedirs(_template_write_dir(), exist_ok=True)
    template = _crop_button_template(full_img, point)
    path = _next_game_template_write_path()
    cv2.imwrite(path, template)

    screen_left = int(monitor_info.get("left", 0)) if monitor_info else 0
    screen_top = int(monitor_info.get("top", 0)) if monitor_info else 0
    config = _load_existing_config()
    config.update({
        "next_game_button_x": float(point[0] + screen_left),
        "next_game_button_y": float(point[1] + screen_top),
        "next_game_screen_left": screen_left,
        "next_game_screen_top": screen_top,
        "next_game_template_w": int(template.shape[1]),
        "next_game_template_h": int(template.shape[0]),
    })
    if monitor_info:
        config["screen_monitor"] = int(monitor_info.get("index", config.get("screen_monitor", 1)))
    save_config(config)
    print(f"Saved {CONFIG_FILE_NAME} and templates/{NEXT_GAME_BUTTON_TEMPLATE_NAME}.")


def run_next_game_recognition(prompt=True):
    print("Next game button recognition")
    print("Open the game-over screen so the Next Game button is visible.")
    if prompt:
        input("Press ENTER to capture the screen...")

    full_img, monitor_info = capture_screen(return_monitor=True)
    point = pick_point(full_img, "Recognize Next Game", [
        "Click the CENTER of the Next Game button.",
        "Press ENTER to save. Press C to clear. Press ESC to cancel.",
    ])
    save_next_game_button(full_img, point, monitor_info)
    print("Done.")


def _circle_mask(size=PATCH_SIZE):
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.94 / 2), (1, 1, 1), -1)
    return mask


def _extract_patch(img, cx, cy, half_w, half_h):
    height, width = img.shape[:2]
    x1 = max(0, int(round(cx - half_w)))
    y1 = max(0, int(round(cy - half_h)))
    x2 = min(width, int(round(cx + half_w)))
    y2 = min(height, int(round(cy + half_h)))
    patch = img[y1:y2, x1:x2]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return None
    return cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))


def build_templates(img, top_left, dx, dy):
    mask = _circle_mask()
    half_w = dx * 0.45
    half_h = dy * 0.45
    patches_by_name = {name: [] for name in PIECE_NAMES}

    for (file_index, rank_index), name in STARTING_LAYOUT.items():
        cx = top_left[0] + file_index * dx
        cy = top_left[1] + rank_index * dy
        patch = _extract_patch(img, cx, cy, half_w, half_h)
        if patch is not None:
            patches_by_name[name].append(patch * mask)

    templates = {}
    for name, patches in patches_by_name.items():
        if patches:
            templates[name] = np.median(np.stack(patches), axis=0).astype(np.uint8)
            print(f"  {name}: {len(patches)} sample(s)")
        else:
            print(f"  missing: {name}")
    return templates


def _load_existing_config():
    path = _config_read_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(full_img, top_left, bottom_right, templates, monitor_info=None):
    x1, y1 = top_left
    x2, y2 = bottom_right
    dx = (x2 - x1) / 8.0
    dy = (y2 - y1) / 9.0
    if dx <= 4 or dy <= 4:
        sys.exit("Corners look wrong. Click top-left first, then bottom-right.")

    template_dir = _template_write_dir()
    os.makedirs(template_dir, exist_ok=True)

    pad_x = int(dx * 1.5)
    pad_y = int(dy * 1.5)
    height, width = full_img.shape[:2]
    ref_x1 = max(0, int(x1 - pad_x))
    ref_y1 = max(0, int(y1 - pad_y))
    ref_x2 = min(width, int(x2 + pad_x))
    ref_y2 = min(height, int(y2 + pad_y))
    ref_board = full_img[ref_y1:ref_y2, ref_x1:ref_x2]

    config = _load_existing_config()
    config.update({
        "top_left_x": float(x1),
        "top_left_y": float(y1),
        "bottom_right_x": float(x2),
        "bottom_right_y": float(y2),
        "dx": float(dx),
        "dy": float(dy),
        "ref_x1": int(ref_x1),
        "ref_y1": int(ref_y1),
        "ref_w": int(ref_board.shape[1]),
        "ref_h": int(ref_board.shape[0]),
    })
    if monitor_info:
        config.update({
            "screen_monitor": int(monitor_info.get("index", config.get("screen_monitor", 1))),
            "screen_left": int(monitor_info.get("left", 0)),
            "screen_top": int(monitor_info.get("top", 0)),
        })

    save_config(config)

    cv2.imwrite(os.path.join(template_dir, "ref_board.png"), ref_board)
    for name in PIECE_NAMES:
        path = os.path.join(template_dir, f"{name}.png")
        if os.path.exists(path):
            os.remove(path)
        if name in templates:
            cv2.imwrite(path, templates[name])

    print(f"Saved {CONFIG_FILE_NAME}, templates/ref_board.png, and {len(templates)} templates.")


def run_calibration(prompt=True):
    print("Pikafish helper calibration")
    print("Put the board in the standard starting position.")
    if prompt:
        input("Press ENTER to capture the screen...")

    full_img, monitor_info = capture_screen(return_monitor=True)
    top_left, bottom_right = pick_corners(full_img)
    dx = (bottom_right[0] - top_left[0]) / 8.0
    dy = (bottom_right[1] - top_left[1]) / 9.0

    print(f"Grid spacing: dx={dx:.2f}, dy={dy:.2f}")
    print("Building templates...")
    templates = build_templates(full_img, top_left, dx, dy)
    if len(templates) != len(PIECE_NAMES):
        print("Warning: not every piece type was captured. Make sure the board is at start.")

    save_calibration(full_img, top_left, bottom_right, templates, monitor_info)
    print("Done. Run `python main.py` to start the helper.")


def main():
    run_calibration(prompt="--no-prompt" not in sys.argv)


if __name__ == "__main__":
    main()
