import cv2
import numpy as np
import os
import json

from app_paths import candidate_paths, resolve_resource_path

TEMPLATE_DIR_NAME = 'templates'
CONFIG_FILE_NAME = 'config.json'
REQUIRED_CALIBRATION_KEYS = (
    'top_left_x',
    'top_left_y',
    'bottom_right_x',
    'bottom_right_y',
    'dx',
    'dy',
)

PIECE_NAMES = [
    'R_King', 'R_Advisor', 'R_Elephant', 'R_Horse', 'R_Chariot', 'R_Cannon', 'R_Pawn',
    'B_King', 'B_Advisor', 'B_Elephant', 'B_Horse', 'B_Chariot', 'B_Cannon', 'B_Pawn',
]

DETECTION_CLASSES = ['Empty'] + PIECE_NAMES

FEN_CHAR = {
    'R_King': 'K', 'R_Advisor': 'A', 'R_Elephant': 'B', 'R_Horse': 'N',
    'R_Chariot': 'R', 'R_Cannon': 'C', 'R_Pawn': 'P',
    'B_King': 'k', 'B_Advisor': 'a', 'B_Elephant': 'b', 'B_Horse': 'n',
    'B_Chariot': 'r', 'B_Cannon': 'c', 'B_Pawn': 'p',
}

MAX_PIECES = {
    'R_King': 1, 'B_King': 1,
    'R_Advisor': 2, 'B_Advisor': 2,
    'R_Elephant': 2, 'B_Elephant': 2,
    'R_Horse': 2, 'B_Horse': 2,
    'R_Chariot': 2, 'B_Chariot': 2,
    'R_Cannon': 2, 'B_Cannon': 2,
    'R_Pawn': 5, 'B_Pawn': 5,
}


def _make_circle_mask(size):
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.94 / 2), (1, 1, 1), -1)
    return mask


class Vision:
    def __init__(self, threshold=0.90,
                 confidence_threshold=0.5,
                 homography_refresh_every=100,
                 template_margin_min=0.005,
                 template_search_radius=2):
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.homography_refresh_every = homography_refresh_every
        self.template_margin_min = template_margin_min
        self.template_search_radius = template_search_radius

        self.templates = {}
        self.config = {}
        self.crop_x = 0
        self.crop_y = 0

        self._load_config()
        self._load_templates()

        self._cell_size = 32
        self._mask32 = _make_circle_mask(self._cell_size)
        self._masked_templates = self._precompute_masked_templates()

        print("[Vision] using masked-template matching.")

        self.ref_board = cv2.imread(os.path.join(self.template_dir, "ref_board.png"))
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            self.sift = None

        if self.ref_board is not None and self.sift is not None:
            self.ref_gray = cv2.cvtColor(self.ref_board, cv2.COLOR_BGR2GRAY)
            self.kp1, self.des1 = self.sift.detectAndCompute(self.ref_gray, None)
            FLANN_INDEX_KDTREE = 1
            self.flann = cv2.FlannBasedMatcher(
                dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
                dict(checks=50),
            )
        else:
            self.kp1 = self.des1 = None
            self.flann = None
            print("[Vision] dynamic homography disabled. Re-run calibrate.py if tracking is poor.")

        self.H_inv_cache = None
        self._homography_age = self.homography_refresh_every  # force refresh on first call
        self.board_visible = False

        self.last_grid = None
        self.last_confidence = [[0.0] * 9 for _ in range(10)]

    def _load_config(self):
        self.config_path = resolve_resource_path(CONFIG_FILE_NAME)
        if not os.path.exists(self.config_path):
            checked = ", ".join(candidate_paths(CONFIG_FILE_NAME))
            raise FileNotFoundError(
                f"{CONFIG_FILE_NAME} not found. Click Calibrate first, "
                f"or run `python main.py --calibrate`. Checked: {checked}"
            )
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        missing = [key for key in REQUIRED_CALIBRATION_KEYS if key not in self.config]
        if missing:
            raise ValueError(
                f"{CONFIG_FILE_NAME} is missing calibration values: {', '.join(missing)}. "
                "Click Calibrate first."
            )

    def _load_templates(self):
        self.template_dir = resolve_resource_path(TEMPLATE_DIR_NAME)
        for name in PIECE_NAMES:
            path = os.path.join(self.template_dir, f"{name}.png")
            if os.path.exists(path):
                self.templates[name] = cv2.imread(path)

    def _precompute_masked_templates(self):
        out = {}
        for name, tmpl in self.templates.items():
            t = cv2.resize(tmpl, (self._cell_size, self._cell_size))
            masked = (t * self._mask32).astype(np.uint8)
            out[name] = (
                masked,
                cv2.rotate(masked, cv2.ROTATE_180),
            )
        return out

    def _match_template_score(self, patch, template):
        radius = int(self.template_search_radius)
        if radius <= 0:
            return float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])

        search = cv2.copyMakeBorder(
            patch,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_REPLICATE,
        )
        return float(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).max())

    def invalidate_homography(self):
        self._homography_age = self.homography_refresh_every

    def _old_detect_board(self, screen_img):
        x1 = self.config['top_left_x']
        y1 = self.config['top_left_y']
        x2 = self.config['bottom_right_x']
        y2 = self.config['bottom_right_y']
        padding = 60
        sh, sw = screen_img.shape[:2]
        self.crop_x = max(0, int(x1 - padding))
        self.crop_y = max(0, int(y1 - padding))
        crop_xw = min(sw, int(x2 + padding))
        crop_yh = min(sh, int(y2 + padding))
        self.board_visible = True
        return screen_img[self.crop_y:crop_yh, self.crop_x:crop_xw]

    def detect_board(self, screen_img):
        if self.ref_board is None or self.sift is None or self.flann is None:
            return self._old_detect_board(screen_img)

        if self._homography_age >= self.homography_refresh_every:
            gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
            kp2, des2 = self.sift.detectAndCompute(gray, None)
            if self.des1 is not None and des2 is not None and len(des2) > 0:
                matches = self.flann.knnMatch(self.des1, des2, k=2)
                good = [m for m_n in matches if len(m_n) == 2
                        for m, n in [m_n] if m.distance < 0.7 * n.distance]
                if len(good) > 15:
                    src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        try:
                            self.H_inv_cache = np.linalg.inv(H)
                            self._homography_age = 0
                        except np.linalg.LinAlgError:
                            pass
        else:
            self._homography_age += 1

        if self.H_inv_cache is not None:
            ref_w = self.config.get('ref_w', self.ref_board.shape[1])
            ref_h = self.config.get('ref_h', self.ref_board.shape[0])
            warped = cv2.warpPerspective(screen_img, self.H_inv_cache, (ref_w, ref_h))
            self.crop_x = self.config.get('ref_x1', 0)
            self.crop_y = self.config.get('ref_y1', 0)
            self.board_visible = True
            return warped

        self.board_visible = False
        return self._old_detect_board(screen_img)

    def _extract_cell_patches(self, board_img):
        """Return list of 90 patches (10*9) at 32x32, masked. Order: r=0..9, f=0..8."""
        patches = np.zeros((90, self._cell_size, self._cell_size, 3), dtype=np.uint8)
        valid = np.zeros(90, dtype=bool)
        idx = 0
        half_w = float(self.config['dx']) * 0.45
        half_h = float(self.config['dy']) * 0.45
        for r in range(10):
            for f in range(9):
                abs_x = self.config['top_left_x'] + f * self.config['dx']
                abs_y = self.config['top_left_y'] + r * self.config['dy']
                rel_x = float(abs_x - self.crop_x)
                rel_y = float(abs_y - self.crop_y)
                y1 = max(0, int(round(rel_y - half_h)))
                y2 = min(board_img.shape[0], int(round(rel_y + half_h)))
                x1 = max(0, int(round(rel_x - half_w)))
                x2 = min(board_img.shape[1], int(round(rel_x + half_w)))
                p = board_img[y1:y2, x1:x2]
                if p.shape[0] >= 30 and p.shape[1] >= 30:
                    p = cv2.resize(p, (self._cell_size, self._cell_size))
                    patches[idx] = (p * self._mask32).astype(np.uint8)
                    valid[idx] = True
                idx += 1
        return patches, valid

    def _classify_templates(self, patches, valid):
        cls = np.zeros(90, dtype=np.int64)  # 0 = Empty
        conf = np.zeros(90, dtype=np.float32)
        if not self._masked_templates:
            conf[:] = 1.0
            return cls, conf

        names = list(self._masked_templates.keys())
        scores = np.full((90, len(names)), -1.0, dtype=np.float32)
        for i, p in enumerate(patches):
            if not valid[i]:
                conf[i] = 1.0
                continue
            for j, n in enumerate(names):
                scores[i, j] = max(
                    self._match_template_score(p, t)
                    for t in self._masked_templates[n]
                )

        order = np.argsort(-scores, axis=1)
        top1 = scores[np.arange(90), order[:, 0]]
        top2 = scores[np.arange(90), order[:, 1]] if scores.shape[1] > 1 else np.zeros(90)
        margin = top1 - top2

        for i in range(90):
            if not valid[i]:
                continue
            if top1[i] >= self.threshold and margin[i] >= self.template_margin_min:
                piece = names[order[i, 0]]
                cls[i] = DETECTION_CLASSES.index(piece)
                conf[i] = float(min(1.0, top1[i]))
            else:
                cls[i] = 0  # Empty
                conf[i] = 1.0 if top1[i] < self.threshold else 0.0
        return cls, conf

    def _classify_patches(self, patches, valid):
        return self._classify_templates(patches, valid)

    def _enforce_max_pieces(self, cls, conf):
        """Drop lowest-confidence overflow when more pieces of one type exist than allowed."""
        per_type = {}
        for i, c in enumerate(cls):
            if c == 0:
                continue
            name = DETECTION_CLASSES[c]
            per_type.setdefault(name, []).append((conf[i], i))
        for name, lst in per_type.items():
            cap = MAX_PIECES.get(name, 99)
            if len(lst) > cap:
                lst.sort(key=lambda x: -x[0])
                for _, idx in lst[cap:]:
                    cls[idx] = 0
                    conf[idx] = max(conf[idx], 0.0)

    def screen_pos_to_cell(self, sx, sy):
        """Map a screen pixel (sx, sy) to (row, file) board cell, or None."""
        local_x = float(sx) - float(self.config.get('screen_left', 0))
        local_y = float(sy) - float(self.config.get('screen_top', 0))

        if self.H_inv_cache is not None:
            pt = np.array([local_x, local_y, 1.0], dtype=np.float64)
            ref = self.H_inv_cache @ pt
            ref /= ref[2]
            rx = float(ref[0]) + float(self.config.get('ref_x1', 0))
            ry = float(ref[1]) + float(self.config.get('ref_y1', 0))
        else:
            rx, ry = local_x, local_y

        dx = float(self.config['dx'])
        dy = float(self.config['dy'])
        x0 = float(self.config['top_left_x'])
        y0 = float(self.config['top_left_y'])
        f = round((rx - x0) / dx)
        r = round((ry - y0) / dy)
        if 0 <= r <= 9 and 0 <= f <= 8:
            cx = x0 + f * dx
            cy = y0 + r * dy
            if abs(rx - cx) < dx * 0.5 and abs(ry - cy) < dy * 0.5:
                return r, f
        return None

    def board_cell_screen_pos(self, row, col, x_offset=0.0, y_offset=0.0):
        local_x = float(self.config["top_left_x"]) + col * float(self.config["dx"])
        local_y = float(self.config["top_left_y"]) + row * float(self.config["dy"])

        if self.H_inv_cache is not None:
            try:
                h = np.linalg.inv(self.H_inv_cache)
                ref_x = local_x - float(self.config.get('ref_x1', 0))
                ref_y = local_y - float(self.config.get('ref_y1', 0))
                pt = h @ np.array([ref_x, ref_y, 1.0], dtype=np.float64)
                pt /= pt[2]
                local_x, local_y = float(pt[0]), float(pt[1])
            except np.linalg.LinAlgError:
                pass

        screen_x = local_x + float(self.config.get('screen_left', 0)) + float(x_offset)
        screen_y = local_y + float(self.config.get('screen_top', 0)) + float(y_offset)
        return screen_x, screen_y

    def get_board_state(self, board_img, sticky=True, mouse_screen_pos=None):
        if board_img.size == 0:
            return [['' for _ in range(9)] for _ in range(10)]

        patches, valid = self._extract_cell_patches(board_img)

        cls, conf = self._classify_patches(patches, valid)

        self._enforce_max_pieces(cls, conf)

        grid = [['' for _ in range(9)] for _ in range(10)]
        confidence = [[0.0] * 9 for _ in range(10)]
        for r in range(10):
            for f in range(9):
                idx = r * 9 + f
                c = int(cls[idx])
                confidence[r][f] = float(conf[idx])
                if c == 0:
                    continue
                grid[r][f] = FEN_CHAR[DETECTION_CLASSES[c]]

        # Determine which cell the mouse is over and always keep last value there
        mouse_cell = None
        if mouse_screen_pos is not None:
            mouse_cell = self.screen_pos_to_cell(*mouse_screen_pos)

        if sticky and self.last_grid is not None:
            for r in range(10):
                for f in range(9):
                    if confidence[r][f] < self.confidence_threshold or (mouse_cell == (r, f)):
                        grid[r][f] = self.last_grid[r][f]

        self.last_grid = [row[:] for row in grid]
        self.last_confidence = confidence
        return grid

    def to_fen(self, grid, turn='w'):
        rows = []
        for row in grid:
            empty = 0
            s = ''
            for cell in row:
                if cell == '':
                    empty += 1
                else:
                    if empty:
                        s += str(empty); empty = 0
                    s += cell
            if empty:
                s += str(empty)
            rows.append(s)
        return "/".join(rows) + f" {turn} - - 0 1"

    def draw_move(self, board_img, best_move):
        if not best_move or len(best_move) < 4:
            return board_img
        sf = ord(best_move[0]) - ord('a')
        sr = 9 - int(best_move[1])
        ef = ord(best_move[2]) - ord('a')
        er = 9 - int(best_move[3])
        sx = int(self.config['top_left_x'] + sf * self.config['dx'] - self.crop_x)
        sy = int(self.config['top_left_y'] + sr * self.config['dy'] - self.crop_y)
        ex = int(self.config['top_left_x'] + ef * self.config['dx'] - self.crop_x)
        ey = int(self.config['top_left_y'] + er * self.config['dy'] - self.crop_y)
        out = board_img.copy()
        cv2.circle(out, (sx, sy), 30, (0, 255, 255), 4)
        cv2.arrowedLine(out, (sx, sy), (ex, ey), (0, 0, 255), 6, tipLength=0.1)
        return out
