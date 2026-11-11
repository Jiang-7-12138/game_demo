"""Camera-space block game: the real video is the level; colour masks are physics."""
from __future__ import annotations

import json
import math
import random
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
MODEL_PATH = ROOT / "models" / "gesture_recognizer.task"
CAM_W, CAM_H = 640, 480
VIEW_W, VIEW_H = 900, 675
PIXEL_W, PIXEL_H = 300, 225
VIEW_SCALE = VIEW_W / CAM_W
CELL = 4                         # Low-resolution collision map, in camera pixels.
BALL_R = 10
MATERIALS = ("green", "blue", "red", "yellow")
DEBUG_BGR = {"green": (145, 209, 162), "blue": (231, 171, 94),
             "red": (143, 143, 233), "yellow": (92, 218, 245)}
DOOR_HALF_W, DOOR_HEIGHT = 14, 52
CREAM = "#FAF8F2"
APRICOT = "#F4E9D6"
SAGE = "#91A98F"
PALE_SAGE = "#E7EDE4"
CHARCOAL = "#414743"
HEALTH_RED = "#E93743"
DOOR_VIEW_W = round(DOOR_HALF_W * 2 * VIEW_SCALE)
DOOR_VIEW_H = round(DOOR_HEIGHT * VIEW_SCALE)
CHINESE_FONT = Path("C:/Windows/Fonts/simhei.ttf")
ENGLISH_FONT = Path("C:/Windows/Fonts/consolab.ttf")
START_GLYPHS = {
    "S": ("01111", "11000", "11000", "01110", "00011", "00011", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "A": ("01110", "11011", "11011", "11111", "11011", "11011", "11011"),
    "R": ("11110", "11011", "11011", "11110", "11100", "11010", "11011"),
}
PIXEL_GLYPHS = START_GLYPHS | {
    "0": ("01110", "11011", "11011", "11011", "11011", "11011", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "11011", "00011", "00110", "01100", "11000", "11111"),
    "3": ("11110", "00011", "00011", "01110", "00011", "00011", "11110"),
    "4": ("11011", "11011", "11011", "11111", "00011", "00011", "00011"),
    "5": ("11111", "11000", "11000", "11110", "00011", "00011", "11110"),
    "6": ("01111", "11000", "11000", "11110", "11011", "11011", "01110"),
    "7": ("11111", "00011", "00110", "00100", "01100", "01100", "01100"),
    "8": ("01110", "11011", "11011", "01110", "11011", "11011", "01110"),
    "9": ("01110", "11011", "11011", "01111", "00011", "00011", "11110"),
    "C": ("01111", "11000", "11000", "11000", "11000", "11000", "01111"),
    "L": ("11000", "11000", "11000", "11000", "11000", "11000", "11111"),
    "E": ("11111", "11000", "11000", "11110", "11000", "11000", "11111"),
}


def draw_pixel_text(draw, value, x, y, pixel, color):
    for index, letter in enumerate(str(value)):
        for row, pattern in enumerate(PIXEL_GLYPHS[letter]):
            for col, bit in enumerate(pattern):
                if bit == "1":
                    left, top = x + (index * 6 + col) * pixel, y + row * pixel
                    draw.rectangle((left, top, left + pixel - 1, top + pixel - 1), fill=color)


def pixel_text_mask(value, size, scale):
    """Rasterize small bold glyphs, then enlarge without smoothing."""
    font_path = CHINESE_FONT if any(ord(char) > 127 for char in value) else ENGLISH_FONT
    if not font_path.is_file():
        font_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    font = ImageFont.truetype(str(font_path), size)
    box = font.getbbox(value)
    mask = Image.new("L", (max(1, box[2] - box[0]), max(1, box[3] - box[1])))
    ImageDraw.Draw(mask).text((-box[0], -box[1]), value, font=font, fill=255)
    mask = mask.point(lambda alpha: 255 if alpha >= 128 else 0)
    return mask.resize((mask.width * scale, mask.height * scale), Image.Resampling.NEAREST)


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def normalized_frame(frame):
    if frame.shape[:2] == (CAM_H, CAM_W):
        return frame
    return cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)


def make_ball_sprite(glow=1.0):
    """A white square and soft light bloom, with no stroked edge."""
    size = 96
    center = size // 2
    half = round(BALL_R * VIEW_SCALE)
    box = (center - half, center - half, center + half, center + half)
    light = Image.new("L", (size, size))
    ImageDraw.Draw(light).rectangle(box, fill=230)
    wide = Image.new("RGBA", (size, size), "white")
    wide.putalpha(light.filter(ImageFilter.GaussianBlur(12)).point(
        lambda alpha: min(255, round(alpha * .60 * glow))))
    close = Image.new("RGBA", (size, size), "white")
    close.putalpha(light.filter(ImageFilter.GaussianBlur(5)).point(
        lambda alpha: min(255, round(alpha * .80 * glow))))
    sprite = Image.alpha_composite(wide, close)
    ImageDraw.Draw(sprite).rectangle(box, fill=(255, 255, 255, 255))
    return sprite


def make_door_sprite(glow=1.0):
    """Solid white door, soft edge light, and one short fading threshold."""
    x0, y0 = 30, 18
    size = (DOOR_VIEW_W + 60, DOOR_VIEW_H + 50)
    box = (x0, y0, x0 + DOOR_VIEW_W - 1, y0 + DOOR_VIEW_H - 1)
    light = Image.new("L", size)
    ImageDraw.Draw(light).rectangle(box, fill=255)
    halo = Image.new("RGBA", size, "white")
    halo.putalpha(light.filter(ImageFilter.GaussianBlur(8)).point(
        lambda alpha: min(255, round(alpha * .34 * glow))))
    line = Image.new("RGBA", size)
    line_draw = ImageDraw.Draw(line)
    line_width = round(DOOR_VIEW_W * 1.3)
    center = x0 + (DOOR_VIEW_W - 1) / 2
    line_y = y0 + DOOR_VIEW_H + 3
    for x in range(round(center - line_width / 2), round(center + line_width / 2)):
        fade = max(0.0, 1.0 - abs(x - center) / (line_width / 2))
        alpha = round(200 * fade ** 1.4)
        line_draw.line((x, line_y, x, line_y + 1), fill=(255, 255, 255, alpha))
    line = line.filter(ImageFilter.GaussianBlur(.7))
    sprite = Image.alpha_composite(halo, line)
    ImageDraw.Draw(sprite).rectangle(box, fill=(255, 255, 255, 255))
    return sprite


class CollisionMap:
    """Private material map. It shares exact coordinates with the camera image."""

    def __init__(self, config):
        self.config = config
        self.size = (CAM_W // CELL, CAM_H // CELL)
        self.confidence = {}
        self.bits = {name: np.zeros(self.size[::-1], np.bool_) for name in MATERIALS}
        self.ground_y = CAM_H - 32
        self.initialized = False

    def update(self, frame):
        # Classify at half camera resolution, then average into the 4-pixel
        # physics grid. The visible video still uses the original frame.
        detect_w, detect_h = CAM_W // 2, CAM_H // 2
        reduced = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
        rx, ry, rw, rh = self.config["roi"]
        x0, y0 = int(rx * detect_w), int(ry * detect_h)
        x1, y1 = min(detect_w, int((rx + rw) * detect_w)), min(detect_h, int((ry + rh) * detect_h))
        x0, y0 = max(0, x0), max(0, y0)
        if x1 <= x0 or y1 <= y0:
            return
        roi_hsv = hsv[y0:y1, x0:x1]
        kernel = np.ones((3, 3), np.uint8)
        for name in MATERIALS:
            low, high = self.config["hsv"][name]
            if low[0] <= high[0]:
                cropped = cv2.inRange(roi_hsv, np.array(low, np.uint8), np.array(high, np.uint8))
            else:
                a = cv2.inRange(roi_hsv, np.array([0, low[1], low[2]], np.uint8), np.array(high, np.uint8))
                b = cv2.inRange(roi_hsv, np.array(low, np.uint8), np.array([179, high[1], high[2]], np.uint8))
                cropped = cv2.bitwise_or(a, b)
            cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(cropped, 8)
            keep = np.zeros(count, np.uint8)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(self.config.get("min_area", 180)) // 4)
            cropped = keep[labels] * 255
            full = np.zeros((detect_h, detect_w), np.uint8)
            full[y0:y1, x0:x1] = cropped
            # Area averaging makes each physics pixel represent a 4x4 camera
            # patch. Hysteresis prevents one-frame colour fluctuations from
            # changing a platform edge, while genuine changes appear quickly.
            observation = cv2.resize(full, self.size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if not self.initialized:
                confidence = observation
                active = confidence >= .46
            else:
                confidence = .55 * self.confidence[name] + .45 * observation
                active = (confidence >= .58) | (self.bits[name] & (confidence >= .32))
            # A narrow colour sliver at the edge of the backdrop must not
            # become a river or a coin in the physics map.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), 8)
            keep = np.zeros(count, np.bool_)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 32
            active = keep[labels]
            self.confidence[name] = confidence
            self.bits[name] = active
        had_previous_frame = self.initialized
        self.initialized = True
        # Yellow coins and warm desk reflections cannot determine the floor.
        # The bottom edge of green, blue and red blocks is the contact line
        # with the white turntable in this camera view.
        terrain = np.logical_or.reduce([self.bits[name] for name in ("green", "blue", "red")])
        occupied_rows = np.flatnonzero(terrain.any(axis=1))
        if len(occupied_rows):
            observed_ground = min(CAM_H - 8, (int(occupied_rows[-1]) + 1) * CELL)
            if not had_previous_frame or abs(observed_ground - self.ground_y) > 32:
                self.ground_y = observed_ground
            else:
                self.ground_y = .75 * self.ground_y + .25 * observed_ground

    def at(self, name, x, y):
        col, row = int(x // CELL), int(y // CELL)
        return 0 <= row < self.bits[name].shape[0] and 0 <= col < self.bits[name].shape[1] and bool(self.bits[name][row, col])

    def touches(self, name, x, y, radius=BALL_R):
        mask = self.bits[name]
        x0, x1 = max(0, int((x-radius) // CELL)), min(mask.shape[1], int((x+radius) // CELL) + 1)
        y0, y1 = max(0, int((y-radius) // CELL)), min(mask.shape[0], int((y+radius) // CELL) + 1)
        if x0 >= x1 or y0 >= y1:
            return False
        return bool(np.any(mask[y0:y1, x0:x1]))

    def green_surface(self, x, old_bottom, new_bottom):
        mask = self.bits["green"]
        xa = max(0, int((x - BALL_R * .65) // CELL))
        xb = min(mask.shape[1], int((x + BALL_R * .65) // CELL) + 1)
        ya = max(0, int((old_bottom - 7) // CELL))
        yb = min(mask.shape[0], int((new_bottom + 4) // CELL) + 1)
        for row in range(ya, yb):
            if np.count_nonzero(mask[row, xa:xb]) >= 2:
                top = row * CELL
                if old_bottom <= top + 8 and new_bottom >= top - 4:
                    return top
        return None

    def starting_point(self):
        mask = self.bits["green"].astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        pieces = [stats[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 8]
        if pieces:
            left = min(pieces, key=lambda s: s[cv2.CC_STAT_LEFT])
            return ((left[cv2.CC_STAT_LEFT] + left[cv2.CC_STAT_WIDTH] / 2) * CELL,
                    left[cv2.CC_STAT_TOP] * CELL - BALL_R)
        return CAM_W * .3, self.ground_y - BALL_R

    def coins(self):
        mask = self.bits["yellow"].astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        return [((s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] / 2) * CELL,
                 (s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] / 2) * CELL,
                 s[cv2.CC_STAT_WIDTH] * CELL, s[cv2.CC_STAT_HEIGHT] * CELL)
                for s in stats[1:count] if s[cv2.CC_STAT_AREA] >= 6]

    def tinted_debug_frame(self, frame):
        image = frame.copy()
        for name in MATERIALS:
            mask = cv2.resize(self.bits[name].astype(np.uint8), (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST).astype(bool)
            image[mask] = (image[mask] * .65 + np.asarray(DEBUG_BGR[name]) * .35).astype(np.uint8)
        return image


class HandController:
    """Optional MediaPipe landmarks; the model is used only for hand input."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
        self.timestamp = -1
        self.pinched = False

    def read(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame[:int(CAM_H * .55)], cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.timestamp = max(self.timestamp + 1, int(time.monotonic() * 1000))
        result = self.recognizer.recognize_for_video(image, self.timestamp)
        if not result.hand_landmarks:
            self.pinched = False
            return 0, False
        points = result.hand_landmarks[0]
        palm_x = sum(points[i].x for i in (0, 5, 9, 17)) / 4
        direction = -1 if palm_x < .37 else (1 if palm_x > .63 else 0)
        scale = max(.04, math.dist((points[0].x, points[0].y), (points[9].x, points[9].y)))
        pinch = math.dist((points[4].x, points[4].y), (points[8].x, points[8].y)) / scale < .42
        jump = pinch and not self.pinched
        self.pinched = pinch
        return direction, jump

    def close(self):
        self.recognizer.close()


class Game:
    def __init__(self, root):
        self.root = root
        self.root.title("积木与真实画面")
        self.config = load_config()
        self.camera = None
        self.camera_failures = 0
        self.hands = None
        self.hand_direction = 0
        self.hand_jump = False
        self.frame_count = 0
        self.keys = set()
        self.drag_start = None
        self.drag_current = None
        self.photo = None
        self.display_image = None
        self.playing = False
        self.starting = False
        self.language = "zh"
        self.menu_page = "home"
        self.menu_targets = []
        self.debug_open = False
        self.blood_mask = pixel_text_mask("血条", 14, 2)
        self.ball_sprites = [make_ball_sprite(.82 + .36 * index / 11) for index in range(12)]
        self.door_sprites = [make_door_sprite(.82 + .36 * index / 11) for index in range(12)]
        self.collision = CollisionMap(self.config)
        self.frame = np.full((CAM_H, CAM_W, 3), 255, np.uint8)
        self.source = ""
        self.build_ui()
        self.render_start()
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.last_tick = time.monotonic()

    def build_ui(self):
        self.root.configure(bg=CREAM)
        wrap = tk.Frame(self.root, bg=CREAM)
        wrap.pack()
        self.canvas = tk.Canvas(wrap, width=VIEW_W, height=VIEW_H, bg=CREAM, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.drag_begin)
        self.canvas.bind("<B1-Motion>", self.drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.drag_end)
        self.canvas.bind("<Motion>", self.menu_motion)
        self.bar = tk.Frame(wrap, bg=CREAM, pady=7)
        self.save_label = tk.StringVar(value=self.t("保存当前画面", "SAVE FRAME"))
        self.save_button = tk.Button(self.bar, textvariable=self.save_label,
                                     command=self.save_current_frame, bg=APRICOT, fg=CHARCOAL,
                                     activebackground=PALE_SAGE, activeforeground=CHARCOAL,
                                     relief="flat", bd=0, padx=16, pady=5,
                                     font=("Microsoft YaHei", 10))
        self.save_button.pack(side="left", padx=(14, 8))
        self.debug_toggle_label = tk.StringVar(value=self.t("调试 ▾", "TOOLS ▾"))
        self.debug_toggle = tk.Button(self.bar, textvariable=self.debug_toggle_label,
                                      command=self.toggle_debug_panel, bg=PALE_SAGE, fg=CHARCOAL,
                                      activebackground=SAGE, activeforeground=CHARCOAL,
                                      relief="flat", bd=0, padx=16, pady=5,
                                      font=("Microsoft YaHei", 10))
        self.debug_toggle.pack(side="left")
        self.debug_panel = tk.Frame(wrap, bg=CREAM, padx=14, pady=7)
        self.debug_controls = tk.Frame(self.debug_panel, bg=CREAM)
        self.debug_controls.pack(fill="x")
        self.edit_var = tk.BooleanVar(value=False)
        self.edit_label = tk.StringVar(value=self.t("圈定积木区域", "SELECT BLOCK AREA"))
        tk.Checkbutton(self.debug_controls, textvariable=self.edit_label, variable=self.edit_var,
                       command=self.toggle_roi, bg=CREAM, fg=CHARCOAL,
                       activebackground=CREAM, selectcolor=PALE_SAGE,
                       relief="flat", font=("Microsoft YaHei", 10)).pack(side="left", padx=(0, 12))
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_label = tk.StringVar(value=self.t("显示碰撞层", "SHOW COLLISIONS"))
        tk.Checkbutton(self.debug_controls, textvariable=self.debug_label, variable=self.debug_var,
                       bg=CREAM, fg=CHARCOAL, activebackground=CREAM,
                       selectcolor=PALE_SAGE, relief="flat",
                       font=("Microsoft YaHei", 10)).pack(side="left", padx=(0, 12))
        self.gesture_var = tk.BooleanVar(value=False)
        self.gesture_label = tk.StringVar(value=self.t("手势", "GESTURES"))
        tk.Checkbutton(self.debug_controls, textvariable=self.gesture_label, variable=self.gesture_var,
                       command=self.toggle_hands, bg=CREAM, fg=CHARCOAL,
                       activebackground=CREAM, selectcolor=PALE_SAGE,
                       relief="flat", font=("Microsoft YaHei", 10)).pack(side="left")
        self.status_var = tk.StringVar(value=self.play_instructions())
        self.status_label = tk.Label(self.debug_panel, textvariable=self.status_var,
                                     bg=CREAM, fg=CHARCOAL, font=("Microsoft YaHei", 9))
        self.status_label.pack(anchor="w", pady=(5, 0))

    def toggle_debug_panel(self):
        self.debug_open = not self.debug_open
        self.debug_toggle_label.set(self.t("调试 ▴", "TOOLS ▴") if self.debug_open
                                    else self.t("调试 ▾", "TOOLS ▾"))
        if self.debug_open:
            self.debug_panel.pack(fill="x")
        else:
            self.debug_panel.pack_forget()

    def t(self, chinese, english):
        return chinese if self.language == "zh" else english

    def play_instructions(self):
        return self.t("← → 移动  ↑ 二段跳  R 重开  Esc 返回首页",
                      "← → MOVE  ↑ DOUBLE JUMP  R RESTART  Esc HOME")

    def menu_label(self, image, value, center_y, size, scale, action=None, fill=None):
        mask = pixel_text_mask(value, size, scale)
        x = (VIEW_W - mask.width) // 2
        y = center_y - mask.height // 2
        if fill:
            ImageDraw.Draw(image).rectangle((x - 25, y - 16, x + mask.width + 25,
                                             y + mask.height + 16), fill=fill)
        image.paste(CHARCOAL, (x, y), mask)
        if action:
            self.menu_targets.append(((x - 24, y - 15, x + mask.width + 24, y + mask.height + 15), action))
        return x, y, mask.width, mask.height

    def show_menu_image(self, image):
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.configure(bg=CREAM, cursor="")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def render_start(self, message=None):
        self.menu_page = "home"
        self.menu_targets = []
        image = Image.new("RGB", (VIEW_W, VIEW_H), CREAM)
        self.menu_label(image, self.t("开始", "START"), 285, 25, 4,
                        self.start_game, APRICOT)
        self.menu_label(image, self.t("设置", "SETTINGS"), 420, 17, 3,
                        self.render_settings, PALE_SAGE)
        if message:
            self.menu_label(image, message, 550, 14, 3)
        self.show_menu_image(image)

    def render_settings(self):
        self.menu_page = "settings"
        self.menu_targets = []
        image = Image.new("RGB", (VIEW_W, VIEW_H), CREAM)
        self.menu_label(image, self.t("设置", "SETTINGS"), 190, 24, 4)
        choices = (("zh", "中文", "CHINESE", 315), ("en", "英文", "ENGLISH", 390))
        for language, chinese, english, y in choices:
            x, top, _, height = self.menu_label(
                image, self.t(chinese, english), y, 17, 3,
                lambda language=language: self.set_language(language),
                PALE_SAGE if self.language == language else None)
            if self.language == language:
                ImageDraw.Draw(image).rectangle((x - 30, top + height//2 - 6,
                                                x - 18, top + height//2 + 6), fill=SAGE)
        self.menu_label(image, self.t("返回", "BACK"), 505, 16, 3,
                        self.render_start, APRICOT)
        self.show_menu_image(image)

    def set_language(self, language):
        self.language = language
        self.blood_mask = pixel_text_mask(self.t("血条", "HP"), 14, 2)
        self.root.title(self.t("积木与真实画面", "BLOCKS AND REALITY"))
        self.edit_label.set(self.t("圈定积木区域", "SELECT BLOCK AREA"))
        self.save_label.set(self.t("保存当前画面", "SAVE FRAME"))
        self.debug_label.set(self.t("显示碰撞层", "SHOW COLLISIONS"))
        self.gesture_label.set(self.t("手势", "GESTURES"))
        self.debug_toggle_label.set(self.t("调试 ▴", "TOOLS ▴") if self.debug_open
                                    else self.t("调试 ▾", "TOOLS ▾"))
        self.status_var.set(self.play_instructions())
        self.render_settings()

    def render_connecting(self):
        self.menu_page = "connecting"
        self.menu_targets = []
        image = Image.new("RGB", (VIEW_W, VIEW_H), CREAM)
        self.menu_label(image, self.t("连接中", "CONNECTING"), VIEW_H // 2, 24, 4)
        self.show_menu_image(image)

    def menu_motion(self, event):
        if self.playing:
            return
        clickable = any(x0 <= event.x <= x1 and y0 <= event.y <= y1
                        for (x0, y0, x1, y1), _ in self.menu_targets)
        self.canvas.configure(cursor="hand2" if clickable else "")

    def menu_click(self, event):
        for (x0, y0, x1, y1), action in self.menu_targets:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                action()
                break

    def start_game(self):
        if self.playing or self.starting:
            return
        self.starting = True
        self.render_connecting()
        self.root.update_idletasks()
        connected = self.open_camera()
        self.starting = False
        if not connected:
            self.render_start(self.t("未检测到摄像头", "NO CAMERA FOUND"))
            return
        self.playing = True
        self.bar.pack(fill="x")
        self.last_tick = time.monotonic()
        self.render()
        self.root.after(33, self.tick)

    def reset(self):
        self.x, self.y = self.collision.starting_point()
        self.vx = self.vy = 0.0
        self.on_ground = False
        self.lives = 7
        self.score = 0
        self.river_inside = False
        self.collected = []
        self.won = False
        self.jump_count = 0
        self.jump_requested = False
        self.last_up_release = -1e9
        self.door_x, self.door_bottom = self.random_door_position()

    def door_clear(self, x, bottom):
        """Check the entire doorway, with a small gap from every colour mask."""
        padding = 4
        left = max(0, int((x - DOOR_HALF_W - padding) // CELL))
        right = min(CAM_W // CELL, int((x + DOOR_HALF_W + padding) // CELL) + 1)
        top = max(0, int((bottom - DOOR_HEIGHT - padding) // CELL))
        low = min(CAM_H // CELL, int((bottom + padding) // CELL) + 1)
        if left >= right or top >= low:
            return False
        return not any(np.any(self.collision.bits[name][top:low, left:right]) for name in MATERIALS)

    def random_door_position(self):
        # Put the exit across the level from the player's starting position.
        # Prefer the space beyond the detected blocks, including when the
        # recognition ROI covers the full camera picture.
        # Include the whole visible height, down to a small top margin.
        occupied = np.logical_or.reduce([self.collision.bits[name] for name in MATERIALS])
        columns = np.flatnonzero(occupied.any(axis=0))
        if len(columns):
            block_left, block_right = int(columns[0]) * CELL, (int(columns[-1]) + 1) * CELL
        else:
            block_left = block_right = CAM_W // 2
        start_on_left = self.x <= CAM_W / 2
        if start_on_left:
            outside = list(range(min(CAM_W - 30, block_right + DOOR_HALF_W + 6), CAM_W - 29, 8))
        else:
            outside = list(range(30, max(30, block_left - DOOR_HALF_W - 6), 8))
        all_x = [x for x in range(30, CAM_W - 29, 8)
                 if (x > self.x + 75 if start_on_left else x < self.x - 75)]
        ground = int(round(self.collision.ground_y))
        topmost_bottom = DOOR_HEIGHT + 8
        bottoms = list(range(ground, topmost_bottom - 1, -24))
        if bottoms and bottoms[-1] != topmost_bottom:
            bottoms.append(topmost_bottom)
        for positions in (outside, all_x):
            candidates = [(x, bottom) for x in positions for bottom in bottoms
                          if (x > self.x + 75 if start_on_left else x < self.x - 75)
                          and self.door_clear(x, bottom)]
            if candidates:
                return random.choice(candidates)
        # A completely filled image has no empty doorway; keep the fallback at
        # the opposite edge so the game remains playable after recalibration.
        return (CAM_W - 30 if start_on_left else 30), ground

    def key_down(self, event):
        key = event.keysym.lower()
        if not self.playing:
            if key == "escape" and self.menu_page == "settings":
                self.render_start()
            elif key in ("return", "space") and self.menu_page == "home":
                self.start_game()
            return
        if key == "escape":
            self.show_start()
            return
        if key == "up" and key not in self.keys and time.monotonic() - self.last_up_release > .04:
            self.jump_requested = True
        self.keys.add(key)
        if key == "r":
            self.reset()
        elif key in ("0", "1"):
            self.open_camera(int(key))

    def key_up(self, event):
        key = event.keysym.lower()
        self.keys.discard(key)
        if key == "up":
            self.last_up_release = time.monotonic()

    def open_camera(self, index=None):
        self.release_camera()
        preferred = self.config["camera"] if index is None else index
        self.status_var.set(self.t("连接中", "CONNECTING"))
        self.root.update_idletasks()
        for device in dict.fromkeys((preferred, 1, 0, 2, 3)):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                cap = cv2.VideoCapture(device, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    continue
                self.camera = cap
                self.camera_failures = 0
                self.config["camera"] = device
                self.source = f"Logi {device}"
                self.collision = CollisionMap(self.config)
                self.accept_frame(frame)
                self.reset()
                self.status_var.set(self.play_instructions())
                return True
        self.status_var.set(self.t("未检测到摄像头", "NO CAMERA FOUND"))
        if self.playing:
            self.show_start(self.t("未检测到摄像头", "NO CAMERA FOUND"))
        return False

    def show_start(self, message=None):
        self.release_camera()
        self.playing = False
        self.keys.clear()
        self.bar.pack_forget()
        self.debug_panel.pack_forget()
        self.debug_open = False
        self.debug_toggle_label.set(self.t("调试 ▾", "TOOLS ▾"))
        self.render_start(message)

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def accept_frame(self, frame):
        self.frame = normalized_frame(frame)
        self.collision.update(self.frame)

    def toggle_hands(self):
        if not self.gesture_var.get():
            if self.hands:
                self.hands.close()
                self.hands = None
            self.hand_direction = 0
            return
        try:
            self.hands = HandController()
            self.status_var.set(self.t("手势开启：手掌左右移动，捏合跳跃",
                                       "GESTURES ON: MOVE HAND, PINCH TO JUMP"))
        except Exception as exc:
            self.gesture_var.set(False)
            self.status_var.set(f"{self.t('手势不可用', 'GESTURES UNAVAILABLE')}: {exc}")

    def drag_begin(self, event):
        if not self.playing:
            self.menu_click(event)
            return
        if self.edit_var.get():
            self.drag_start = (event.x, event.y)
            self.drag_current = self.drag_start

    def toggle_roi(self):
        self.drag_start = self.drag_current = None
        if self.edit_var.get():
            self.status_var.set(self.t("在画面上拖动，框住积木区域", "DRAG TO SELECT BLOCK AREA"))
            return
        self.config["roi"] = [0.0, 0.0, 1.0, 1.0]
        self.collision = CollisionMap(self.config)
        self.collision.update(self.frame)
        self.reset()
        self.status_var.set(self.t("已恢复全画面颜色识别", "FULL FRAME DETECTION RESTORED"))

    def drag_move(self, event):
        if self.drag_start is not None:
            self.drag_current = (event.x, event.y)

    def drag_end(self, event):
        if self.drag_start is None:
            return
        x0, x1 = sorted((self.drag_start[0], event.x))
        y0, y1 = sorted((self.drag_start[1], event.y))
        self.drag_start = self.drag_current = None
        x0, x1 = max(0, x0), min(VIEW_W, x1)
        y0, y1 = max(0, y0), min(VIEW_H, y1)
        if x1 - x0 < 60 or y1 - y0 < 60:
            return
        self.config["roi"] = [x0 / VIEW_W, y0 / VIEW_H, (x1-x0) / VIEW_W, (y1-y0) / VIEW_H]
        self.collision = CollisionMap(self.config)
        self.collision.update(self.frame)
        self.reset()
        self.status_var.set(self.t("已按圈定区域识别积木；取消勾选可恢复全画面识别",
                                   "BLOCK AREA SET; UNCHECK TO USE THE FULL FRAME"))

    def save_current_frame(self):
        if self.display_image is None:
            return
        try:
            folder = ROOT / "captures"
            folder.mkdir(exist_ok=True)
            path = folder / f"画面_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            self.display_image.save(path)
            self.status_var.set(f"{self.t('当前画面已保存', 'FRAME SAVED')}: {path}")
        except OSError as exc:
            self.status_var.set(f"{self.t('保存画面失败', 'SAVE FAILED')}: {exc}")

    def hurt(self):
        self.lives -= 1
        if self.lives <= 0:
            self.reset()
            self.status_var.set(self.t("生命耗尽，已开始新一局", "OUT OF LIVES; NEW ROUND"))

    def physics(self, dt):
        if self.won:
            return
        direction = int("right" in self.keys) - int("left" in self.keys)
        if direction == 0:
            direction = self.hand_direction
        self.vx = direction * 165
        if (self.jump_requested or self.hand_jump) and self.jump_count < 2:
            self.vy = -420
            self.on_ground = False
            self.jump_count += 1
        self.jump_requested = False
        self.hand_jump = False
        if "down" in self.keys:
            self.vy = max(self.vy, 170)
        candidate_x = min(CAM_W - BALL_R, max(BALL_R, self.x + self.vx * dt))
        side_x = candidate_x + math.copysign(BALL_R, self.vx) if self.vx else candidate_x
        blocked = self.vx and any(self.collision.at("green", side_x, self.y + dy)
                                  for dy in (-BALL_R * .55, 0, BALL_R * .55))
        if not blocked:
            self.x = candidate_x
        old_bottom = self.y + BALL_R
        self.vy = min(650, self.vy + 1050 * dt)
        self.y += self.vy * dt
        self.on_ground = False
        if self.vy >= 0:
            surface = self.collision.green_surface(self.x, old_bottom, self.y + BALL_R)
            if surface is not None:
                self.y, self.vy, self.on_ground = surface - BALL_R, 0, True
                self.jump_count = 0
        ground = self.collision.ground_y
        if self.y + BALL_R >= ground:
            self.y, self.vy, self.on_ground = ground - BALL_R, 0, True
            self.jump_count = 0
        in_river = self.collision.touches("blue", self.x, self.y)
        if in_river and not self.river_inside:
            self.hurt()
        self.river_inside = in_river
        coins = self.collision.coins()
        self.collected = [point for point in self.collected if any(
            math.dist(point, coin[:2]) < 18 for coin in coins)]
        for cx, cy, w, h in coins:
            if abs(self.x - cx) < w / 2 + BALL_R and abs(self.y - cy) < h / 2 + BALL_R:
                if all(math.dist((cx, cy), point) >= 18 for point in self.collected):
                    self.score += 1
                    self.collected.append((cx, cy))
        if (self.x + BALL_R >= self.door_x - DOOR_HALF_W
                and self.x - BALL_R <= self.door_x + DOOR_HALF_W
                and self.y + BALL_R >= self.door_bottom - DOOR_HEIGHT
                and self.y - BALL_R <= self.door_bottom):
            self.won = True
            self.status_var.set(self.t("通关啦！按 R 开始新一局", "CLEAR! PRESS R FOR A NEW ROUND"))

    def build_display_image(self):
        shown = self.collision.tinted_debug_frame(self.frame) if self.debug_var.get() else self.frame
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        image = (Image.fromarray(rgb)
                 .resize((PIXEL_W, PIXEL_H), Image.Resampling.BOX)
                 .resize((VIEW_W, VIEW_H), Image.Resampling.NEAREST))
        draw = ImageDraw.Draw(image)
        s = VIEW_SCALE
        dx, dy = self.door_x * s, self.door_bottom * s
        draw.rectangle((round(dx - DOOR_HALF_W*s), round(dy - DOOR_HEIGHT*s),
                        round(dx + DOOR_HALF_W*s), round(dy)), fill="white")
        bx, by = self.x * VIEW_SCALE, self.y * VIEW_SCALE
        image.paste(self.ball_sprite,
                    (round(bx - self.ball_sprite.width / 2), round(by - self.ball_sprite.height / 2)),
                    self.ball_sprite)
        draw = ImageDraw.Draw(image)
        coin = ("001111100", "011222110", "112222211", "122232221", "122232221",
                "122232221", "112222211", "011222110", "001111100")
        coin_colors = {"1": "#E59600", "2": "#FFCA00", "3": "#FFF078"}
        for row, pattern in enumerate(coin):
            for col, value in enumerate(pattern):
                if value != "0":
                    x, y = 18 + col * 3, 16 + row * 3
                    draw.rectangle((x, y, x + 2, y + 2), fill=coin_colors[value])
        draw_pixel_text(draw, self.score, 55, 19, 3, "#171717")
        image.paste((20, 20, 20), (18, 58), self.blood_mask)
        for index in range(7):
            x = 85 + index * 16
            draw.rectangle((x, 56, x + 10, 85),
                           fill="#E93743" if index < self.lives else "#A9A9A9")
        if self.edit_var.get():
            rx, ry, rw, rh = self.config["roi"]
            draw.rectangle((round(rx*VIEW_W), round(ry*VIEW_H),
                            round((rx+rw)*VIEW_W), round((ry+rh)*VIEW_H)),
                           outline="#FFD447", width=3)
            if self.drag_start and self.drag_current:
                x0, x1 = sorted((self.drag_start[0], self.drag_current[0]))
                y0, y1 = sorted((self.drag_start[1], self.drag_current[1]))
                draw.rectangle((x0, y0, x1, y1), outline="#FFD447", width=3)
        if self.won:
            draw_pixel_text(draw, "CLEAR", 348, 100, 7, "#171717")
        return image

    def render(self):
        self.display_image = self.build_display_image()
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def tick(self):
        if not self.playing:
            return
        now = time.monotonic()
        dt = min(.05, now - self.last_tick)
        self.last_tick = now
        if self.camera is not None:
            ok, frame = self.camera.read()
            if ok:
                self.camera_failures = 0
                self.accept_frame(frame)
                self.frame_count += 1
                if self.hands and self.frame_count % 3 == 0:
                    try:
                        self.hand_direction, self.hand_jump = self.hands.read(self.frame)
                    except Exception as exc:
                        self.gesture_var.set(False)
                        self.toggle_hands()
                        self.status_var.set(f"{self.t('手势识别出错', 'GESTURE ERROR')}: {exc}")
            else:
                self.camera_failures += 1
                if self.camera_failures >= 8:
                    self.show_start(self.t("摄像头画面中断", "CAMERA DISCONNECTED"))
                    return
        self.physics(dt)
        self.render()
        self.root.after(max(1, 33 - int((time.monotonic() - now) * 1000)), self.tick)

    def close(self):
        self.release_camera()
        if self.hands:
            self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Game(window)
    window.mainloop()
