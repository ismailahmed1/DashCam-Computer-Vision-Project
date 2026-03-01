import cv2
import numpy as np

# Per-class box colors (BGR)
CLASS_COLORS = {
    "person":        (0,   220, 255),   # cyan-yellow
    "bicycle":       (50,  180, 255),   # orange
    "motorcycle":    (50,  180, 255),
    "car":           (200, 200, 200),   # white
    "bus":           (200, 200, 200),
    "truck":         (200, 200, 200),
    "traffic light": (0,   255, 100),   # green
    "stop sign":     (0,   80,  220),   # red
}

HUD_HEIGHT = 90


def _draw_label(frame, text: str, x: int, y: int, color, font_scale=0.45, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad), (x + tw + pad * 2, y + baseline), (15, 15, 15), -1)
    cv2.putText(frame, text, (x + pad, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(name, (180, 180, 180))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _draw_label(frame, f"{name}  {conf:.0%}", x1, y1 - 4, color)


def draw_hud(frame, score: float, label: str, color: tuple, detections: list[dict]) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent dark banner
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, HUD_HEIGHT), (12, 12, 12), -1)
    cv2.addWeighted(banner, 0.72, frame, 0.28, 0, frame)

    # Colored left accent bar
    cv2.rectangle(frame, (0, 0), (6, HUD_HEIGHT), color, -1)

    # ── DANGER LEVEL label + value ────────────────────────────────────────────
    cv2.putText(frame, "DANGER LEVEL", (20, 22), font, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (20, 68), font, 1.6, color, 3, cv2.LINE_AA)

    # ── Score bar ─────────────────────────────────────────────────────────────
    bar_x, bar_y = 230, 28
    bar_w, bar_h = min(w - 460, 360), 22
    fill_w = int(bar_w * score / 100)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    cv2.putText(frame, f"{score:.0f} / 100", (bar_x, bar_y + bar_h + 18),
                font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Level ticks on bar
    for pct, tick_label in [(0.25, "25"), (0.5, "50"), (0.75, "75")]:
        tx = bar_x + int(bar_w * pct)
        cv2.line(frame, (tx, bar_y), (tx, bar_y + bar_h), (100, 100, 100), 1)

    # ── Detected objects chip strip ───────────────────────────────────────────
    counts: dict[str, int] = {}
    for det in detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1

    chip_x = bar_x
    chip_y = bar_y + bar_h + 38
    cv2.putText(frame, "DETECTED:", (chip_x, chip_y - 14),
                font, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    for name, cnt in counts.items():
        chip_text = f" {cnt}× {name} "
        c_color = CLASS_COLORS.get(name, (180, 180, 180))
        (cw, ch), _ = cv2.getTextSize(chip_text, font, 0.40, 1)
        cv2.rectangle(frame, (chip_x - 2, chip_y - ch - 3), (chip_x + cw + 2, chip_y + 3), (35, 35, 35), -1)
        cv2.rectangle(frame, (chip_x - 2, chip_y - ch - 3), (chip_x + cw + 2, chip_y + 3), c_color, 1)
        cv2.putText(frame, chip_text, (chip_x, chip_y), font, 0.40, c_color, 1, cv2.LINE_AA)
        chip_x += cw + 10
        if chip_x > w - 120:
            break

    # ── Critical flash border ─────────────────────────────────────────────────
    if label == "CRITICAL":
        border = 4
        cv2.rectangle(frame, (0, HUD_HEIGHT), (border, h), color, -1)
        cv2.rectangle(frame, (w - border, HUD_HEIGHT), (w, h), color, -1)
        cv2.rectangle(frame, (0, h - border), (w, h), color, -1)


def annotate_frame(frame, detections: list[dict], score: float, label: str, color: tuple):
    """Full pipeline: boxes + HUD on a single frame (modifies in-place)."""
    draw_detections(frame, detections)
    draw_hud(frame, score, label, color, detections)
