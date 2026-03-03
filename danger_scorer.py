import numpy as np
from collections import deque

# How much each object class contributes to danger
CLASS_WEIGHTS = {
    "person":        2.5,   # highest — pedestrian strikes are critical
    "bicycle":       1.8,
    "motorcycle":    1.8,
    "car":           1.0,
    "bus":           1.2,
    "truck":         1.3,
    "traffic light": 0.2,
    "stop sign":     0.2,
}

# (min_score, label, BGR color)
DANGER_LEVELS = [
    (75, "CRITICAL", (0,   30,  220)),
    (50, "HIGH",     (0,  120,  255)),
    (25, "CAUTION",  (0,  210,  255)),
    (0,  "LOW",      (60, 200,   60)),
]


class DangerScorer:
    def __init__(self, smoothing_window: int = 6):
        self._history: deque[float] = deque(maxlen=smoothing_window)
        self._prev_areas: dict[int, float] = {}  # track_id → previous box area

    def score(self, detections: list[dict], frame_w: int, frame_h: int) -> float:
        frame_area = frame_w * frame_h
        # Horizontal danger corridor: center third of frame
        cx_min = frame_w * 0.30
        cx_max = frame_w * 0.70

        raw = 0.0
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            box_area = max((x2 - x1) * (y2 - y1), 1)

            # Proximity: fraction of frame covered
            proximity = (box_area / frame_area) ** 0.5

            class_w = CLASS_WEIGHTS.get(det["class_name"], 1.0)

            # Position in danger corridor
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            lateral_w  = 1.6 if cx_min <= cx <= cx_max else 1.0
            vertical_w = 1.4 if cy > frame_h * 0.35 else 1.0

            # Approach velocity: boost score if box is growing (object closing in)
            track_id = det.get("track_id")
            velocity_w = 1.0
            if track_id is not None and track_id in self._prev_areas:
                growth = (box_area - self._prev_areas[track_id]) / self._prev_areas[track_id]
                velocity_w = min(1.0 + max(growth, 0) * 3.0, 2.0)
            if track_id is not None:
                self._prev_areas[track_id] = box_area

            raw += proximity * class_w * lateral_w * vertical_w * velocity_w * 120

        score = min(raw, 100.0)
        self._history.append(score)
        return float(np.mean(self._history))

    @staticmethod
    def get_level(score: float) -> tuple[str, tuple[int, int, int]]:
        for threshold, label, color in DANGER_LEVELS:
            if score >= threshold:
                return label, color
        return "LOW", (60, 200, 60)

    def reset(self):
        self._history.clear()
        self._prev_areas.clear()
