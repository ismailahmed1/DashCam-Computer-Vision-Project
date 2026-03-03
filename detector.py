from ultralytics import YOLO

# COCO class IDs relevant to road scenes
ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}


class DashcamDetector:
    def __init__(self, model_size: str = "n", conf: float = 0.4):
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.conf = conf

    def detect(self, frame) -> list[dict]:
        results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in ROAD_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": ROAD_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                    "track_id": int(box.id[0]) if box.id is not None else None,
                }
            )
        return detections
