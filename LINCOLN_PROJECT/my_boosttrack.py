from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from boxmot import BoostTrack


# ==============================
# CONFIG
# ==============================
@dataclass
class Config:
    image_dir: Path
    detections_json: Path
    output_dir: Path
    reid_weights: Path

    # Device is kept internal for now
    device: str = "auto"

    # Output layout (fixed to match the project convention)
    output_video_name: str = "tracking.mp4"
    mot_output_name: str = "results.txt"
    save_frames_dir_name: str = "results_frames"

    # Drawing
    box_thickness: int = 2
    id_offset: int = 10
    fps: float = 30.0


# ==============================
# FORMAT UTILITIES
# ==============================
def _to_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert [x, y, w, h] -> [x1, y1, x2, y2].
    If width/height are not positive, assume the input already comes as xyxy.
    """
    x, y, w, h = map(float, b)
    if w > 0 and h > 0:
        return x, y, x + w, y + h
    return x, y, w, h


def _labels_iter(labels_field) -> Iterable[Tuple[str, Tuple[float, float, float, float]]]:
    """Yield (label, bbox_xyxy) from any supported `Labels` structure.

    Compatible with:
      - {"Class": str, "BoundingBoxes": [x, y, w, h]}
      - {"label": str, "bbox": [x, y, w, h]}
      - [{...}, {...}]
      - None / []
    """
    if labels_field is None:
        return []

    if isinstance(labels_field, dict):
        lab = labels_field.get("Class") or labels_field.get("class") or labels_field.get("label") or labels_field.get("Label")
        bb = labels_field.get("BoundingBoxes") or labels_field.get("bbox") or labels_field.get("box")
        if lab is not None and bb is not None:
            yield str(lab), _to_xyxy(bb)
        return

    if isinstance(labels_field, list):
        for item in labels_field:
            if not isinstance(item, dict):
                continue
            lab = item.get("Class") or item.get("class") or item.get("label") or item.get("Label")
            bb = item.get("BoundingBoxes") or item.get("bbox") or item.get("box")
            if lab is None or bb is None:
                continue
            yield str(lab), _to_xyxy(bb)
        return

    return []


def detections_from_json_record(rec: Dict, class_map: Dict[str, int]) -> np.ndarray:
    """Convert one JSON record into a BoxMOT detections array Nx6.

    Output format per detection: [x1, y1, x2, y2, conf, cls]
    """
    dets = []
    dropped = 0

    for lab, (x1, y1, x2, y2) in _labels_iter(rec.get("Labels")):
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue

        if lab not in class_map:
            class_map[lab] = len(class_map)
        cls_id = float(class_map[lab])

        dets.append([x1, y1, x2, y2, 1.0, cls_id])

    if dropped:
        warnings.warn(f"Dropped {dropped} invalid bbox(es) in record {rec.get('File', '<no-file>')}")

    if not dets:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(dets, dtype=np.float32)


# ==============================
# DEVICE AND VISUAL UTILITIES
# ==============================
def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _use_half(device: str) -> bool:
    return "cuda" in str(device) and torch.cuda.is_available()


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR color from track id."""
    tid = int(track_id)
    # Simple stable HSV-like hashing projected to BGR
    h = (tid * 47) % 180
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ==============================
# TRACKER FACTORY
# ==============================
def build_tracker(cfg: Config) -> BoostTrack:
    device = _resolve_device(cfg.device)
    half = _use_half(device)

    if "cuda" in str(device):
        print("Running with GPU (CUDA)")
    else:
        print("Running with CPU")

    # Match the effective CLI/default repo behavior for BoostTrack
    return BoostTrack(
        reid_weights=cfg.reid_weights,      # BoostTrack uses 'model_weights' as the argument name
        device=device,                      # Resolved device string
        half=half,                  # Use half precision only if CUDA is available  
        max_age=60,                 # Frames to keep a lost track
        min_hits=3,                 # Minimum frames to consider a track valid
        det_thresh=0.6,             # Detection confidence threshold (not used if input dets already have conf)
        iou_threshold=0.3,          # IOU threshold for matching
        use_ecc=True,               # Use ECC for motion compensation
        min_box_area=10,            # Minimum area of boxes to consider (filter out noise)  
        aspect_ratio_thresh=1.6,    # Maximum aspect ratio (w/h or h/w) to consider valid
        cmc_method="ecc",           # Motion compensation method: "ecc" or "kalman"
        lambda_iou=0.5,             # Weight for IOU cost
        lambda_mhd=0.25,            # Weight for MHD cost
        lambda_shape=0.25,          # Weight for shape cost
        use_dlo_boost=True,         # Whether to use DLOBoost (dynamic learning of optimal boost coefficient)
        use_duo_boost=True,         # Whether to use DUOBoost (dynamic update of boost coefficient)
        dlo_boost_coef=0.65,        # Initial boost coefficient for DLOBoost (will be dynamically adjusted)
        s_sim_corr=False,           # Whether to apply similarity correction to the boost coefficient
        use_rich_s=True,            # Whether to use rich features for ReID (if with_reid is True)
        use_sb=True,                # Whether to use spatial boosting (SB) in the matching cost
        use_vt=True,                # Whether to use velocity term (VT) in the matching cost
        with_reid=True,             # Whether to use ReID features for matching (requires reid_weights)
        per_class=False,            # Whether to maintain separate trackers per class (not needed if input dets already have class in the array)
    )


# ==============================
# MOT EXPORT
# ==============================
class MotWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    @staticmethod
    def _normalize_conf(value: float) -> float:
        value = float(value)
        if value > 1.0 and value <= 100.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    def write_frame(self, frame_idx: int, tracks: np.ndarray):
        lines = []
        for t in tracks:
            if len(t) < 5:
                continue
            track_id = int(t[4])
            x1, y1, x2, y2 = map(float, t[:4])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            conf = 1.0 if len(t) <= 5 else self._normalize_conf(float(t[5]))
            # MOTChallenge-like export: frame,id,x,y,w,h,conf,-1,-1,-1
            lines.append(f"{frame_idx},{track_id},{x1:f},{y1:f},{w:f},{h:f},{conf:f},-1,-1,-1\n")
        with open(self.path, "a", encoding="utf-8") as f:
            f.writelines(lines)


# ==============================
# MAIN PIPELINE
# ==============================
def run(cfg: Config):
    recs = json.loads(cfg.detections_json.read_text(encoding="utf-8"))
    tracker = build_tracker(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_frames_dir = cfg.output_dir / cfg.save_frames_dir_name
    save_frames_dir.mkdir(parents=True, exist_ok=True)
    mot_writer = MotWriter(cfg.output_dir / cfg.mot_output_name)

    writer = None
    first_frame_size = None
    class_map: Dict[str, int] = {}

    for frame_idx, rec in enumerate(recs, start=1):
        dets = detections_from_json_record(rec, class_map)

        file_name = Path(rec.get("File", f"frame_{frame_idx:05d}.png")).name
        frame_path = cfg.image_dir / file_name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            warnings.warn(f"Missing or unreadable frame: {frame_path}")
            if first_frame_size is None:
                first_frame_size = (720, 1280, 3)
            frame = np.zeros(first_frame_size, dtype=np.uint8)
        else:
            first_frame_size = frame.shape

        tracks = tracker.update(dets, frame)
        mot_writer.write_frame(frame_idx, tracks)

        for t in tracks:
            if len(t) < 5:
                continue
            track_id = int(t[4])
            x1, y1, x2, y2 = map(int, t[:4])
            color = _color_for_id(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.box_thickness)
            cv2.putText(
                frame,
                f"ID{track_id}",
                (x1, max(0, y1 - cfg.id_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(cfg.output_dir / cfg.output_video_name), fourcc, cfg.fps, (w, h))
        writer.write(frame)

        cv2.imwrite(str(save_frames_dir / file_name), frame)

    if writer is not None:
        writer.release()

    print("Processing completed:")
    print(f"- Frames processed: {len(recs)}")
    print(f"- MOT results: {cfg.output_dir / cfg.mot_output_name}")
    print(f"- Video: {cfg.output_dir / cfg.output_video_name}")
    print(f"- Frames: {save_frames_dir}")


# ==============================
# MAIN
# ==============================
def main():
    p = argparse.ArgumentParser(description="BoostTrack wrapper for MOT tracking from JSON detections")
    p.add_argument("--image_dir", required=True, type=Path, help="Path to the frames folder")
    p.add_argument("--json_path", required=True, type=Path, help="Path to the detections JSON file")
    p.add_argument("--output_dir", required=True, type=Path, help="Path to the output folder")
    p.add_argument("--model_path", required=True, type=Path, help="Path to the ReID weights file")
    args = p.parse_args()

    cfg = Config(
        image_dir=args.image_dir,
        detections_json=args.json_path,
        output_dir=args.output_dir,
        reid_weights=args.model_path,
    )
    run(cfg)


if __name__ == "__main__":
    main()
