"""
Batch wrapper for my_bytetrack_es_v0.py: traverses a root hierarchy with
Annotation/ and Data/ and generates results in Results/ByteTrack/ for all
detected scenes and cameras.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from my_bytetrack import Config, run as run_tracker


# ==============================
# CONFIG
# ==============================
CAM_DIRS = ["fisheye_images_12", "fisheye_images_13", "fisheye_images_14", "output_images"]


# ==============================
# FILE DISCOVERY
# ==============================
def iter_annotation_scenes(root: Path) -> Iterable[Path]:
    ann_root = root / "Annotation"
    if not ann_root.is_dir():
        return []
    return [p for p in ann_root.iterdir() if p.is_dir()]


def find_annotation_json(cam_dir: Path) -> Optional[Path]:
    candidates = sorted(cam_dir.glob("cam*_ann.json"))
    return candidates[0] if candidates else None


# ==============================
# BATCH PROCESSING
# ==============================
def process_scene(root: Path, scene_ann_dir: Path) -> None:
    scene_ann_name = scene_ann_dir.name
    scene_label_name = scene_ann_name.replace("_json_files", "_label")

    data_scene_dir = root / "Data" / scene_label_name
    if not data_scene_dir.is_dir():
        print(f"[WARN] Data folder does not exist for scenario {scene_ann_name}: {data_scene_dir}")
        return

    out_scene_dir = root / "Results/ByteTrack" / scene_label_name

    for cam in CAM_DIRS:
        ann_cam_dir = scene_ann_dir / cam
        data_cam_dir = data_scene_dir / cam
        if not ann_cam_dir.is_dir():
            print(f"[WARN] Annotations folder is missing {ann_cam_dir}")
            continue
        if not data_cam_dir.is_dir():
            print(f"[WARN] Frames folder is missing {data_cam_dir}")
            continue

        json_path = find_annotation_json(ann_cam_dir)
        if json_path is None:
            print(f"[WARN] JSON file not found in {ann_cam_dir}")
            continue

        out_cam_dir = out_scene_dir / cam
        out_cam_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config(
            image_dir=data_cam_dir,
            detections_json=json_path,
            output_dir=out_cam_dir,
        )

        print(f"[INFO] Scenario {scene_ann_name} | Camera {cam}")
        print(f"       JSON: {json_path}")
        print(f"       Frames: {data_cam_dir}")
        print(f"       Output: {out_cam_dir}")
        run_tracker(cfg)


# ==============================
# MAIN
# ==============================
def main():
    p = argparse.ArgumentParser(
        description="Processes all scenarios/cameras from a root directory containing Annotation/ and Data/ folders, and generates tracking results in Results/ByteTrack/"
    )
    p.add_argument("--root", required=True, type=Path, help="Root directory containing Annotation/ and Data/ folders")

    args = p.parse_args()
    root = args.root.resolve()

    scenes = list(iter_annotation_scenes(root))
    if not scenes:
        print(f"[ERROR] No scenarios found in {root/'Annotation'}")
        return

    for scene_ann_dir in scenes:
        process_scene(root, scene_ann_dir)


if __name__ == "__main__":
    main()
