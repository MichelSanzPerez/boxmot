"""
Batch wrapper for my_botsort_es_v0.py: traverses a root hierarchy with
Annotation/ and Data/ and generates results in Results/BotSort/ for all
detected scenes and cameras.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from my_botsort import Config, run as run_tracker


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
def process_scene(root: Path, scene_ann_dir: Path, args) -> None:
    scene_ann_name = scene_ann_dir.name
    scene_label_name = scene_ann_name.replace("_json_files", "_label")

    data_scene_dir = root / "Data" / scene_label_name
    if not data_scene_dir.is_dir():
        print(f"[WARN] Data folder does not exist for scenario {scene_ann_name}: {data_scene_dir}")
        return

    out_scene_dir = root / "Results/BotSort" / scene_label_name

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
        cfg = Config(
            image_dir=data_cam_dir,
            detections_json=json_path,
            output_dir=out_cam_dir,
            reid_weights=args.model_path,
        )

        print(f"[INFO] Scenario={scene_label_name} | camera={cam}")
        run_tracker(cfg)


# ==============================
# MAIN
# ==============================
def main():
    p = argparse.ArgumentParser(
        description="Process all scenarios/cameras from a root directory containing Annotation/ and Data/ folders, and generates tracking results in Results/BotSort/"
    )
    p.add_argument("--root", required=True, type=Path, help="Root directory containing Annotation/ and Data/")
    p.add_argument("--model_path", required=True, type=Path, help="Path to the ReID weights file")
    args = p.parse_args()

    root = args.root
    scenes = list(iter_annotation_scenes(root))
    if not scenes:
        print(f"[WARN] No scenarios found in {root / 'Annotation'}")
        return

    for scene_ann_dir in scenes:
        process_scene(root, scene_ann_dir, args)


if __name__ == "__main__":
    main()
