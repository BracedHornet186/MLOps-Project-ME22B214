"""
scripts/validate_custom_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DEFAULT_DATASET_DIR


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def main() -> None:
    output_dir = Path(DEFAULT_DATASET_DIR) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(DEFAULT_DATASET_DIR) / "test"

    if not test_dir.exists():
        report = {
            "status": "warn",
            "message": "No test directory found",
            "total_images": 0,
            "total_scenes": 0,
            "unreadable_images": 0,
        }
        _write_outputs(output_dir, report)
        return

    # Find all image files
    all_images = [
        p for p in test_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # Group by parent directory (scene)
    scenes: dict[str, list[Path]] = {}
    for img_path in all_images:
        scene_name = img_path.parent.name
        scenes.setdefault(scene_name, []).append(img_path)

    # Validate readability
    unreadable = 0
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            unreadable += 1
            print(f"[warn] Unreadable image: {img_path}")

    # Check minimum images per scene
    small_scenes = [
        name for name, imgs in scenes.items()
        if len(imgs) < 3
    ]

    status = "ok"
    if unreadable > 0 or len(small_scenes) > 0:
        status = "warn"

    report = {
        "status": status,
        "total_images": len(all_images),
        "total_scenes": len(scenes),
        "unreadable_images": unreadable,
        "scenes_with_lt_3_images": len(small_scenes),
        "small_scene_names": small_scenes,
        "scenes": {name: len(imgs) for name, imgs in sorted(scenes.items())},
    }

    _write_outputs(output_dir, report)


def _write_outputs(output_dir: Path, report: dict) -> None:
    with open(output_dir / "custom_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    metrics = {
        "custom_total_images": report.get("total_images", 0),
        "custom_total_scenes": report.get("total_scenes", 0),
        "custom_unreadable": report.get("unreadable_images", 0),
    }
    with open(output_dir / "custom_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
