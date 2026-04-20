"""
scripts/image_processing.py
───────────────────────────
Standalone CLI runner for image preprocessing only (orientation + deblur).

Writes:
    - data/processed/images/** (upright + deblurred where applicable)
    - data/processed/preprocess_report.json
    - data/processed/preprocess_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mlflow
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.data import DEFAULT_DATASET_DIR, IMC2025TrainData
from pipelines.scene import Scene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("image_processing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files


def _relative_to_data_root(path: Path, data_dir: Path) -> Path:
    data_dir = data_dir.resolve()
    p = path.resolve() if path.is_absolute() else path
    if p.is_absolute():
        return p.relative_to(data_dir)
    if p.parts and p.parts[0] == data_dir.name:
        return p.relative_to(data_dir.name)
    return p


def _to_upright_if_needed(img: Any, degree: int) -> Any:
    if degree in (0, None):
        return img
    from preprocesses.orientation import OrientationNormalizer

    normalizer = OrientationNormalizer(int(degree))
    return normalizer.set_original_image(img).get_upright_image_ndarray()


def _save_processed_images(scene: Scene, data_dir: Path, processed_images_dir: Path) -> dict[str, int]:
    saved_count = 0
    failed_count = 0
    orientation_applied_count = 0

    for image_path in scene.image_paths:
        src_path = Path(image_path)
        img = scene.get_image(image_path)

        degree = scene.get_orientation_degree(image_path) or 0
        if degree:
            orientation_applied_count += 1
            try:
                img = _to_upright_if_needed(img, int(degree))
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                logger.warning("Orientation normalization failed for %s: %s", image_path, exc)

        rel = _relative_to_data_root(src_path, data_dir)
        out_path = processed_images_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(out_path), img)
        if ok:
            saved_count += 1
        else:
            failed_count += 1

    return {
        "saved_count": int(saved_count),
        "failed_count": int(failed_count),
        "orientation_applied_count": int(orientation_applied_count),
    }


def _build_scene_groups(
    data_schema: IMC2025TrainData,
    data_dir: Path,
) -> list[tuple[str, str, list[str], str]]:
    groups: list[tuple[str, str, list[str], str]] = []

    # Train scenes from labels
    for (dataset, scene_name), group in data_schema.df.groupby(["dataset", "scene"]):
        image_paths = [data_schema.resolve_image_path(row) for _, row in group.iterrows()]
        image_paths = [str(Path(p)) for p in image_paths if Path(p).exists()]
        if image_paths:
            groups.append((str(dataset), str(scene_name), image_paths, "train"))

    return groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image preprocessing: orientation + deblur")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Restrict to these dataset names (e.g. ETs stairs)",
    )
    p.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Restrict to these scene names",
    )
    p.add_argument(
        "--preprocess-conf",
        default=str(ROOT / "conf" / "preprocess.yaml"),
        help="Path to preprocessing config YAML",
    )
    p.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Process at most N scenes (useful for testing)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scenes that would be processed without running",
    )
    p.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    p.add_argument("--data-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument(
        "--preprocess-report-path",
        default=None,
        help="Path to write preprocessing report JSON",
    )
    p.add_argument(
        "--preprocess-metrics-path",
        default=None,
        help="Path to write preprocessing metrics JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    processed_images_dir = processed_dir / "images"
    preprocess_report_path = Path(
        args.preprocess_report_path or (processed_dir / "preprocess_report.json")
    )
    preprocess_metrics_path = Path(
        args.preprocess_metrics_path or (processed_dir / "preprocess_metrics.json")
    )
    preprocess_conf_path = Path(args.preprocess_conf)

    data_schema = IMC2025TrainData.create(
        data_root_dir=data_dir,
        datasets_to_use=args.datasets,
        scenes_to_use=args.scenes,
    )
    data_schema.preprocess()
    logger.info("Data schema loaded: %d rows", len(data_schema.df))

    scene_groups = _build_scene_groups(data_schema=data_schema, data_dir=data_dir)
    if args.max_scenes:
        scene_groups = scene_groups[: args.max_scenes]

    if args.dry_run:
        print(f"\nDRY RUN - would process {len(scene_groups)} scenes:")
        for ds, sc, image_paths, split in scene_groups:
            print(f"  [{split}] {ds}/{sc}  ({len(image_paths)} images)")
        return

    if processed_images_dir.exists():
        shutil.rmtree(processed_images_dir)
    processed_images_dir.mkdir(parents=True, exist_ok=True)

    from preprocesses.pipeline import PreprocessingConfig, PreprocessingPipeline

    preprocess_conf = PreprocessingConfig.from_yaml(args.preprocess_conf)
    preprocess_pipeline = PreprocessingPipeline(preprocess_conf, device=device)

    logger.info("Processing %d scenes", len(scene_groups))

    t_total = time.perf_counter()
    preprocess_rows: list[dict[str, Any]] = []

    for dataset, scene_name, image_paths, split in scene_groups:
        if not image_paths:
            logger.warning("No images found for %s/%s, skipping", dataset, scene_name)
            continue

        scene = Scene(
            dataset=dataset,
            scene=scene_name,
            image_paths=image_paths,
            image_dir=str(Path(image_paths[0]).parent),
            data_schema=data_schema,
        )

        with scene.create_space():
            scene.cache_all_images()
            preprocess_pipeline.run(scene)
            persist_stats = _save_processed_images(
                scene=scene,
                data_dir=data_dir,
                processed_images_dir=processed_images_dir,
            )
            scene.release_cached_images()

        preprocess_rows.append(
            {
                "split": split,
                "dataset": dataset,
                "scene": scene_name,
                "n_images": len(image_paths),
                "orientation_count": len(scene.orientations),
                "deblurred_count": len(scene.deblurred_images),
                "segmentation_mask_count": len(scene.segmentation_mask_images),
                "depth_map_count": len(scene.depth_images),
                "processed_saved_count": persist_stats["saved_count"],
                "processed_failed_count": persist_stats["failed_count"],
                "orientation_applied_count": persist_stats["orientation_applied_count"],
            }
        )

    elapsed = time.perf_counter() - t_total

    preprocess_report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "active_steps": preprocess_conf.active_steps(),
        "scenes_processed": len(preprocess_rows),
        "images_seen": int(sum(r["n_images"] for r in preprocess_rows)),
        "orientation_updated_total": int(sum(r["orientation_count"] for r in preprocess_rows)),
        "deblurred_total": int(sum(r["deblurred_count"] for r in preprocess_rows)),
        "segmentation_masks_total": int(sum(r["segmentation_mask_count"] for r in preprocess_rows)),
        "depth_maps_total": int(sum(r["depth_map_count"] for r in preprocess_rows)),
        "processed_images_saved_total": int(sum(r["processed_saved_count"] for r in preprocess_rows)),
        "processed_images_failed_total": int(sum(r["processed_failed_count"] for r in preprocess_rows)),
        "orientation_applied_total": int(sum(r["orientation_applied_count"] for r in preprocess_rows)),
        "total_elapsed_sec": round(float(elapsed), 2),
        "processed_images_dir": str(processed_images_dir),
        "per_scene": preprocess_rows,
    }

    preprocess_metrics = {
        "scenes_processed": preprocess_report["scenes_processed"],
        "images_seen": preprocess_report["images_seen"],
        "orientation_updated_total": preprocess_report["orientation_updated_total"],
        "deblurred_total": preprocess_report["deblurred_total"],
        "segmentation_masks_total": preprocess_report["segmentation_masks_total"],
        "depth_maps_total": preprocess_report["depth_maps_total"],
        "processed_images_saved_total": preprocess_report["processed_images_saved_total"],
        "processed_images_failed_total": preprocess_report["processed_images_failed_total"],
        "orientation_applied_total": preprocess_report["orientation_applied_total"],
        "total_elapsed_sec": preprocess_report["total_elapsed_sec"],
    }

    _write_json(preprocess_report_path, preprocess_report)
    _write_json(preprocess_metrics_path, preprocess_metrics)

    try:
        mlflow_uri = args.mlflow_uri
        if "://" not in mlflow_uri:
            mlflow_uri = f"http://{mlflow_uri}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("scene_reconstruction_dvc")
        parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
        run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

        with mlflow.start_run(run_name="image_preprocess", nested=True, tags=run_tags):
            mlflow.log_params(
                {
                    "n_scenes": len(scene_groups),
                    "device": str(device),
                    "datasets": str(args.datasets),
                    "preprocess_conf": args.preprocess_conf,
                }
            )
            mlflow.log_metrics(
                {
                    f"preprocess_{k}": float(v)
                    for k, v in preprocess_metrics.items()
                    if _is_number(v)
                }
            )
            mlflow.log_artifact(str(preprocess_conf_path), artifact_path="preprocess_conf")
            mlflow.log_artifact(str(preprocess_report_path), artifact_path="reports")
            mlflow.log_artifact(str(preprocess_metrics_path), artifact_path="metrics")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("MLflow logging skipped: %s", exc)

    logger.info("Image preprocessing complete: %d scenes in %.1fs", len(preprocess_rows), elapsed)


if __name__ == "__main__":
    main()
