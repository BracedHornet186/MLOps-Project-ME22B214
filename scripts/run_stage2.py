"""
scripts/run_stage2.py
──────────────────────
Standalone CLI runner for Stage 2: Preprocessing & Feature Engineering.
Use this when running outside Docker/Airflow (e.g. local dev, pytest).

Usage:
  # Run full Stage 2 on all scenes
  python scripts/run_stage2.py

  # Run on specific datasets only
  python scripts/run_stage2.py --datasets ETs stairs

  # Skip preprocessing, only extract features
  python scripts/run_stage2.py --skip-preprocess

  # Dry run — print what would be done without executing
  python scripts/run_stage2.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.data import IMC2025TrainData, DEFAULT_DATASET_DIR
from pipelines.scene import Scene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_stage2")


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Preprocessing & Features")
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help="Restrict to these dataset names (e.g. ETs stairs)",
    )
    p.add_argument(
        "--scenes", nargs="*", default=None,
        help="Restrict to these scene names",
    )
    p.add_argument(
        "--preprocess-conf", default=str(ROOT / "conf" / "preprocess.yaml"),
        help="Path to preprocessing config YAML",
    )
    p.add_argument(
        "--features-conf", default=str(ROOT / "conf" / "features.yaml"),
        help="Path to feature engineering config YAML",
    )
    p.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip image preprocessing, jump straight to feature extraction",
    )
    p.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature extraction (preprocessing only)",
    )
    p.add_argument(
        "--max-scenes", type=int, default=None,
        help="Process at most N scenes (useful for testing)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print scenes that would be processed without running",
    )
    p.add_argument(
        "--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    p.add_argument(
        "--data-dir", default=str(DEFAULT_DATASET_DIR),
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("stage2_preprocess_features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)

    # ── Load data schema ───────────────────────────────────────────────────────
    data_schema = IMC2025TrainData.create(
        data_root_dir=data_dir,
        datasets_to_use=args.datasets,
        scenes_to_use=args.scenes,
    )
    data_schema.preprocess()
    logger.info(f"Data schema loaded: {len(data_schema.df)} rows")

    # ── Build scene list ───────────────────────────────────────────────────────
    scene_groups = list(data_schema.df.groupby(["dataset", "scene"]))
    if args.max_scenes:
        scene_groups = scene_groups[: args.max_scenes]

    if args.dry_run:
        print(f"\nDRY RUN — would process {len(scene_groups)} scenes:")
        for (ds, sc), grp in scene_groups:
            print(f"  {ds}/{sc}  ({len(grp)} images)")
        return

    logger.info(f"Processing {len(scene_groups)} scenes")

    # ── Load configs ───────────────────────────────────────────────────────────
    if not args.skip_preprocess:
        from preprocesses.pipeline import PreprocessingPipeline, PreprocessingConfig
        preprocess_conf = PreprocessingConfig.from_yaml(args.preprocess_conf)
        preprocess_pipeline = PreprocessingPipeline(preprocess_conf, device=device)

    if not args.skip_features:
        from features.engineer import FeatureEngineer, FeatureEngineeringConfig
        features_conf = FeatureEngineeringConfig.from_yaml(args.features_conf)
        # Point output and baselines to actual data dir
        features_conf.output_dir = str(data_dir / "processed" / "features")
        features_conf.baselines_path = str(data_dir / "processed" / "eda_baselines.json")
        feature_engineer = FeatureEngineer(features_conf, device=device)

    # ── Run per scene ──────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    n_done = 0

    with mlflow.start_run(run_name="stage2_full"):
        mlflow.log_params({
            "n_scenes": len(scene_groups),
            "skip_preprocess": args.skip_preprocess,
            "skip_features": args.skip_features,
            "device": str(device),
            "datasets": str(args.datasets),
        })

        for (dataset, scene_name), group in scene_groups:
            image_paths = [
                data_schema.resolve_image_path(row)
                for _, row in group.iterrows()
            ]
            image_paths = [p for p in image_paths if Path(p).exists()]
            if not image_paths:
                logger.warning(f"No images found for {dataset}/{scene_name}, skipping")
                continue

            scene = Scene(
                dataset=dataset,
                scene=scene_name,
                image_paths=image_paths,
                image_dir=str(Path(image_paths[0]).parent),
                data_schema=data_schema,
            )

            # Pre-cache images for faster processing
            scene.cache_all_images()

            # ── Preprocessing ──────────────────────────────────────────────
            if not args.skip_preprocess:
                preprocess_pipeline.run(scene)

            # ── Feature extraction ─────────────────────────────────────────
            if not args.skip_features:
                result = feature_engineer.run(scene)
                if result.drift_alerts:
                    logger.warning(
                        f"[{dataset}/{scene_name}] "
                        f"Drift alerts: {result.drift_alerts}"
                    )

            scene.release_cached_images()
            n_done += 1
            logger.info(
                f"[{n_done}/{len(scene_groups)}] "
                f"{dataset}/{scene_name} done"
            )

        elapsed = time.perf_counter() - t_total
        mlflow.log_metrics({
            "scenes_processed": n_done,
            "total_elapsed_sec": round(elapsed, 1),
        })

    logger.info(
        f"\nStage 2 complete: {n_done} scenes processed in {elapsed:.1f}s"
    )

    # ── Run final drift check ──────────────────────────────────────────────────
    from scripts.drift_monitor import DriftMonitor
    monitor = DriftMonitor(
        baselines_path=data_dir / "processed" / "eda_baselines.json",
        features_dir=data_dir / "processed" / "features",
    )
    report = monitor.check(
        report_path=data_dir / "processed" / "drift_report.json"
    )
    if report.status != "ok":
        logger.warning(f"Drift detected: {report.status}. Check drift_report.json")
        for alert in report.alerts:
            logger.warning(f"  {alert.message}")


if __name__ == "__main__":
    main()
