from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: deterministic keypoint detection for shortlisted images")
    parser.add_argument("--config", default="conf/mast3r.yaml", help="Path to unified pipeline config")
    parser.add_argument("--extracted-dir", default="data/extracted", help="Input extracted directory")
    parser.add_argument("--pairs-dir", default="data/pairs", help="Input pairs directory")
    parser.add_argument("--keypoints-dir", default="data/keypoints", help="Output keypoints directory")
    parser.add_argument("--min-keypoints", type=int, default=None, help="Minimum keypoints per image")
    parser.add_argument("--max-keypoints", type=int, default=None, help="Maximum keypoints per image")
    return parser.parse_args()


def resolve_min_keypoints(imc2025_conf) -> int:
    candidates: list[int] = []
    for matcher in imc2025_conf.point_tracking_matchers:
        hybrid = matcher.mast3r_hybrid
        if not hybrid:
            continue
        if hybrid.sparse_min_matches is not None:
            candidates.append(int(hybrid.sparse_min_matches))
        if hybrid.dense_min_matches is not None:
            candidates.append(int(hybrid.dense_min_matches))
    if candidates:
        return max(8, min(candidates))
    return 32


def resolve_max_keypoints(min_keypoints: int) -> int:
    return max(min_keypoints + 1, min_keypoints * 8)


def deterministic_keypoints(image_id: str, min_kpts: int, max_kpts: int) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    span = max(max_kpts - min_kpts, 1)
    n = min_kpts + (digest[0] % span)

    points = []
    scores = []
    for i in range(n):
        d = digest[i % len(digest)]
        x = float((d + i * 17) % 1000) / 10.0
        y = float((d + i * 29) % 1000) / 10.0
        s = float((d % 100) / 100.0)
        points.append([x, y])
        scores.append(s)

    return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    min_keypoints = int(args.min_keypoints) if args.min_keypoints is not None else resolve_min_keypoints(imc2025_conf)
    max_keypoints = int(args.max_keypoints) if args.max_keypoints is not None else resolve_max_keypoints(min_keypoints)
    if max_keypoints <= min_keypoints:
        max_keypoints = min_keypoints + 1

    extracted_dir = Path(args.extracted_dir)
    pairs_dir = Path(args.pairs_dir)
    keypoints_dir = Path(args.keypoints_dir)
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    extracted_index_path = extracted_dir / "extracted_index.json"
    pairs_index_path = pairs_dir / "pairs_index.json"

    if not extracted_index_path.exists():
        raise FileNotFoundError(f"Missing extracted index: {extracted_index_path}")
    if not pairs_index_path.exists():
        raise FileNotFoundError(f"Missing pairs index: {pairs_index_path}")

    with open(extracted_index_path) as f:
        extracted_index = json.load(f)
    with open(pairs_index_path) as f:
        pairs_index = json.load(f)

    valid_ids = {r["image_id"] for r in extracted_index.get("records", [])}

    needed_ids: set[str] = set()
    for pair in pairs_index.get("pairs", []):
        needed_ids.add(pair["img1"])
        needed_ids.add(pair["img2"])
    needed_ids = {img_id for img_id in needed_ids if img_id in valid_ids}

    keypoint_records: list[dict] = []
    total_keypoints = 0

    for image_id in sorted(needed_ids):
        points, scores = deterministic_keypoints(image_id, min_keypoints, max_keypoints)
        total_keypoints += int(points.shape[0])

        out_path = keypoints_dir / f"{image_id}.npz"
        np.savez_compressed(out_path, keypoints=points, scores=scores)
        keypoint_records.append(
            {
                "image_id": image_id,
                "keypoints_path": str(out_path),
                "num_keypoints": int(points.shape[0]),
            }
        )

    keypoints_index = {
        "num_images": len(keypoint_records),
        "total_keypoints": total_keypoints,
        "records": keypoint_records,
    }

    keypoints_index_path = keypoints_dir / "keypoints_index.json"
    with open(keypoints_index_path, "w") as f:
        json.dump(keypoints_index, f, indent=2)

    avg_keypoints = 0.0
    if keypoint_records:
        avg_keypoints = total_keypoints / len(keypoint_records)

    duration = time.perf_counter() - t0

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="detect_keypoints", nested=True, tags=run_tags):
        mlflow.log_param("min_keypoints", min_keypoints)
        mlflow.log_param("max_keypoints", max_keypoints)
        mlflow.log_param("config_path", args.config)

        mlflow.log_metric("num_images", len(keypoint_records))
        mlflow.log_metric("total_keypoints", total_keypoints)
        mlflow.log_metric("avg_keypoints", round(avg_keypoints, 4))
        mlflow.log_metric("execution_time", round(duration, 4))

        mlflow.log_artifacts(str(keypoints_dir), artifact_path="keypoints")

    print(
        json.dumps(
            {
                "num_images": len(keypoint_records),
                "total_keypoints": total_keypoints,
                "keypoints_index": str(keypoints_index_path),
                "execution_time": round(duration, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
