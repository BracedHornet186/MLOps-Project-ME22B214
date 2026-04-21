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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: preprocess images and extract deterministic placeholder features")
    parser.add_argument("--config", default="conf/mast3r.yaml", help="Path to unified pipeline config")
    parser.add_argument("--train-dir", default="data/train", help="Directory containing train images")
    parser.add_argument("--preprocessed-dir", default="data/processed/images", help="Directory containing preprocessed images")
    parser.add_argument("--extracted-dir", default="data/extracted", help="Output extracted metadata directory")
    parser.add_argument("--features-dir", default="data/features", help="Output features directory")
    parser.add_argument("--feature-dim", type=int, default=None, help="Feature vector dimensionality")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed metadata")
    return parser.parse_args()


def resolve_feature_dim(imc2025_conf) -> int:
    # Placeholder feature vectors are kept bounded for lightweight DVC stage execution.
    for matcher in imc2025_conf.point_tracking_matchers:
        hybrid = matcher.mast3r_hybrid
        if hybrid and hybrid.size:
            return max(16, min(int(hybrid.size), 128))
    return 16


def resolve_seed(imc2025_conf) -> int:
    return int(getattr(imc2025_conf, "seed", 42) or 42)


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files


def file_digest(path: Path) -> bytes:
    hasher = hashlib.sha256()
    stat = path.stat()
    hasher.update(str(path).encode("utf-8"))
    hasher.update(str(stat.st_size).encode("utf-8"))
    with open(path, "rb") as f:
        hasher.update(f.read(4096))
    return hasher.digest()


def vector_from_digest(digest: bytes, dim: int) -> np.ndarray:
    repeated = (digest * ((dim // len(digest)) + 1))[:dim]
    arr = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
    arr = arr / 255.0
    return arr


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_to_data_root(path: Path, data_root_name: str = "data") -> Path:
    if path.is_absolute() and data_root_name in path.parts:
        idx = path.parts.index(data_root_name)
        return Path(*path.parts[idx + 1 :])
    if path.parts and path.parts[0] == data_root_name:
        return Path(*path.parts[1:])
    return path


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    feature_dim = int(args.feature_dim) if args.feature_dim is not None else resolve_feature_dim(imc2025_conf)
    seed = int(args.seed) if args.seed is not None else resolve_seed(imc2025_conf)

    train_dir = Path(args.train_dir)
    preprocessed_dir = Path(args.preprocessed_dir)
    extracted_dir = Path(args.extracted_dir)
    features_dir = Path(args.features_dir)
    vectors_dir = features_dir / "vectors"

    ensure_dir(extracted_dir)
    ensure_dir(vectors_dir)

    raw_image_paths = list_images(train_dir)

    extracted_records: list[dict] = []
    feature_records: list[dict] = []
    processed_inputs = 0

    for raw_path in raw_image_paths:
        rel = relative_to_data_root(raw_path)
        processed_candidate = preprocessed_dir / rel

        if processed_candidate.exists():
            source_path = processed_candidate
            preprocess_status = "processed"
            processed_inputs += 1
        else:
            source_path = raw_path
            preprocess_status = "raw_fallback"

        digest = file_digest(source_path)
        image_id = hashlib.sha1(str(raw_path).encode("utf-8")).hexdigest()
        vec = vector_from_digest(digest, feature_dim)

        vector_path = vectors_dir / f"{image_id}.npy"
        np.save(vector_path, vec)

        extracted_records.append(
            {
                "image_id": image_id,
                "source_path": str(source_path),
                "raw_path": str(raw_path),
                "relative_path": str(rel),
                "size_bytes": source_path.stat().st_size,
                "preprocess_status": preprocess_status,
            }
        )
        feature_records.append(
            {
                "image_id": image_id,
                "vector_path": str(vector_path),
                "feature_dim": feature_dim,
            }
        )

    extracted_index = {
        "num_images": len(extracted_records),
        "seed": seed,
        "num_processed_inputs": processed_inputs,
        "records": extracted_records,
    }
    features_index = {
        "num_images": len(feature_records),
        "feature_dim": feature_dim,
        "records": feature_records,
    }

    extracted_index_path = extracted_dir / "extracted_index.json"
    features_index_path = features_dir / "features_index.json"

    with open(extracted_index_path, "w") as f:
        json.dump(extracted_index, f, indent=2)
    with open(features_index_path, "w") as f:
        json.dump(features_index, f, indent=2)

    duration = time.perf_counter() - t0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)        
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="extract_features", nested=True, tags=run_tags):
        mlflow.log_param("train_dir", str(train_dir))
        mlflow.log_param("preprocessed_dir", str(preprocessed_dir))
        mlflow.log_param("feature_dim", feature_dim)
        mlflow.log_param("seed", seed)
        mlflow.log_param("config_path", args.config)

        mlflow.log_metric("num_images", len(raw_image_paths))
        mlflow.log_metric("num_processed_inputs", processed_inputs)
        mlflow.log_metric("feature_dim", feature_dim)
        mlflow.log_metric("execution_time", round(duration, 4))

        mlflow.log_artifacts(str(extracted_dir), artifact_path="extracted")
        mlflow.log_artifacts(str(features_dir), artifact_path="features")

    print(
        json.dumps(
            {
                "num_images": len(raw_image_paths),
                "num_processed_inputs": processed_inputs,
                "feature_dim": feature_dim,
                "extracted_index": str(extracted_index_path),
                "features_index": str(features_index_path),
                "execution_time": round(duration, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
