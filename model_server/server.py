"""
model_server/server.py
────────────────────────────────────────────────────────────────────────────
Stage 4 — Model Server

Loads all model weights once at startup, then accepts inference requests
from the API server. This process holds the GPU exclusively.

Endpoints
---------
GET  /ready        Returns 200 once all weights are loaded
GET  /health       Liveness probe
POST /infer        Run scene reconstruction on a ZIP of images
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import mlflow
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Project source on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("model_server")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (from environment)
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DEFAULT_DATASET_DIR  = Path(os.environ.get("DEFAULT_DATASET_DIR", str(ROOT / "data")))
DEFAULT_CONFIG_PATH  = os.environ.get(
    "PIPELINE_CONFIG",
    str(ROOT / "conf/mast3r.yaml"),
)
SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "8001"))

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

_model_ready: bool = False
_inference_lock = threading.Lock()   # single GPU — one job at a time
_pipeline = None                      # loaded at startup

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Scene Reconstruction — Model Server", version="1.0.0")


@app.on_event("startup")
def startup():
    """Load all model weights on startup (runs synchronously before first request)."""
    _load_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/ready")
def ready():
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model weights still loading")
    return {"status": "ready", "device": _get_device_str()}


@app.post("/infer")
async def infer(
    images: UploadFile = File(..., description="ZIP archive of images"),
    job_id: str = Form(...),
    dataset_name: str = Form("custom"),
    scene_name: str = Form("scene_01"),
):
    """
    Main inference endpoint.
    Accepts a ZIP of images, runs the full reconstruction pipeline,
    returns metrics and the path to the resulting CSV.
    """
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not _inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another inference job is already running. Try again shortly.",
        )

    try:
        result = _run_inference(
            job_id=job_id,
            upload_content=await images.read(),
            upload_filename=images.filename or "images.zip",
            dataset_name=dataset_name,
            scene_name=scene_name,
        )
        return JSONResponse(content=result)
    except Exception as e:
        log.exception("Inference failed for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _inference_lock.release()
        _cleanup_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline() -> None:
    """Load the IMC2025MASt3RPipeline once. Sets _model_ready = True on success."""
    global _pipeline, _model_ready

    log.info("Loading pipeline from config: %s", DEFAULT_CONFIG_PATH)
    t0 = time.perf_counter()

    try:
        from scripts.config import PipelineConfig, SubmissionConfig
        from scripts.pipeline import create_pipeline
        from scripts.distributed import DistConfig

        pipeline_conf = PipelineConfig.load_config(DEFAULT_CONFIG_PATH)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", device)

        _pipeline = create_pipeline(
            conf=pipeline_conf,
            dist_conf=DistConfig.single(),
            device=device,
        )
        _model_ready = True
        elapsed = time.perf_counter() - t0
        log.info("Pipeline loaded in %.1fs", elapsed)

    except Exception as e:
        log.error("Failed to load pipeline: %s", e)
        _model_ready = False
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Inference execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(
    job_id: str,
    upload_content: bytes,
    upload_filename: str,
    dataset_name: str,
    scene_name: str,
) -> dict[str, Any]:
    """
    Extract images from ZIP, build a temporary DataSchema, run the pipeline,
    log results to MLflow, and return a result dict.
    """
    import pandas as pd
    from scripts.config import PipelineConfig, SubmissionConfig
    from scripts.data import IMC2025TestData
    from scripts.distributed import DistConfig

    t_start = time.perf_counter()

    # ── Extract ZIP to temp dir ────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix=f"job_{job_id[:8]}_") as tmpdir:
        tmpdir = Path(tmpdir)
        image_dir = tmpdir / "images"
        image_dir.mkdir()

        with zipfile.ZipFile(io.BytesIO(upload_content)) as zf:
            image_names = [
                n for n in zf.namelist()
                if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]
            if not image_names:
                raise ValueError("ZIP contains no supported image files (.jpg/.jpeg/.png/.bmp/.tiff)")

            for name in image_names:
                dest = image_dir / Path(name).name
                dest.write_bytes(zf.read(name))

        n_images = len(image_names)
        log.info("Job %s: extracted %d images", job_id[:8], n_images)

        # ── Build a minimal submission DataFrame ───────────────────────────
        rows = []
        for img_path in sorted(image_dir.glob("*")):
            rows.append({
                "image_id": f"{job_id[:8]}_{img_path.stem}",
                "dataset": dataset_name,
                "scene": "cluster0",       # iterate_scenes resets to cluster0
                "image": f"{dataset_name}/{img_path.name}",
                "rotation_matrix": ";".join(["nan"] * 9),
                "translation_vector": ";".join(["nan"] * 3),
            })
        submission_input_df = pd.DataFrame(rows)

        # Copy images into the expected path structure
        scene_img_dir = tmpdir / "test" / dataset_name
        scene_img_dir.mkdir(parents=True)
        for img_path in sorted(image_dir.glob("*")):
            shutil.copy(img_path, scene_img_dir / img_path.name)

        # ── Build DataSchema pointing at tmpdir ────────────────────────────
        from scripts.data_schema import DataSchema

        class TmpDataSchema(DataSchema):
            columns = ("image_id", "dataset", "scene", "image",
                       "rotation_matrix", "translation_vector")

            def format_output_key(self, dataset, scene, name):
                return name

            def build_image_relative_path(self, row):
                img_name = Path(row["image"]).name
                return f"test/{row['dataset']}/{img_name}"

            def get_output_metadata(self, dataset, scene, name):
                df_row = self.df[self.df["image"].str.endswith(name)]
                if len(df_row) > 0:
                    return {"image_id": str(df_row.iloc[0]["image_id"])}
                return {"image_id": name}

        data_schema = TmpDataSchema(
            df=submission_input_df,
            data_root_dir=tmpdir,
        )

        # ── MLflow run ────────────────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("scene_reconstruction_inference")

        with mlflow.start_run(run_name=f"infer_{job_id[:8]}") as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_params({
                "job_id": job_id[:8],
                "dataset_name": dataset_name,
                "scene_name": scene_name,
                "n_images": n_images,
                "config_file": Path(DEFAULT_CONFIG_PATH).name,
            })

            # ── Run pipeline ───────────────────────────────────────────────
            log.info("Job %s: starting pipeline...", job_id[:8])
            submission_df = _pipeline.run(
                df=submission_input_df,
                data_schema=data_schema,
                save_snapshot=False,
            )

            # ── Compute registration rate ──────────────────────────────────
            def is_valid_R(r_str):
                try:
                    vals = [float(x) for x in str(r_str).split(";")]
                    return len(vals) == 9 and not any(v != v for v in vals)
                except Exception:
                    return False

            registered = submission_df["rotation_matrix"].apply(is_valid_R).sum()
            reg_rate = float(registered / max(len(submission_df), 1))

            elapsed = time.perf_counter() - t_start

            # Save result CSV
            result_csv = tmpdir / f"submission_{job_id[:8]}.csv"
            submission_df.to_csv(result_csv, index=False)

            # Persist result outside tmpdir so it survives cleanup
            persist_dir = Path(os.environ.get("RESULTS_DIR", "/tmp/reconstruction_results"))
            persist_dir.mkdir(parents=True, exist_ok=True)
            persist_csv = persist_dir / f"submission_{job_id[:8]}.csv"
            shutil.copy(result_csv, persist_csv)

            mlflow.log_metrics({
                "n_images": n_images,
                "registration_rate": round(reg_rate, 4),
                "inference_latency_seconds": round(elapsed, 2),
                "registered_images": int(registered),
            })
            mlflow.log_artifact(str(persist_csv), artifact_path="predictions")

            log.info(
                "Job %s done — registered=%d/%d (%.1f%%)  elapsed=%.1fs  run_id=%s",
                job_id[:8], int(registered), n_images, 100 * reg_rate, elapsed, mlflow_run_id,
            )

    return {
        "job_id": job_id,
        "n_images": n_images,
        "registration_rate": round(reg_rate, 4),
        "maa": None,               # no ground truth at inference time
        "mlflow_run_id": mlflow_run_id,
        "inference_latency_seconds": round(elapsed, 2),
        "result_csv_path": str(persist_csv),
    }


def _cleanup_gpu() -> None:
    """Free CUDA cache after each job — important on 12 GB RTX 3060."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _get_device_str() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{name} ({mem:.1f} GB)"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=SERVER_PORT,
        workers=1,            # must be 1 — we own the GPU exclusively
        log_level="info",
    )
