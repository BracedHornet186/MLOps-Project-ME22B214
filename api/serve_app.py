"""
api/serve_app.py
────────────────────────────────────────────────────────────────────────────
Unified Ray Serve deployment graph.

Replaces:
  - api/main.py          (CPU FastAPI gateway)
  - model_server/server.py  (GPU inference server)

Two deployments:
  GPUModelWorker   — num_replicas=1, owns 1 GPU exclusively
                     Loads MASt3R pipeline once at startup.
                     Exposes reconstruct() + ping() methods via Ray RPC.

  APIGateway       — num_replicas=2, CPU-only
                     Wraps FastAPI via @serve.ingress.
                     Handles uploads, job management, Prometheus metrics,
                     drift endpoints.  All inference delegated to GPU worker
                     over Ray object store (zero HTTP overhead).

Binding:
  gpu_node = GPUModelWorker.bind()
  api_node = APIGateway.bind(gpu_node)

Start with:
  serve run api.serve_app:api_node --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
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
import uuid
import zipfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from ray import serve

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("serve_app")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_cfg_primary  = os.environ.get("PIPELINE_CONFIG",          str(ROOT / "conf/best_config.yaml"))
_cfg_fallback = os.environ.get("PIPELINE_CONFIG_FALLBACK", str(ROOT / "conf/mast3r.yaml"))
DEFAULT_CONFIG_PATH = _cfg_primary if Path(_cfg_primary).exists() else _cfg_fallback

API_VERSION           = "2.0.0"
MAX_CONCURRENT_JOBS   = 1
MAX_UPLOAD_SIZE_MB    = int(os.environ.get("SCENE3D_MAX_UPLOAD_MB", "500"))
VOXEL_SIZE            = float(os.environ.get("SCENE3D_VOXEL_SIZE", "0.02"))
MAX_POINT_COUNT       = int(os.environ.get("SCENE3D_MAX_POINTS", "500000"))
RESULTS_DIR           = Path(os.environ.get("RESULTS_DIR", "/tmp/reconstruction_results"))
AIRFLOW_API_URL       = os.environ.get("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
ALLOWED_IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics  (module-level so both deployments share the same registry)
# ─────────────────────────────────────────────────────────────────────────────

api_requests_total       = Counter("api_requests_total",       "Total HTTP requests",              ["method", "endpoint", "status"])
api_errors_total         = Counter("api_errors_total",         "Total 4xx/5xx responses",          ["endpoint"])
inference_latency_seconds= Histogram("inference_latency_seconds","End-to-end inference wall-clock", buckets=[10,30,60,120,180,300,600,900])
reconstruction_maa       = Gauge("reconstruction_maa",         "mAA score of most recent job")
registration_rate_gauge  = Gauge("registered_images_ratio",    "Fraction of images placed")
active_jobs_gauge        = Gauge("active_jobs_total",          "Running jobs")
model_ready_gauge        = Gauge("model_server_ready",         "1 if GPU worker ready")
data_valid_images_gauge  = Gauge("data_valid_images_total",    "Valid images in current dataset")

# ─────────────────────────────────────────────────────────────────────────────
# Job store
# ─────────────────────────────────────────────────────────────────────────────

class JobStage(str, Enum):
    QUEUED       = "queued"
    EXTRACTING   = "extracting"
    MATCHING     = "matching"
    TRIANGULATING= "triangulating"
    DECIMATING   = "decimating"
    DONE         = "success"
    FAILED       = "failed"

_STAGE_PROGRESS = {
    JobStage.QUEUED:        0,
    JobStage.EXTRACTING:   10,
    JobStage.MATCHING:     30,
    JobStage.TRIANGULATING:70,
    JobStage.DECIMATING:   85,
    JobStage.DONE:        100,
    JobStage.FAILED:        0,
}

class JobRecord(BaseModel):
    job_id:       str
    stage:        JobStage = JobStage.QUEUED
    progress:     int      = 0
    message:      str      = "Waiting in queue …"
    created_at:   float    = 0.0
    started_at:   Optional[float] = None
    finished_at:  Optional[float] = None
    n_images:     int      = 0
    n_points:     int      = 0
    result_path:  Optional[str]   = None
    ply_path:     Optional[str]   = None
    error:        Optional[str]   = None
    registration_rate: Optional[float] = None

class JobManager:
    def __init__(self, max_concurrent: int):
        self.jobs: dict[str, JobRecord] = {}
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.max_concurrent = max_concurrent

    def init_semaphore(self):
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def create_job(self, job_id: str) -> JobRecord:
        rec = JobRecord(job_id=job_id, created_at=time.time())
        self.jobs[job_id] = rec
        return rec

    def get_job(self, job_id: str) -> JobRecord:
        return self.jobs[job_id]

    def update_job(self, job_id: str, stage: JobStage, message: str, **extra):
        rec = self.jobs[job_id]
        rec.stage    = stage
        rec.progress = _STAGE_PROGRESS.get(stage, 0)
        rec.message  = message
        for k, v in extra.items():
            setattr(rec, k, v)

# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: float

class JobStatusResponse(BaseModel):
    job_id:            str
    stage:             str
    status:            str          # legacy alias for stage
    progress:          int
    message:           str
    created_at:        float
    started_at:        Optional[float]
    finished_at:       Optional[float]
    n_images:          int
    n_points:          int
    registration_rate: Optional[float]
    error:             Optional[str]
    download_url:      Optional[str]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared between deployments
# ─────────────────────────────────────────────────────────────────────────────

def _extract_zip_images(content: bytes, destination_dir: Path) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTS:
                continue
            file_name = Path(info.filename).name
            if not file_name:
                continue
            target = destination_dir / file_name
            base, suffix = target.stem, target.suffix
            idx = 1
            while target.exists():
                target = destination_dir / f"{base}_{idx}{suffix}"
                idx += 1
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    if extracted == 0:
        raise ValueError("ZIP contains no supported image files")
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# GPU Worker Deployment
# ─────────────────────────────────────────────────────────────────────────────

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
)
class GPUModelWorker:
    """
    Holds the MASt3R pipeline permanently in VRAM.
    Loads once at startup; never reloads between requests.
    No MLflow logging — job state lives in the API gateway's JobManager.
    """

    def __init__(self):
        log.info("GPUModelWorker: loading pipeline from %s", DEFAULT_CONFIG_PATH)
        t0 = time.perf_counter()

        from scripts.config import PipelineConfig
        from scripts.pipeline import create_pipeline
        from scripts.distributed import DistConfig

        pipeline_conf = PipelineConfig.load_config(DEFAULT_CONFIG_PATH)
        self.device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("GPUModelWorker: device = %s", self.device)

        self.pipeline = create_pipeline(
            conf=pipeline_conf,
            dist_conf=DistConfig.single(),
            device=self.device,
        )
        self._is_ready = True
        log.info("GPUModelWorker: pipeline loaded in %.1fs", time.perf_counter() - t0)

    # ── Liveness / readiness ──────────────────────────────────────────────────

    def ping(self) -> dict:
        """Lightweight probe called by /ready — never touches the GPU."""
        device_str = ""
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            device_str = f"{name} ({mem:.1f} GB)"
        else:
            device_str = "cpu"
        return {"ready": self._is_ready, "device": device_str}

    # ── Main inference entry point ────────────────────────────────────────────

    def reconstruct(
        self,
        job_id:          str,
        upload_content:  bytes,
        upload_filename: str,
        dataset_name:    str = "custom",
        scene_name:      str = "scene_01",
    ) -> dict[str, Any]:
        """
        Full reconstruction pipeline, mirroring reconstruct_scenes.py:
          1. Extract ZIP → temp dir
          2. Build DataSchema pointing at temp dir
          3. Run pipeline.run()
          4. Compute registration rate
          5. Persist CSV + PLY to RESULTS_DIR
          6. Return result dict (no MLflow)
        """
        import pandas as pd
        from scripts.data_schema import DataSchema

        t_start = time.perf_counter()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory(prefix=f"job_{job_id[:8]}_") as _tmpdir:
                tmpdir    = Path(_tmpdir)
                image_dir = tmpdir / "images"
                image_dir.mkdir()

                # ── 1. Extract images ─────────────────────────────────────────
                with zipfile.ZipFile(io.BytesIO(upload_content)) as zf:
                    image_names = [
                        n for n in zf.namelist()
                        if Path(n).suffix.lower() in
                        {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
                    ]
                    if not image_names:
                        raise ValueError("ZIP contains no supported image files")
                    for name in image_names:
                        dest = image_dir / Path(name).name
                        dest.write_bytes(zf.read(name))

                n_images = len(image_names)
                log.info("Job %s: extracted %d images", job_id[:8], n_images)

                # ── 2. Build submission DataFrame ─────────────────────────────
                rows = []
                for img_path in sorted(image_dir.glob("*")):
                    rows.append({
                        "image_id":           f"{job_id[:8]}_{img_path.stem}",
                        "dataset":            dataset_name,
                        "scene":              "cluster0",
                        "image":              f"{dataset_name}/{img_path.name}",
                        "rotation_matrix":    ";".join(["nan"] * 9),
                        "translation_vector": ";".join(["nan"] * 3),
                    })
                submission_input_df = pd.DataFrame(rows)

                # Copy images into expected path structure:
                #   tmpdir/test/<dataset_name>/<image>
                # DataSchema.build_image_relative_path returns test/<dataset>/<image>
                # so data_root_dir = tmpdir
                scene_img_dir = tmpdir / "test" / dataset_name
                scene_img_dir.mkdir(parents=True)
                for img_path in sorted(image_dir.glob("*")):
                    shutil.copy(img_path, scene_img_dir / img_path.name)

                # ── 3. Build TmpDataSchema ────────────────────────────────────
                class TmpDataSchema(DataSchema):
                    columns = (
                        "image_id", "dataset", "scene", "image",
                        "rotation_matrix", "translation_vector",
                    )

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

                # ── 4. Run pipeline ───────────────────────────────────────────
                log.info("Job %s: starting pipeline …", job_id[:8])
                submission_df = self.pipeline.run(
                    df=submission_input_df,
                    data_schema=data_schema,
                    save_snapshot=False,
                )

                # ── 5. Compute registration rate ──────────────────────────────
                def _is_valid_R(r_str: str) -> bool:
                    try:
                        vals = [float(x) for x in str(r_str).split(";")]
                        return len(vals) == 9 and not any(v != v for v in vals)
                    except Exception:
                        return False

                registered = int(submission_df["rotation_matrix"].apply(_is_valid_R).sum())
                reg_rate   = float(registered / max(len(submission_df), 1))
                elapsed    = time.perf_counter() - t_start

                # ── 6. Persist CSV ────────────────────────────────────────────
                persist_csv = RESULTS_DIR / f"submission_{job_id[:8]}.csv"
                submission_df.to_csv(persist_csv, index=False)

                # ── 7. Export PLY (COLMAP sparse reconstruction) ──────────────
                persist_ply: Optional[Path] = None
                try:
                    import pycolmap
                    max_pts      = -1
                    best_rec_dir = None
                    for pts_file in tmpdir.rglob("points3D.bin"):
                        try:
                            rec = pycolmap.Reconstruction(str(pts_file.parent))
                            if len(rec.points3D) > max_pts:
                                max_pts      = len(rec.points3D)
                                best_rec_dir = pts_file.parent
                        except Exception:
                            pass
                    if best_rec_dir is not None:
                        persist_ply = RESULTS_DIR / f"sparse_{job_id[:8]}.ply"
                        pycolmap.Reconstruction(str(best_rec_dir)).export_PLY(str(persist_ply))
                        log.info(
                            "Job %s: exported PLY (%d 3D points) → %s",
                            job_id[:8], max_pts, persist_ply,
                        )
                    else:
                        log.info("Job %s: no COLMAP reconstruction found for PLY export", job_id[:8])
                except Exception as e:
                    log.warning("Job %s: PLY export failed: %s", job_id[:8], e)

                log.info(
                    "Job %s done — registered=%d/%d (%.1f%%)  elapsed=%.1fs",
                    job_id[:8], registered, n_images, 100 * reg_rate, elapsed,
                )

                return {
                    "job_id":                  job_id,
                    "n_images":                n_images,
                    "registered":              registered,
                    "registration_rate":       round(reg_rate, 4),
                    "inference_latency_seconds": round(elapsed, 2),
                    "result_csv_path":         str(persist_csv),
                    "raw_ply_path":            str(persist_ply) if persist_ply and persist_ply.exists() else None,
                }

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (shared across API gateway replicas)
# ─────────────────────────────────────────────────────────────────────────────

fastapi_app = FastAPI(title="Scene Reconstruction API", version=API_VERSION)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# API Gateway Deployment
# ─────────────────────────────────────────────────────────────────────────────

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 0, "num_cpus": 2},
)
@serve.ingress(fastapi_app)
class APIGateway:

    def __init__(self, gpu_worker_handle):
        self.gpu_worker  = gpu_worker_handle
        self.job_manager = JobManager(MAX_CONCURRENT_JOBS)
        self.job_manager.init_semaphore()
        self._refresh_data_metrics()

    # ── Infrastructure ────────────────────────────────────────────────────────

    @fastapi_app.get("/health", response_model=HealthResponse, tags=["infra"])
    async def health(self):
        api_requests_total.labels("GET", "/health", "200").inc()
        return HealthResponse(status="ok", version=API_VERSION, timestamp=time.time())

    @fastapi_app.get("/ready", tags=["infra"])
    async def ready(self):
        try:
            result = await self.gpu_worker.ping.remote()
            if result.get("ready"):
                model_ready_gauge.set(1)
                api_requests_total.labels("GET", "/ready", "200").inc()
                return {"status": "ready", "device": result.get("device", "unknown")}
        except Exception as e:
            model_ready_gauge.set(0)
            api_errors_total.labels("/ready").inc()
            raise HTTPException(status_code=503, detail=f"GPU worker not ready: {e}")
        model_ready_gauge.set(0)
        api_errors_total.labels("/ready").inc()
        raise HTTPException(status_code=503, detail="GPU worker not ready")

    @fastapi_app.get("/metrics", tags=["infra"])
    async def metrics(self):
        return StreamingResponse(
            io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST
        )

    # ── Drift monitoring ─────────────────────────────

    @fastapi_app.get("/drift", tags=["monitoring"])
    async def drift_check(self):
        from scripts.drift_monitor import DriftMonitor, update_prometheus_drift_metrics
        data_dir = Path(os.environ.get("DEFAULT_DATASET_DIR", "data"))
        monitor  = DriftMonitor(
            baselines_path=data_dir / "processed" / "eda_baselines.json",
            features_dir=data_dir / "processed" / "features",
        )
        report = monitor.check(
            report_path=data_dir / "processed" / "drift_report.json",
            check_performance=True,
        )
        update_prometheus_drift_metrics(report)
        api_requests_total.labels("GET", "/drift", "200").inc()
        return report.as_dict()

    @fastapi_app.post("/drift/trigger-retrain", tags=["monitoring"])
    async def trigger_retrain(self):
        import datetime
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"{AIRFLOW_API_URL}/api/v2/dags/experiment_pipeline_dag/dagRuns",
                    json={
                        "logical_date": datetime.datetime.utcnow().isoformat() + "Z",
                        "conf": {"triggered_by": "api"},
                    },
                    headers={"Content-Type": "application/json"},
                )
                return {"status": "triggered"} if r.status_code in (200, 201) else {"status": "error"}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ── Async upload (primary UI endpoint) ───────────────────────────────────

    @fastapi_app.post("/upload", tags=["inference"])
    async def upload_zip(
        self,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="ZIP archive of images"),
        dataset_name: str = "custom",
        scene_name:   str = "scene_01",
    ):
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_SIZE_MB} MB")
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise HTTPException(status_code=400, detail="Invalid ZIP")

        job_id = str(uuid.uuid4())
        self.job_manager.create_job(job_id)

        background_tasks.add_task(
            self._background_reconstruction,
            job_id=job_id,
            content=content,
            filename=file.filename or "images.zip",
            dataset_name=dataset_name,
            scene_name=scene_name,
        )
        api_requests_total.labels("POST", "/upload", "202").inc()
        return {"job_id": job_id, "message": "Pipeline started."}

    # ── Status / download ─────────────────────────────────────────────────────

    @fastapi_app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["inference"])
    @fastapi_app.get("/jobs/{job_id}",   response_model=JobStatusResponse, tags=["inference"])
    async def get_status(self, job_id: str):
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        rec          = self.job_manager.get_job(job_id)
        download_url = f"/download/{job_id}" if rec.stage == JobStage.DONE and rec.ply_path else None
        return JobStatusResponse(
            job_id=rec.job_id,
            stage=rec.stage.value,
            status=rec.stage.value,
            progress=rec.progress,
            message=rec.message,
            created_at=rec.created_at,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            n_images=rec.n_images,
            n_points=rec.n_points,
            registration_rate=rec.registration_rate,
            error=rec.error,
            download_url=download_url,
        )

    @fastapi_app.get("/download/{job_id}", tags=["inference"])
    async def download_ply(self, job_id: str):
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404)
        rec = self.job_manager.get_job(job_id)
        if rec.stage != JobStage.DONE:
            raise HTTPException(status_code=409, detail="Not done yet")
        if not rec.ply_path or not Path(rec.ply_path).exists():
            raise HTTPException(status_code=404, detail="PLY not found")
        return FileResponse(rec.ply_path, media_type="application/octet-stream")

    @fastapi_app.get("/jobs/{job_id}/download", tags=["inference"])
    async def download_csv(self, job_id: str):
        """Download the raw submission CSV."""
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404)
        rec = self.job_manager.get_job(job_id)
        if rec.stage != JobStage.DONE:
            raise HTTPException(status_code=409, detail="Not done yet")
        if not rec.result_path or not Path(rec.result_path).exists():
            raise HTTPException(status_code=404, detail="CSV not found")
        return FileResponse(
            rec.result_path,
            filename=f"submission_{job_id[:8]}.csv",
            media_type="text/csv",
        )

    # ── Background reconstruction task ────────────────────────────────────────

    async def _background_reconstruction(
        self,
        job_id:       str,
        content:      bytes,
        filename:     str,
        dataset_name: str,
        scene_name:   str,
    ):
        rec           = self.job_manager.get_job(job_id)
        rec.started_at= time.time()
        active_jobs_gauge.inc()

        async with self.job_manager.semaphore:
            try:
                self.job_manager.update_job(
                    job_id, JobStage.EXTRACTING, "Extracting images from ZIP …"
                )

                self.job_manager.update_job(
                    job_id, JobStage.MATCHING,
                    "Running MASt3R feature matching on GPU …",
                )

                # Dispatch to GPU worker — zero-copy via Ray object store
                result: dict = await self.gpu_worker.reconstruct.remote(
                    job_id=job_id,
                    upload_content=content,
                    upload_filename=filename,
                    dataset_name=dataset_name,
                    scene_name=scene_name,
                )

                self.job_manager.update_job(
                    job_id, JobStage.TRIANGULATING,
                    "Inference complete — triangulating …",
                    n_images=result.get("n_images", 0),
                    registration_rate=result.get("registration_rate"),
                    result_path=result.get("result_csv_path"),
                )

                raw_ply = result.get("raw_ply_path")

                # CPU-side decimation ─────────────────────────────────────────
                if raw_ply and Path(raw_ply).exists():
                    self.job_manager.update_job(
                        job_id, JobStage.DECIMATING,
                        "Optimising 3D point cloud for browser …",
                    )
                    from utils.decimate import voxel_downsample_ply, get_point_cloud_stats

                    decimated_ply = str(
                        Path(raw_ply).parent / f"decimated_{job_id[:8]}.ply"
                    )
                    await asyncio.to_thread(
                        voxel_downsample_ply,
                        raw_ply, decimated_ply,
                        voxel_size=VOXEL_SIZE,
                        max_points=MAX_POINT_COUNT,
                    )
                    stats = get_point_cloud_stats(decimated_ply)
                    self.job_manager.update_job(
                        job_id, JobStage.DONE,
                        f"Reconstruction successful ({stats.n_points:,} points)",
                        ply_path=decimated_ply,
                        n_points=stats.n_points,
                    )
                else:
                    log.warning("Job %s: no PLY from GPU worker — skipping decimation", job_id[:8])
                    self.job_manager.update_job(
                        job_id, JobStage.DONE,
                        "Reconstruction successful (no point cloud available).",
                    )

                # Update Prometheus ────────────────────────────────────────────
                elapsed = time.time() - rec.started_at
                inference_latency_seconds.observe(elapsed)
                if rec.registration_rate is not None:
                    registration_rate_gauge.set(rec.registration_rate)

            except Exception as e:
                log.error("Job %s failed: %s", job_id, e, exc_info=True)
                self.job_manager.update_job(
                    job_id, JobStage.FAILED,
                    f"Pipeline error: {e}",
                    error=str(e),
                )
                api_errors_total.labels("/upload").inc()
            finally:
                rec.finished_at = time.time()
                active_jobs_gauge.dec()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _refresh_data_metrics(self):
        report_path = (
            Path(os.environ.get("DEFAULT_DATASET_DIR", "data"))
            / "processed"
            / "validation_report.json"
        )
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                data_valid_images_gauge.set(report.get("total_images", 0))
            except Exception as e:
                log.warning("Could not read validation_report.json: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Bind the deployment graph
# ─────────────────────────────────────────────────────────────────────────────

gpu_node = GPUModelWorker.bind()
api_node = APIGateway.bind(gpu_node)