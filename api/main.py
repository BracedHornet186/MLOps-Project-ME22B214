from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model-server:8001")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
API_VERSION = "1.0.0"
MAX_CONCURRENT_JOBS = 1
MAX_UPLOAD_SIZE_MB = int(os.environ.get("SCENE3D_MAX_UPLOAD_MB", "500"))

# Scene3D decimation config
VOXEL_SIZE = float(os.environ.get("SCENE3D_VOXEL_SIZE", "0.02"))
MAX_POINT_COUNT = int(os.environ.get("SCENE3D_MAX_POINTS", "500000"))
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

api_requests_total = Counter("api_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
api_errors_total = Counter("api_errors_total", "Total 4xx/5xx responses", ["endpoint"])
inference_latency_seconds = Histogram("inference_latency_seconds", "End-to-end inference wall-clock time", buckets=[10, 30, 60, 120, 180, 300, 600, 900])
reconstruction_maa = Gauge("reconstruction_maa", "mAA score from the most recent completed job")
registration_rate_gauge = Gauge("registered_images_ratio", "Fraction of images successfully placed")
active_jobs_gauge = Gauge("active_jobs_total", "Number of running jobs")
model_ready_gauge = Gauge("model_server_ready", "1 if model server is ready, 0 otherwise")
data_valid_images_gauge = Gauge("data_valid_images_total", "Valid images in current dataset version")

# ─────────────────────────────────────────────────────────────────────────────
# Job Store (merging legacy and new UI)
# ─────────────────────────────────────────────────────────────────────────────

class JobStage(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    MATCHING = "matching"         # When model-server is crunching
    TRIANGULATING = "triangulating"
    DECIMATING = "decimating"     # Local voxel grid
    DONE = "success"              # Renamed to success to match legacy JobStatus.SUCCESS
    FAILED = "failed"

_STAGE_PROGRESS = {
    JobStage.QUEUED: 0,
    JobStage.EXTRACTING: 10,
    JobStage.MATCHING: 30,
    JobStage.TRIANGULATING: 70,
    JobStage.DECIMATING: 85,
    JobStage.DONE: 100,
    JobStage.FAILED: 0,
}

class JobRecord(BaseModel):
    job_id: str
    stage: JobStage = JobStage.QUEUED
    progress: int = 0
    message: str = "Waiting in queue …"
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    
    n_images: int = 0
    n_points: int = 0
    
    # Legacy fields
    result_path: Optional[str] = None
    ply_path: Optional[str] = None
    error: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    maa: Optional[float] = None
    registration_rate: Optional[float] = None

class JobManager:
    def __init__(self, max_concurrent: int):
        self.jobs: dict[str, JobRecord] = {}
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.max_concurrent = max_concurrent

    def init_semaphore(self):
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def get_job(self, job_id: str) -> JobRecord:
        return self.jobs[job_id]

    def create_job(self, job_id: str) -> JobRecord:
        record = JobRecord(job_id=job_id, created_at=time.time())
        self.jobs[job_id] = record
        return record

    def update_job(self, job_id: str, stage: JobStage, message: str, **extra):
        rec = self.jobs[job_id]
        rec.stage = stage
        rec.progress = _STAGE_PROGRESS.get(stage, 0)
        rec.message = message
        for k, v in extra.items():
            setattr(rec, k, v)

job_manager = JobManager(MAX_CONCURRENT_JOBS)
# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    job_manager.init_semaphore()
    _refresh_data_metrics()
    asyncio.create_task(_poll_model_server_ready())
    yield

def _refresh_data_metrics():
    report_path = Path(os.environ.get("DEFAULT_DATASET_DIR", "data")) / "processed" / "validation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            data_valid_images_gauge.set(report.get("total_images", 0))
        except Exception as e:
            log.warning("Could not read validation_report.json: %s", e)

async def _poll_model_server_ready():
    for attempt in range(30):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{MODEL_SERVER_URL}/ready")
                if r.status_code == 200:
                    model_ready_gauge.set(1)
                    return
        except Exception:
            pass
        await asyncio.sleep(10)
    model_ready_gauge.set(0)

# ─────────────────────────────────────────────────────────────────────────────
# App & Schemas
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Scene Reconstruction API (Unified)", version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float

class ReconstructResponse(BaseModel):
    job_id: str
    status: str
    n_images: int
    maa: Optional[float]
    registration_rate: Optional[float]
    mlflow_run_id: Optional[str]
    latency_seconds: Optional[float]

class JobStatusResponse(BaseModel):
    job_id: str
    stage: str
    progress: int
    message: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    n_images: int
    n_points: int
    maa: Optional[float]
    registration_rate: Optional[float]
    mlflow_run_id: Optional[str]
    error: Optional[str]
    download_url: Optional[str]
    # For legacy backwards compat
    status: str

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints - Legacy Infra
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health():
    api_requests_total.labels("GET", "/health", "200").inc()
    return HealthResponse(status="ok", version=API_VERSION, timestamp=time.time())

@app.get("/ready", tags=["infra"])
async def ready():
    is_ready = False
    status = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{MODEL_SERVER_URL}/ready")
            is_ready = r.status_code == 200
            status = "ready" if is_ready else "loading"
    except Exception as e:
        status = f"error: {e}"

    model_ready_gauge.set(1 if is_ready else 0)
    if not is_ready:
        api_errors_total.labels("/ready").inc()
        raise HTTPException(status_code=503, detail=f"Model server not ready: {status}")
    api_requests_total.labels("GET", "/ready", "200").inc()
    return {"status": "ready", "model_server": status, "model_ready": is_ready}

@app.get("/metrics", tags=["infra"])
async def metrics():
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)

@app.get("/experiments", tags=["mlflow"])
async def list_experiments(top_n: int = 10):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [],
                    "filter": "attributes.status = 'FINISHED'",
                    "order_by": ["metrics.mAA_overall DESC"],
                    "max_results": top_n,
                },
            )
            return r.json() if r.status_code == 200 else {"error": f"MLflow {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/drift", tags=["monitoring"])
async def drift_check():
    from scripts.drift_monitor import DriftMonitor, update_prometheus_drift_metrics
    data_dir = Path(os.environ.get("DEFAULT_DATASET_DIR", "data"))
    monitor = DriftMonitor(
        baselines_path=data_dir / "processed" / "eda_baselines.json",
        features_dir=data_dir / "processed" / "features",
        mlflow_uri=MLFLOW_TRACKING_URI,
    )
    report = monitor.check(report_path=data_dir / "processed" / "drift_report.json", check_performance=True)
    update_prometheus_drift_metrics(report)
    api_requests_total.labels("GET", "/drift", "200").inc()
    return report.as_dict()

@app.post("/drift/trigger-retrain", tags=["monitoring"])
async def trigger_retrain():
    import datetime
    airflow_url = os.environ.get("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{airflow_url}/api/v2/dags/drift_retrain_pipeline/dagRuns",
                json={"logical_date": datetime.datetime.utcnow().isoformat() + "Z", "conf": {"triggered_by": "api"}},
                headers={"Content-Type": "application/json"},
            )
            return {"status": "triggered"} if r.status_code in (200, 201) else {"status": "error"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints - Inference (UI and Legacy)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload", tags=["inference"])
@app.post("/reconstruct/async", tags=["inference"])
async def upload_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP archive of images"),
    dataset_name: str = "custom",
    scene_name: str = "scene_01",
):
    """
    Unified endpoint for the new UI (/upload) and old ML (/reconstruct/async).
    """
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_SIZE_MB} MB")
    if not zipfile.is_zipfile(io.BytesIO(content)):
        raise HTTPException(status_code=400, detail="Invalid ZIP")

    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    background_tasks.add_task(
        _background_reconstruction,
        job_id=job_id,
        content=content,
        filename=file.filename or "images.zip",
        dataset_name=dataset_name,
        scene_name=scene_name,
    )
    api_requests_total.labels("POST", "/upload", "202").inc()
    return {"job_id": job_id, "message": "Pipeline started."}


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["inference"])
@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["inference"])
async def get_status(job_id: str):
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    rec = job_manager.get_job(job_id)
    
    download_url = None
    if rec.stage == JobStage.DONE and rec.result_path:
        # Legacy compatibility uses /jobs/../download to fetch CSV
        download_url = f"/jobs/{job_id}/download"
        
    return JobStatusResponse(
        job_id=rec.job_id,
        stage=rec.stage.value,
        status=rec.stage.value,  # Legacy alias
        progress=rec.progress,
        message=rec.message,
        created_at=rec.created_at,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        n_images=rec.n_images,
        n_points=rec.n_points,
        maa=rec.maa,
        registration_rate=rec.registration_rate,
        mlflow_run_id=rec.mlflow_run_id,
        error=rec.error,
        download_url=download_url,
    )

@app.get("/download/{job_id}", tags=["inference"])
async def download_ply(job_id: str):
    """UI endpoint for getting the 3D PLY model."""
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404)
    rec = job_manager.get_job(job_id)
    if rec.stage != JobStage.DONE:
        raise HTTPException(status_code=409, detail="Not done yet")
    if not rec.ply_path or not Path(rec.ply_path).exists():
        raise HTTPException(status_code=404, detail="PLY not found")
    return FileResponse(rec.ply_path, media_type="application/octet-stream")

@app.get("/jobs/{job_id}/download", tags=["inference"])
async def download_legacy_csv(job_id: str):
    """Legacy endpoint for downloading submission CSV."""
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404)
    rec = job_manager.get_job(job_id)
    if rec.stage != JobStage.DONE:
        raise HTTPException(status_code=409)
    if not rec.result_path or not Path(rec.result_path).exists():
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(rec.result_path, filename=f"submission_{job_id[:8]}.csv", media_type="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Background Task
# ─────────────────────────────────────────────────────────────────────────────

# Moved to JobManager definition

async def _background_reconstruction(
    job_id: str, content: bytes, filename: str, dataset_name: str, scene_name: str
):
    rec = job_manager.get_job(job_id)
    rec.started_at = time.time()
    active_jobs_gauge.inc()

    async with job_manager.semaphore:
        try:
            job_manager.update_job(job_id, JobStage.MATCHING, "Running MASt3R AI pipeline on GPU ...")
            
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(
                    f"{MODEL_SERVER_URL}/infer",
                    files={"images": (filename, content, "application/zip")},
                    data={"job_id": job_id, "dataset_name": dataset_name, "scene_name": scene_name},
                )
                if r.status_code != 200:
                    raise RuntimeError(f"Model server error {r.status_code}: {r.text[:300]}")
                
                result = r.json()
                
            # Parse result
            job_manager.update_job(job_id, JobStage.TRIANGULATING, "Inference completed ...",
                    maa=result.get("maa"),
                    registration_rate=result.get("registration_rate"),
                    mlflow_run_id=result.get("mlflow_run_id"),
                    n_images=result.get("n_images", 0),
                    result_path=result.get("result_csv_path"))

            raw_ply_path = result.get("raw_ply_path")
            
            # Decimate
            if raw_ply_path and Path(raw_ply_path).exists():
                job_manager.update_job(job_id, JobStage.DECIMATING, "Optimising 3D point cloud for browser ...")
                
                import sys
                project_root = Path(__file__).resolve().parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from utils.decimate import voxel_downsample_ply, get_point_cloud_stats
                
                # Write to the same directory as the raw PLY
                decimated_ply = str(Path(raw_ply_path).parent / f"decimated_{job_id[:8]}.ply")
                voxel_downsample_ply(raw_ply_path, decimated_ply, voxel_size=VOXEL_SIZE, max_points=MAX_POINT_COUNT)
                
                stats = get_point_cloud_stats(decimated_ply)
                job_manager.update_job(job_id, JobStage.DONE, f"Reconstruction successful ({stats.n_points:,} points)",
                        ply_path=decimated_ply, n_points=stats.n_points)
            else:
                log.warning("No PLY received from model-server. Using default success state.")
                job_manager.update_job(job_id, JobStage.DONE, "Reconstruction successful (no point cloud available).")

            elapsed = time.time() - rec.started_at
            inference_latency_seconds.observe(elapsed)
            if rec.maa is not None:
                reconstruction_maa.set(rec.maa)
            if rec.registration_rate is not None:
                registration_rate_gauge.set(rec.registration_rate)

        except Exception as e:
            log.error("Job %s failed: %s", job_id, e)
            job_manager.update_job(job_id, JobStage.FAILED, f"Pipeline error: {e}", error=str(e))
            api_errors_total.labels("/reconstruct/async").inc()
        finally:
            rec.finished_at = time.time()
            active_jobs_gauge.dec()
