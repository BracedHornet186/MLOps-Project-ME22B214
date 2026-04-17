"""
api/main.py
────────────────────────────────────────────────────────────────────────────
Stage 4 — Model Deployment: FastAPI inference server

Endpoints
---------
GET  /health          Liveness probe — returns 200 if process is alive
GET  /ready           Readiness probe — 200 only after model weights loaded
POST /reconstruct     Synchronous: upload ZIP of images → submission CSV
POST /reconstruct/async  Async: returns job_id immediately
GET  /jobs/{job_id}   Poll job status, download result when done
GET  /metrics         Prometheus metrics endpoint
GET  /experiments     List recent MLflow runs (leaderboard)

Design
------
- Loose coupling: this file has NO import of pipeline code.
  All ML work is delegated to model_server (http://model-server:8001)
  via httpx async calls. The API and model server are independent.
- Single GPU (RTX 3060): only one job runs at a time (concurrency=1).
- MLflow run context is attached to every /reconstruct response so
  the caller can trace the exact config that produced the result.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
MAX_CONCURRENT_JOBS = 1          # single GPU — serialise all jobs
MAX_UPLOAD_SIZE_MB = 500
RESULT_TTL_SECONDS = 3600        # clean up job results after 1 hour

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

api_requests_total = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
api_errors_total = Counter(
    "api_errors_total",
    "Total 4xx/5xx responses",
    ["endpoint"],
)
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "End-to-end inference wall-clock time per job",
    buckets=[10, 30, 60, 120, 180, 300, 600, 900],
)
reconstruction_maa = Gauge(
    "reconstruction_maa",
    "mAA score from the most recent completed reconstruction job",
)
registration_rate_gauge = Gauge(
    "registered_images_ratio",
    "Fraction of images successfully placed in the last reconstruction",
)
active_jobs_gauge = Gauge(
    "active_jobs_total",
    "Number of reconstruction jobs currently running",
)
model_ready_gauge = Gauge(
    "model_server_ready",
    "1 if model server is ready, 0 otherwise",
)
data_valid_images_gauge = Gauge(
    "data_valid_images_total",
    "Valid images in current dataset version (from validation_report.json)",
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    n_images: int = 0
    mlflow_run_id: Optional[str] = None
    maa: Optional[float] = None
    registration_rate: Optional[float] = None


_jobs: dict[str, JobRecord] = {}
_job_semaphore: asyncio.Semaphore  # initialised in lifespan


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_semaphore
    _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    log.info("API starting — model server: %s", MODEL_SERVER_URL)

    # Refresh data metrics from validation report
    _refresh_data_metrics()

    # Warm-up: check model server readiness
    asyncio.create_task(_poll_model_server_ready())
    yield
    log.info("API shutting down")


def _refresh_data_metrics() -> None:
    report_path = Path(os.environ.get("DEFAULT_DATASET_DIR", "data")) / "processed" / "validation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            data_valid_images_gauge.set(report.get("total_images", 0))
        except Exception as e:
            log.warning("Could not read validation_report.json: %s", e)


async def _poll_model_server_ready() -> None:
    """Background task: poll model-server /ready until it responds 200."""
    for attempt in range(30):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{MODEL_SERVER_URL}/ready")
                if r.status_code == 200:
                    model_ready_gauge.set(1)
                    log.info("Model server is ready.")
                    return
        except Exception:
            pass
        await asyncio.sleep(10)
    model_ready_gauge.set(0)
    log.warning("Model server did not become ready within 300s.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scene Reconstruction API",
    description="3D scene reconstruction from multi-view images using MASt3R + COLMAP",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float


class ReadyResponse(BaseModel):
    status: str
    model_server: str
    model_ready: bool
    version: str


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
    status: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    n_images: int
    maa: Optional[float]
    registration_rate: Optional[float]
    mlflow_run_id: Optional[str]
    error: Optional[str]
    download_url: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health():
    """Liveness probe. Returns 200 immediately if the process is alive."""
    api_requests_total.labels("GET", "/health", "200").inc()
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        timestamp=time.time(),
    )


@app.get("/ready", response_model=ReadyResponse, tags=["infra"])
async def ready():
    """
    Readiness probe. Returns 200 only when the model server has loaded
    all weights and is accepting inference requests.
    """
    is_ready = False
    model_server_status = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{MODEL_SERVER_URL}/ready")
            is_ready = r.status_code == 200
            model_server_status = "ready" if is_ready else "loading"
    except Exception as e:
        model_server_status = f"error: {e}"

    model_ready_gauge.set(1 if is_ready else 0)
    status_code = 200 if is_ready else 503
    api_requests_total.labels("GET", "/ready", str(status_code)).inc()

    if not is_ready:
        api_errors_total.labels("/ready").inc()
        raise HTTPException(status_code=503, detail=f"Model server not ready: {model_server_status}")

    return ReadyResponse(
        status="ready",
        model_server=model_server_status,
        model_ready=is_ready,
        version=API_VERSION,
    )


@app.get("/metrics", tags=["infra"])
async def metrics():
    """Prometheus metrics endpoint."""
    return StreamingResponse(
        io.BytesIO(generate_latest()),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/reconstruct", response_model=ReconstructResponse, tags=["inference"])
async def reconstruct_sync(
    images: UploadFile = File(..., description="ZIP archive of images"),
    dataset_name: str = "custom",
    scene_name: str = "scene_01",
):
    """
    Synchronous reconstruction endpoint.
    Upload a ZIP of images; receive the submission CSV inline.
    Blocks until inference is complete (may take several minutes).
    """
    job_id = str(uuid.uuid4())
    t_start = time.perf_counter()


    try:
        result = await _run_reconstruction(
            job_id=job_id,
            upload=images,
            dataset_name=dataset_name,
            scene_name=scene_name,
        )
    except Exception as e:
        api_errors_total.labels("/reconstruct").inc()
        api_requests_total.labels("POST", "/reconstruct", "500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - t_start
    inference_latency_seconds.observe(elapsed)

    if result.get("maa") is not None:
        reconstruction_maa.set(result["maa"])
    if result.get("registration_rate") is not None:
        registration_rate_gauge.set(result["registration_rate"])

    api_requests_total.labels("POST", "/reconstruct", "200").inc()

    return ReconstructResponse(
        job_id=job_id,
        status="success",
        n_images=result.get("n_images", 0),
        maa=result.get("maa"),
        registration_rate=result.get("registration_rate"),
        mlflow_run_id=result.get("mlflow_run_id"),
        latency_seconds=round(elapsed, 2),
    )


@app.post("/reconstruct/async", response_model=JobStatusResponse, tags=["inference"])
async def reconstruct_async(
    background_tasks: BackgroundTasks,
    images: UploadFile = File(...),
    dataset_name: str = "custom",
    scene_name: str = "scene_01",
):
    """
    Asynchronous reconstruction endpoint.
    Returns a job_id immediately; poll /jobs/{job_id} for results.
    """
    job_id = str(uuid.uuid4())
    record = JobRecord(job_id=job_id, created_at=time.time())
    _jobs[job_id] = record

    # Read the upload into memory now so the UploadFile isn't closed
    # before the background task runs.
    content = await images.read()

    background_tasks.add_task(
        _background_reconstruction,
        job_id=job_id,
        content=content,
        filename=images.filename or "images.zip",
        dataset_name=dataset_name,
        scene_name=scene_name,
    )

    api_requests_total.labels("POST", "/reconstruct/async", "202").inc()
    return _job_to_response(record)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["inference"])
async def get_job(job_id: str):
    """Poll job status. When status='success', download_url is populated."""
    if job_id not in _jobs:
        api_errors_total.labels("/jobs").inc()
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    api_requests_total.labels("GET", "/jobs", "200").inc()
    return _job_to_response(_jobs[job_id])


@app.get("/jobs/{job_id}/download", tags=["inference"])
async def download_result(job_id: str):
    """Download the submission CSV for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    record = _jobs[job_id]
    if record.status != JobStatus.SUCCESS:
        raise HTTPException(status_code=409, detail=f"Job status is {record.status}")
    if not record.result_path or not Path(record.result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=record.result_path,
        filename=f"submission_{job_id[:8]}.csv",
        media_type="text/csv",
    )


@app.get("/experiments", tags=["mlflow"])
async def list_experiments(top_n: int = 10):
    """
    Proxy to MLflow: return the top-N runs from the scene_reconstruction
    experiment ordered by mAA descending.
    """
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
            if r.status_code == 200:
                return r.json()
            return {"error": f"MLflow returned {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/drift", tags=["monitoring"])
async def drift_check():
    """
    Run on-demand drift detection against EDA baselines.
    Returns the drift report with all alerts.
    """
    from scripts.drift_monitor import (
        DriftMonitor,
        update_prometheus_drift_metrics,
    )

    data_dir = Path(os.environ.get("DEFAULT_DATASET_DIR", "data"))
    monitor = DriftMonitor(
        baselines_path=data_dir / "processed" / "eda_baselines.json",
        features_dir=data_dir / "processed" / "features",
        mlflow_uri=MLFLOW_TRACKING_URI,
    )
    report = monitor.check(
        report_path=data_dir / "processed" / "drift_report.json",
        check_performance=True,
    )
    update_prometheus_drift_metrics(report)

    api_requests_total.labels("GET", "/drift", "200").inc()
    return report.as_dict()


@app.post("/drift/trigger-retrain", tags=["monitoring"])
async def trigger_retrain():
    """
    Trigger the drift_retrain_pipeline DAG via the Airflow REST API.
    """
    import datetime

    airflow_url = os.environ.get("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{airflow_url}/api/v2/dags/drift_retrain_pipeline/dagRuns",
                json={
                    "logical_date": datetime.datetime.utcnow().isoformat() + "Z",
                    "conf": {"triggered_by": "api"},
                },
                headers={"Content-Type": "application/json"},
            )
            if r.status_code in (200, 201):
                api_requests_total.labels("POST", "/drift/trigger-retrain", "200").inc()
                return {"status": "triggered", "response": r.json()}
            else:
                api_errors_total.labels("/drift/trigger-retrain").inc()
                return {"status": "error", "code": r.status_code, "detail": r.text[:500]}
    except Exception as e:
        api_errors_total.labels("/drift/trigger-retrain").inc()
        raise HTTPException(status_code=502, detail=f"Could not reach Airflow: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _run_reconstruction(
    job_id: str,
    upload: UploadFile,
    dataset_name: str,
    scene_name: str,
) -> dict[str, Any]:
    """
    Delegate inference to the model server via HTTP.
    Returns a dict with keys: n_images, maa, registration_rate, mlflow_run_id.
    """
    content = await upload.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Upload exceeds {MAX_UPLOAD_SIZE_MB} MB limit")

    active_jobs_gauge.inc()
    async with _job_semaphore:
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(
                    f"{MODEL_SERVER_URL}/infer",
                    files={"images": (upload.filename, content, "application/zip")},
                    data={
                        "job_id": job_id,
                        "dataset_name": dataset_name,
                        "scene_name": scene_name,
                    },
                )
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Model server error {r.status_code}: {r.text[:500]}"
                    )
                return r.json()
        finally:
            active_jobs_gauge.dec()


async def _background_reconstruction(
    job_id: str,
    content: bytes,
    filename: str,
    dataset_name: str,
    scene_name: str,
) -> None:
    """Background task for async reconstruction endpoint."""
    record = _jobs[job_id]
    record.status = JobStatus.RUNNING
    record.started_at = time.time()
    active_jobs_gauge.inc()

    async with _job_semaphore:
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                r = await client.post(
                    f"{MODEL_SERVER_URL}/infer",
                    files={"images": (filename, content, "application/zip")},
                    data={
                        "job_id": job_id,
                        "dataset_name": dataset_name,
                        "scene_name": scene_name,
                    },
                )
                if r.status_code != 200:
                    raise RuntimeError(f"Model server returned {r.status_code}: {r.text[:300]}")

                result = r.json()
                record.maa = result.get("maa")
                record.registration_rate = result.get("registration_rate")
                record.mlflow_run_id = result.get("mlflow_run_id")
                record.n_images = result.get("n_images", 0)
                record.result_path = result.get("result_csv_path")
                record.status = JobStatus.SUCCESS

                elapsed = time.time() - record.started_at
                inference_latency_seconds.observe(elapsed)
                if record.maa is not None:
                    reconstruction_maa.set(record.maa)
                if record.registration_rate is not None:
                    registration_rate_gauge.set(record.registration_rate)

        except Exception as e:
            log.error("Job %s failed: %s", job_id, e)
            record.status = JobStatus.FAILED
            record.error = str(e)
            api_errors_total.labels("/reconstruct/async").inc()
        finally:
            record.finished_at = time.time()
            active_jobs_gauge.dec()


def _job_to_response(record: JobRecord) -> JobStatusResponse:
    download_url = None
    if record.status == JobStatus.SUCCESS and record.result_path:
        download_url = f"/jobs/{record.job_id}/download"
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status.value,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        n_images=record.n_images,
        maa=record.maa,
        registration_rate=record.registration_rate,
        mlflow_run_id=record.mlflow_run_id,
        error=record.error,
        download_url=download_url,
    )
