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

  APIGateway       — num_replicas=1, CPU-only
                     Wraps FastAPI via @serve.ingress.
                     Handles uploads, job management, Prometheus metrics,
                     drift endpoints.  All inference delegated to GPU worker
                     over Ray object store (zero HTTP overhead).

Binding:
  gpu_node = GPUModelWorker.bind()
  api_node = APIGateway.bind(gpu_node)

Start with:
  serve run serve_app:api_node
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
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
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
from ray.serve.schema import LoggingConfig

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("serve_app")

serve.start(detached=True, 
            http_options={"host": "0.0.0.0", "port": 8000},
            logging_config={"log_level": "INFO"},
        )

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
DATA_DIR              = Path(os.environ.get("DEFAULT_DATASET_DIR", "data"))
RESULTS_DIR           = Path(os.environ.get("RESULTS_DIR", "/tmp/reconstruction_results"))
AIRFLOW_API_URL       = os.environ.get("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
ALLOWED_IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

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
    has_drift:         Optional[bool]   = None
    drift_severity:    Optional[str]    = None
    drift_report:      Optional[dict]   = None

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
    status:            str
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
    has_drift:         Optional[bool]   = None
    drift_severity:    Optional[str]    = None

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
    LoggingConfig(log_level="INFO"),
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
)
class GPUModelWorker:
    """
    Holds the MASt3R pipeline permanently in VRAM.
    Loads once at startup; never reloads between requests.
    job state lives in the API gateway's JobManager.
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
          6. Return result dict
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
                log.info("Job %s: extracted %d images", job_id, n_images)

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
                log.info("Job %s: starting pipeline …", job_id)
                
                # Monkey-patch the scenes config so it persists data into our job's tmpdir
                import pipelines.scene
                pipelines.scene.IS_SCENE_SPACE_DIR_PERSISTENT = True
                pipelines.scene.DEFAULT_TMP_DIR = tmpdir
                pipelines.scene.DEFAULT_SPACE_NAME = "colmap_outputs"

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
                persist_plys: list[Path] = []
                try:
                    import pycolmap

                    for pts_file in tmpdir.rglob("points3D.bin"):
                        try:
                            rec_dir = pts_file.parent
                            rec = pycolmap.Reconstruction(str(rec_dir))

                            if len(rec.points3D) == 0:
                                continue

                            # Extract cluster name and model name from path
                            # typically: <cluster_name>/colmap_rec/<model_id>/points3D.bin
                            cluster_name = rec_dir.parent.parent.name
                            model_name = rec_dir.name

                            ply_path = RESULTS_DIR / f"{cluster_name}_model{model_name}_{job_id[:8]}.ply"

                            rec.export_PLY(str(ply_path))

                            persist_plys.append(str(ply_path))

                            log.info(
                                "Job %s: exported %s (%d pts) → %s",
                                job_id,
                                cluster_name,
                                len(rec.points3D),
                                ply_path,
                            )

                        except Exception as e:
                            log.warning("Skipping reconstruction: %s", e)

                    if not persist_plys:
                        log.info("Job %s: no valid reconstructions found", job_id)

                except Exception as e:
                    log.warning("Job %s: PLY export failed: %s", job_id, e)
                log.info(
                    "Job %s done — registered=%d/%d (%.1f%%)  elapsed=%.1fs",
                    job_id, registered, n_images, 100 * reg_rate, elapsed,
                )

                return {
                    "job_id":                  job_id,
                    "n_images":                n_images,
                    "registered":              registered,
                    "registration_rate":       round(reg_rate, 4),
                    "inference_latency_seconds": round(elapsed, 2),
                    "result_csv_path":         str(persist_csv),
                    "raw_ply_paths":            [str(ply) for ply in persist_plys] if persist_plys else None,
                }

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (shared across API gateway replicas)
# ─────────────────────────────────────────────────────────────────────────────

fastapi_app = FastAPI(title="Scene Reconstruction API", version=API_VERSION)

# ── Auth router ───────────────────────────────────────────────────────────────
from api.auth import auth_router, get_current_user
fastapi_app.include_router(auth_router)

# ── Middleware ────────────────────────────────────────────────────────────────
from api.middleware import AccessLogMiddleware
fastapi_app.add_middleware(AccessLogMiddleware)

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
    LoggingConfig(log_level="INFO"),
    num_replicas=1,
    ray_actor_options={"num_gpus": 0, "num_cpus": 2},
)
@serve.ingress(fastapi_app)
class APIGateway:


    def __init__(self, gpu_worker_handle):
        self.gpu_worker  = gpu_worker_handle
        self.job_manager = JobManager(MAX_CONCURRENT_JOBS)
        self.job_manager.init_semaphore()
        self._refresh_data_metrics()

        # ─────────────────────────────────────────────────────────────────────────────
        # Prometheus metrics  (module-level so both deployments share the same registry)
        # ─────────────────────────────────────────────────────────────────────────────

        self.api_requests_total       = Counter("api_requests_total",       "Total HTTP requests",              ["method", "endpoint", "status"])
        self.api_errors_total         = Counter("api_errors_total",         "Total 4xx/5xx responses",          ["endpoint"])
        self.inference_latency_seconds= Histogram("inference_latency_seconds","End-to-end inference wall-clock", buckets=[10,30,60,120,180,300,600,900])
        self.reconstruction_maa       = Gauge("reconstruction_maa",         "mAA score of most recent job")
        self.registration_rate_gauge  = Gauge("registered_images_ratio",    "Fraction of images placed")
        self.active_jobs_gauge        = Gauge("active_jobs_total",          "Running jobs")
        self.model_ready_gauge        = Gauge("model_server_ready",         "1 if GPU worker ready")
        self.data_valid_images_gauge  = Gauge("data_valid_images_total",    "Valid images in current dataset")


    # ── Infrastructure ────────────────────────────────────────────────────────

    @fastapi_app.get("/health", response_model=HealthResponse, tags=["infra"])
    async def health(self):
        return HealthResponse(
            status="ok",
            version=API_VERSION,
            timestamp=time.time()
        )

    @fastapi_app.get("/ready", tags=["infra"])
    async def ready(self):
        try:
            result = await self.gpu_worker.ping.remote()
            if result.get("ready"):
                self.model_ready_gauge.set(1)
                self.api_requests_total.labels("GET", "/ready", "200").inc()
                return {"status": "ready", "device": result.get("device", "unknown")}
        except Exception as e:
            self.model_ready_gauge.set(0)
            self.api_errors_total.labels("/ready").inc()
            raise HTTPException(status_code=503, detail=f"GPU worker not ready: {e}")
        self.model_ready_gauge.set(0)
        self.api_errors_total.labels("/ready").inc()
        raise HTTPException(status_code=503, detail="GPU worker not ready")

    @fastapi_app.get("/metrics", tags=["infra"])
    async def metrics(self):
        return StreamingResponse(
            io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST
        )

    # ── Drift monitoring ─────────────────────────────

    async def _analyze_drift(self, content: bytes) -> dict:
        from scripts.drift_monitor import DriftMonitor, update_prometheus_drift_metrics
        import cv2
        import numpy as np

        heights, widths = [], []
        sharpness, brightness, contrast = [], [], []

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if name.startswith("__MACOSX/") or name.startswith("._"): continue
                if not any(name.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTS): continue
                img_bytes = zf.read(name)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None: continue

                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness.append(np.mean(gray))
                contrast.append(np.std(gray))

        if not heights:
            raise ValueError("No valid images found in ZIP")

        live_stats = {
            "height_mean": float(np.mean(heights)),
            "width_mean": float(np.mean(widths)),
            "sharpness_mean": float(np.mean(sharpness)),
            "brightness_mean": float(np.mean(brightness)),
            "contrast_mean": float(np.mean(contrast)),
            "num_images": len(heights),
            "brightness_std": float(np.std(brightness)),
            "contrast_std": float(np.std(contrast)),
            "sharpness_std": float(np.std(sharpness)),
            "aspect_ratio_mean": float(np.mean([w/h for h, w in zip(heights, widths)])),
        }

        monitor  = DriftMonitor(
            baselines_path=DATA_DIR / "baselines" / "eda_baselines.json",
        )
        report = monitor.check(
            live_stats=live_stats,
            report_path=RESULTS_DIR / "drift_report.json",
            check_performance=True,
        )
        update_prometheus_drift_metrics(report)
        return report.as_dict()

    @fastapi_app.post("/drift", tags=["monitoring"])
    async def drift_check(self, file: UploadFile = File(..., description="ZIP archive of images"), user: str = Depends(get_current_user)):
        content = await file.read()
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise HTTPException(status_code=400, detail="Invalid ZIP")

        try:
            report_dict = await self._analyze_drift(content)
            self.api_requests_total.labels("POST", "/drift", "200").inc()
            return report_dict
        except ValueError as e:
            self.api_requests_total.labels("POST", "/drift", "400").inc()
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.api_requests_total.labels("POST", "/drift", "500").inc()
            raise HTTPException(status_code=500, detail=f"Drift check failed: {e}")

    @fastapi_app.post("/drift/trigger-retrain", tags=["monitoring"])
    async def trigger_retrain(self, user: str = Depends(get_current_user)):
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
        user: str = Depends(get_current_user),
    ):
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_SIZE_MB} MB")
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise HTTPException(status_code=400, detail="Invalid ZIP")

        # ── Drift Check ──────────────────────────────────────────────────────
        has_drift, drift_severity, drift_report = None, None, None
        try:
            drift_report = await self._analyze_drift(content)
            has_drift = drift_report.get("drift_detected", False)
            if not has_drift and "has_drift" in drift_report:
                has_drift = drift_report.get("has_drift")
            drift_severity = drift_report.get("severity", "low")
            
            if has_drift:
                log.warning(f"Drift detected in upload. Severity: {drift_severity}")
                if drift_severity == "high":
                    background_tasks.add_task(self.trigger_retrain)
        except Exception as e:
            log.warning("Drift analysis skipped or failed for upload: %s", e)

        job_id = str(uuid.uuid4())
        self.job_manager.create_job(job_id)
        
        if drift_report is not None:
            self.job_manager.update_job(
                job_id, JobStage.QUEUED, "Waiting in queue …",
                has_drift=has_drift,
                drift_severity=drift_severity,
                drift_report=drift_report
            )

        background_tasks.add_task(
            self._background_reconstruction,
            job_id=job_id,
            content=content,
            filename=file.filename or "images.zip",
            dataset_name=dataset_name,
            scene_name=scene_name,
        )
        self.api_requests_total.labels("POST", "/upload", "202").inc()
        return {"job_id": job_id, "message": "Pipeline started."}

    # ── Status / download ─────────────────────────────────────────────────────

    @fastapi_app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["inference"])
    @fastapi_app.get("/jobs/{job_id}",   response_model=JobStatusResponse, tags=["inference"])
    async def get_status(self, job_id: str, user: str = Depends(get_current_user)):
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        rec          = self.job_manager.get_job(job_id)
        download_url = f"/download/jobs/{job_id}" if rec.stage == JobStage.DONE and rec.ply_path else None
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
            has_drift=rec.has_drift,
            drift_severity=rec.drift_severity,
        )

    @fastapi_app.get("/download/jobs/{job_id}", tags=["inference"])
    async def download_ply(self, job_id: str, user: str = Depends(get_current_user)):
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404)
        rec = self.job_manager.get_job(job_id)
        if rec.stage != JobStage.DONE:
            raise HTTPException(status_code=409, detail="Not done yet")
        if not rec.ply_path or not Path(rec.ply_path).exists():
            raise HTTPException(status_code=404, detail="PLY not found")
        ply_paths = json.loads(rec.ply_path)
        zip_path = Path("/tmp") / f"{job_id}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in ply_paths:
                zf.write(p, arcname=Path(p).name)
        return FileResponse(zip_path, media_type="application/zip")

    @fastapi_app.get("/download/jobs/{job_id}/csv", tags=["inference"])
    async def download_csv(self, job_id: str, user: str = Depends(get_current_user)):
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

    @fastapi_app.get("/clusters/{job_id}", tags=["inference"])
    async def get_clusters(self, job_id: str, user: str = Depends(get_current_user)):
        """Per-cluster reconstruction statistics for the dashboard."""
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        rec = self.job_manager.get_job(job_id)
        if rec.stage != JobStage.DONE or not rec.ply_path:
            return {"clusters": []}

        ply_paths = json.loads(rec.ply_path)
        clusters = []
        for idx, ply in enumerate(ply_paths):
            p = Path(ply)
            if not p.exists():
                continue
            # Try to get stats from the decimated PLY
            try:
                from utils.decimate import get_point_cloud_stats
                stats = get_point_cloud_stats(str(p))
                n_pts = stats.n_points
            except Exception:
                n_pts = 0

            # Parse cluster/model name from filename pattern:
            # <cluster>_decimated_<model>_<jobid>.ply
            stem_parts = p.stem.split("_")
            cluster_name = stem_parts[0] if stem_parts else f"cluster{idx}"
            model_name = stem_parts[2] if len(stem_parts) > 2 else f"model{idx}"

            clusters.append({
                "id": idx,
                "name": f"{cluster_name}_{model_name}",
                "num_points3D": n_pts,
                "filename": p.name,
            })
        return {"clusters": clusters}

    @fastapi_app.get("/jobs/{job_id}/insights", tags=["inference"])
    async def get_job_insights(self, job_id: str, user: str = Depends(get_current_user)):
        """Get reconstruction + drift insights for a job."""
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        rec = self.job_manager.get_job(job_id)
        
        recommendation = "No action needed."
        if rec.has_drift and rec.drift_severity == "high":
            recommendation = "High drift detected. Evaluated auto-retraining trigger."
        elif rec.has_drift:
            recommendation = "Moderate drift detected. Monitor future submissions."
            
        return {
            "registration_rate": rec.registration_rate,
            "n_points": rec.n_points,
            "has_drift": rec.has_drift,
            "drift_severity": rec.drift_severity,
            "drift_report": rec.drift_report,
            "recommendation": recommendation,
        }

    @fastapi_app.get("/download/jobs/{job_id}/{filename}", tags=["inference"])
    async def download_single_ply(self, job_id: str, filename: str, user: str = Depends(get_current_user)):
        """Download a single PLY file by name."""
        if job_id not in self.job_manager.jobs:
            raise HTTPException(status_code=404)
        rec = self.job_manager.get_job(job_id)
        if rec.stage != JobStage.DONE or not rec.ply_path:
            raise HTTPException(status_code=409, detail="Not done yet")
        ply_paths = json.loads(rec.ply_path)
        for p in ply_paths:
            if Path(p).name == filename:
                if Path(p).exists():
                    return FileResponse(
                        p,
                        media_type="application/octet-stream",
                        filename=filename,
                    )
        raise HTTPException(status_code=404, detail="PLY file not found")

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
        self.active_jobs_gauge.inc()

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

                raw_plys = result.get("raw_ply_paths", [])

                # CPU-side decimation ─────────────────────────────────────────
                decimated_plys = []

                if raw_plys:
                    self.job_manager.update_job(
                        job_id, JobStage.DECIMATING,
                        f"Optimising {len(raw_plys)} clusters …",
                    )

                    from utils.decimate import voxel_downsample_ply, get_point_cloud_stats

                    for raw_ply in raw_plys:
                        if not Path(raw_ply).exists():
                            continue

                        cluster_name = Path(raw_ply).stem.split("_")[0]
                        model_name = Path(raw_ply).stem.split("_")[1]

                        decimated_ply = str(
                            Path(raw_ply).parent / f"{cluster_name}_decimated_{model_name}_{job_id[:8]}.ply"
                        )

                        await asyncio.to_thread(
                            voxel_downsample_ply,
                            raw_ply,
                            decimated_ply,
                            voxel_size=VOXEL_SIZE,
                            max_points=MAX_POINT_COUNT,
                        )

                        decimated_plys.append(decimated_ply)

                        total_points = 0

                        for ply in decimated_plys:
                            stats = get_point_cloud_stats(ply)
                            total_points += stats.n_points

                    self.job_manager.update_job(
                    job_id,
                    JobStage.DONE,
                    f"{len(decimated_plys)} clusters reconstructed ({total_points:,} total points)",
                    ply_path=json.dumps(decimated_plys),   # store list
                    n_points=total_points,
                )
                else:
                    log.warning("Job %s: no PLY from GPU worker — skipping decimation", job_id)
                    self.job_manager.update_job(
                        job_id, JobStage.DONE,
                        "Reconstruction successful (no point cloud available).",
                    )

                # Update Prometheus ────────────────────────────────────────────
                elapsed = time.time() - rec.started_at
                self.inference_latency_seconds.observe(elapsed)
                if rec.registration_rate is not None:
                    self.registration_rate_gauge.set(rec.registration_rate)

            except Exception as e:
                log.error("Job %s failed: %s", job_id, e, exc_info=True)
                self.job_manager.update_job(
                    job_id, JobStage.FAILED,
                    f"Pipeline error: {e}",
                    error=str(e),
                )
                self.api_errors_total.labels("/upload").inc()
            finally:
                rec.finished_at = time.time()
                self.active_jobs_gauge.dec()

                # Save temporal drift and metrics history
                try:
                    drift_history_path = RESULTS_DIR / "drift_history.jsonl"
                    drift_history_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    drift_entry = {
                        "timestamp": rec.finished_at,
                        "job_id": job_id,
                        "registration_rate": rec.registration_rate,
                        "n_images": rec.n_images,
                        "n_points": rec.n_points,
                        "has_drift": rec.has_drift,
                        "drift_severity": rec.drift_severity,
                        "drift_report": rec.drift_report,
                    }
                    with open(drift_history_path, "a") as f:
                        f.write(json.dumps(drift_entry) + "\n")
                except Exception as e:
                    log.warning("Failed to save drift history: %s", e)

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
                self.data_valid_images_gauge.set(report.get("total_images", 0))
            except Exception as e:
                log.warning("Could not read validation_report.json: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Bind the deployment graph
# ─────────────────────────────────────────────────────────────────────────────

gpu_node = GPUModelWorker.bind()
api_node = APIGateway.bind(gpu_node)