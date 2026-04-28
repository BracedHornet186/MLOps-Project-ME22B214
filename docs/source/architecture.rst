System Architecture
===================

This page describes the high-level system design, the separation of concerns between
components, and how the MLOps tools integrate with each other.

----

High-Level Architecture
------------------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                         User / Browser                                   │
   └─────────────────────────────┬────────────────────────────────────────────┘
                                 │  HTTPS (port 443 / nginx TLS)
                                 ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                         Frontend (scene3d-ui)                            │
   │           React + Three.js · Vite · nginx reverse proxy                  │
   │                         port 5173 / 443                                  │
   └─────────────────────────────┬────────────────────────────────────────────┘
                                 │  HTTP REST  /api/*
                                 ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                    API Gateway (ray-serve)                               │
   │              FastAPI · Ray Serve ingress · port 8000                     │
   │   Auth · Upload · Job Management · Drift · Prometheus metrics            │
   └────────────────┬─────────────────────────────────────────────────────────┘
                    │  Ray RPC (object store, zero-copy)
                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                   GPU Model Worker (ray-serve)                           │
   │      MASt3R · ALIKED · SuperPoint · COLMAP SfM · pycolmap               │
   │                      1 GPU, 4 CPUs                                       │
   └──────────────────────────────────────────────────────────────────────────┘

   ┌────────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────┐
   │    MLflow      │   │   Airflow    │   │   Prometheus    │   │  Grafana │
   │  port 5000     │   │  port 8080   │   │   port 9090     │   │ port 3001│
   │  Experiment    │   │  Orchestrate │   │  Metrics scrape │   │Dashboard │
   │  tracking      │   │  DVC + DAGs  │   │  + alerts       │   │          │
   └────────────────┘   └──────────────┘   └─────────────────┘   └──────────┘

   ┌────────────────────────────────────────────────────────────────────────┐
   │                  Docker Network: mlops_net                             │
   │  All services communicate by container hostname on this bridge network │
   └────────────────────────────────────────────────────────────────────────┘

----

Frontend
---------

The frontend is a **React** single-page application built with **Vite** and styled
with **Tailwind CSS**. The 3D viewer is implemented with **Three.js**.

**Responsibilities**

- Render the upload form and stage tracker
- Poll ``GET /jobs/{job_id}`` every few seconds to update the UI
- Render the interactive 3D point cloud via Three.js
- Display drift warnings and reconstruction statistics

**Build**

The frontend is built into static files at Docker image build time (``npm run build``).
These are served by an **nginx** container which also acts as a TLS-terminating
reverse proxy, forwarding ``/api/*`` requests to the ``ray-serve`` backend.

----

API Gateway and GPU Worker (Ray Serve)
---------------------------------------

The backend is a single Docker container running two **Ray Serve** deployments:

**APIGateway** (CPU-only, 2 cores)

- Wraps a FastAPI application via ``@serve.ingress``
- Handles all HTTP traffic: authentication, file uploads, job state, drift checks
- Delegates all inference to the GPU worker over Ray's object store (zero HTTP overhead)
- Exposes Prometheus metrics at ``GET /metrics``

**GPUModelWorker** (1 GPU, 4 cores)

- Loads the entire MASt3R pipeline once at container startup
- Exposes ``reconstruct()`` and ``ping()`` methods callable via Ray RPC
- Never reloads models between requests — weights stay permanently in VRAM
- Handles the full pipeline: extraction → matching → COLMAP → PLY export

The two deployments communicate via **Ray's distributed object store**, which
avoids serialising large tensors over HTTP and enables near-zero-copy data transfer.

**Concurrency**

A semaphore limits the API gateway to ``MAX_CONCURRENT_JOBS = 1`` running job at
a time, preventing GPU memory exhaustion.

----

Offline MLOps Pipeline
-----------------------

**DVC** defines the pipeline as a DAG of stages in ``dvc.yaml``. Each stage
specifies its command, input dependencies (``deps``), and outputs (``outs``).
DVC tracks content hashes so only changed stages re-run.

**MLflow** is used for experiment tracking. Every ``dvc repro`` run creates a
parent MLflow run, with child runs logged by each script stage. Logged entities
include:

- All pipeline configuration parameters (flattened from YAML)
- Stage-level metrics (registration rate, mAA, etc.)
- Artifacts (eval CSV, PLY file, config YAML, Git status)

**Airflow** orchestrates the full DVC pipeline via the ``experiment_pipeline_dag``
DAG, which:

1. Waits for required data files via ``FileSensor`` tasks.
2. Runs ``dvc repro`` inside an ephemeral Docker container (Docker-out-of-Docker).
3. Calls ``select_best_run.py`` to promote the best MLflow run to production.
4. Sends an email notification on success.

----

Monitoring Stack
-----------------

**Prometheus** scrapes metrics from:

- The Ray Serve API gateway (``/metrics``, every 10 seconds)
- MLflow health endpoint (every 30 seconds)
- Airflow (every 15 seconds)
- Node Exporter (host-level hardware metrics)

**Grafana** provides dashboards powered by Prometheus data, including:

- Reconstruction job throughput and latency
- Registration rate over time
- GPU utilisation (via node exporter)
- Drift metric trends

**Alertmanager** receives alerts defined in ``monitoring/alert_rules.yml``.
Relevant alert names include ``FeatureDriftDetected``, ``InputBrightnessDrift``,
``InputContrastDrift``, and ``PerformanceDecay``. On firing, Alertmanager calls
an Airflow webhook to trigger ``experiment_pipeline_dag`` automatically.

----

Data Flow
----------

**Offline (training/evaluation)**

.. code-block:: text

   data/train/ (images)
     └─► validate → eda_baselines → preprocess → prepare
                                                     └─► run_pipeline (MASt3R + COLMAP)
                                                                └─► evaluate (mAA → MLflow)
                                                                         └─► select_best_run
                                                                                  └─► conf/best_config.yaml

**Online (inference)**

.. code-block:: text

   User ZIP upload
     └─► Drift check (vs EDA baselines)
           └─► GPUModelWorker.reconstruct()
                 ├─► MASt3R matching
                 ├─► COLMAP SfM
                 └─► PLY export + decimation
                       └─► /app/results/ → download by user

----

Networking
-----------

All services run on a single Docker bridge network named ``mlops_net``.
Service hostnames (e.g., ``mlflow``, ``airflow-apiserver``, ``prometheus``)
resolve directly within this network. This allows:

- Airflow DockerOperator containers to reach ``mlflow:5000``
- Alertmanager to call ``airflow-apiserver:8080``
- Ray Serve to reach ``mlflow:5000`` for metric logging

The network name is pinned in ``docker-compose.yaml``:

.. code-block:: yaml

   networks:
     default:
       name: mlops_net

----

Security Boundaries
--------------------

- **External traffic** enters only through nginx (port 443) and the Ray Serve API
  (port 8000).
- **JWT tokens** expire after 15 minutes and are signed with a secret loaded from
  Docker Secrets.
- **Database credentials** are stored as Docker Secrets, not plaintext environment
  variables.
- **CI/CD** includes Trivy image scanning and ``pip-audit`` dependency auditing
  on every push (see ``.github/workflows/security.yml``).

For full security documentation see :doc:`security`.
