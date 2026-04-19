Data: kagglehub.competition_download('image-matching-challenge-2025')
# Install Python-3.11 if needed.
$ uv python install 3.11

# Use Python-3.11, aligning with the Python version in the Kaggle notebook.
$ uv python pin 3.11
$ python --version
Python 3.11.11

# Install dependencies with pyproject.toml.
# Actually, this command will fail because pre-built ASMK and curope packages is not in bundle/oss
$ uv sync
$ . .venv/bin/activate

Or use docker-compose.yaml to run the project.
Configure .env
AIRFLOW_UID=1000
FERNET_KEY=LS47uw30w1OWKkHCGlSjEKkE3FQ2_ynycWQJ-Sd-y30=
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
AIRFLOW__API_AUTH__JWT_SECRET=somekey

Put models in extra/pretrainned_models:
ALIKED:
wget https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth
ISC:
wget https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar
MASt3R:
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth


git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
python3 setup.py build_ext --inplace
cd ..

# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
git clone https://github.com/naver/croco.git
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../

Build the packages as *.whl file and move them into bundle/oss before uv sync.
python -m build --no-isolation

export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

```
./venv/bin/python3 scripts/train_experiment.py \ 
        --config conf/mast3r.yaml \ 
        --datasets ETs stairs \ 
        --experiment-name scene_reconstruction
```

UI
# Backend (terminal 1)
cd /home/abhiyaan-cu/Yash/MLOps-Project-ME22B214
.venv/bin/uvicorn api.scene3d_server:app --reload --port 8002

# Frontend (terminal 2)
cd frontend && npm run dev
# → http://localhost:5173


docker compose -f docker-compose.scene3d.yaml up --build
# → Frontend: http://localhost:5173
# → API:      http://localhost:8002

## Architecture

```mermaid
graph TD
    subgraph Frontend
        A[React UI]
    end
    subgraph Backend APIs
        B[FastAPI Gateway]
        C[Model Server API / GPU Worker]
    end
    subgraph MLOps & Orchestration
        D[MLflow Tracking Server]
        E[Airflow Scheduler & Workers]
    end
    subgraph Observability
        F[Prometheus]
        G[Grafana]
        H[Node Exporter]
    end
    
    A <-->|HTTP REST / Polling| B
    B <-->|HTTP POST /infer| C
    B <-->|Queried by API| D
    C -->|Logs Experiments & Metrics| D
    E -->|Scheduled DAGs & Jobs| C
    
    B -->|Scraped by| F
    C -->|Scraped by| F
    E -->|Scraped by| F
    H -->|Scraped by| F
    
    F -->|Data Source| G
```
The system consists of two distinct layers: an offline MLOps pipeline and an online inference pipeline.

In the offline layer, Airflow orchestrates the execution of the pipeline, while DVC defines the pipeline stages and ensures reproducibility. Different configurations are experimented with using DVC, and each run logs parameters, metrics, and artifacts to MLflow using a local SQLite backend.

The best-performing configuration is selected based on evaluation metrics and is associated with a specific Git commit and MLflow run. This combination of code version and configuration is promoted to production.

In the production layer, FastAPI serves the application. When a user uploads a zip file of images, the same pipeline logic is executed directly (without DVC), using the selected configuration. The pipeline performs preprocessing, feature extraction, matching, and 3D reconstruction using COLMAP, producing a .ply file which is then visualized in the UI.

This separation ensures reproducibility during development and efficiency during inference.