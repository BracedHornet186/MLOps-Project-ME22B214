# api/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Unified GPU container for Ray Serve deployment graph.
#
# Owns the GPU exclusively via GPUModelWorker (num_replicas=1, num_gpus=1).
# APIGateway replicas run on CPU inside the same process space.
#
# Start:
#   serve run api.serve_app:api_node --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        curl git build-essential \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# ── PyTorch (CUDA 12.6) ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu126

# ── Ray Serve ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir "ray[serve]==2.44.0"

# ── API / gateway deps ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.0 \
        python-multipart \
        httpx==0.27.0 \
        prometheus_client==0.20.0 \
        pydantic>=2.0 \
        pyyaml>=6.0

# ── ML pipeline deps (from model_server/Dockerfile) ──────────────────────────
RUN pip install --no-cache-dir \
        mlflow==3.10.1 \
        pandas>=2.0 \
        "numpy<2" \
        opencv-python-headless>=4.8 \
        pycolmap>=3.11.1 \
        kornia>=0.8.0 \
        Pillow>=10.0 \
        tqdm>=4.60 \
        scipy>=1.10

# ── Point cloud decimation deps ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
        open3d>=0.18.0 \
        trimesh>=4.0.0

# ── Copy project source ───────────────────────────────────────────────────────
COPY scripts/        /app/scripts/
COPY pipelines/      /app/pipelines/
COPY clusterings/    /app/clusterings/
COPY features/       /app/features/
COPY global_descriptors/ /app/global_descriptors/
COPY localizers/     /app/localizers/
COPY matchers/       /app/matchers/
COPY models/         /app/models/
COPY postprocesses/  /app/postprocesses/
COPY preprocesses/   /app/preprocesses/
COPY retrievers/     /app/retrievers/
COPY shortlists/     /app/shortlists/
COPY core/           /app/core/
COPY utils/          /app/utils/
COPY conf/           /app/conf/

# ── Copy serve app ────────────────────────────────────────────────────────────
COPY api/serve_app.py /app/api/serve_app.py
# Ensure the api package is importable
RUN touch /app/api/__init__.py

EXPOSE 8000

ENV PYTHONPATH="/app"

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# serve run resolves api.serve_app:api_node via PYTHONPATH=/app
CMD ["serve", "run", "api.serve_app:api_node", "--host", "0.0.0.0", "--port", "8000"]