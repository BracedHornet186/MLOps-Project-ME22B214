# model_server/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# GPU-capable container for the model server.
# Loads MASt3R, ALIKED, DINOv2, ISC weights at startup.
# Workers=1 enforced — owns the GPU exclusively.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        curl git build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Python deps — torch with CUDA 12.6 support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.0 \
        mlflow==2.13.0 \
        httpx==0.27.0 \
        pydantic>=2.0 \
        pyyaml>=6.0 \
        pandas>=2.0 \
        "numpy<2" \
        opencv-python-headless>=4.8 \
        pycolmap>=3.11.1 \
        kornia>=0.8.0 \
        Pillow>=10.0 \
        tqdm>=4.60 \
        scipy>=1.10

# Copy project source
COPY scripts/ /app/scripts/
COPY pipelines/ /app/pipelines/
COPY global_descriptors/ /app/global_descriptors/
COPY preprocesses/ /app/preprocesses/
COPY postprocesses/ /app/postprocesses/
COPY features/ /app/features/
COPY conf/ /app/conf/
COPY model_server/server.py /app/server.py

EXPOSE 8001

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "server.py"]
