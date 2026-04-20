# api/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Lightweight FastAPI container — NO GPU, NO torch dependency.
# All ML inference is delegated to the model-server container.
# This container now includes libgl1 for Open3D decimation logic.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps (curl for healthcheck, libgl1 for Open3D point cloud processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps only — deliberately minimal
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy utils for decimation handling
COPY utils/ /app/utils/

# Copy API source
COPY api/main.py /app/api/main.py

# Copy pipeline modules needed by scripts/run_pipeline.py
COPY core/ /app/core/
COPY clusterings/ /app/clusterings/
COPY features/ /app/features/
COPY global_descriptors/ /app/global_descriptors/
COPY localizers/ /app/localizers/
COPY matchers/ /app/matchers/
COPY models/ /app/models/
COPY pipelines/ /app/pipelines/
COPY postprocesses/ /app/postprocesses/
COPY preprocesses/ /app/preprocesses/
COPY retrievers/ /app/retrievers/
COPY shortlists/ /app/shortlists/
COPY scripts/ /app/scripts/
COPY conf/ /app/conf/

EXPOSE 8000

ENV PYTHONPATH="/app"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
