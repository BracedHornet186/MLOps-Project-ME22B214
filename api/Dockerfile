# api/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Unified GPU container for Ray Serve deployment graph
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        curl git build-essential \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# ── Python setup ──────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel cython build

# ── PyTorch (CUDA 12.6) ───────────────────────────────────────────────────────
RUN pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu126

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# ── Install custom wheels first  ──────────────────────────────────────────────
COPY bundle/ ./bundle/
RUN pip install $(find ./bundle/oss -name "*.whl")

# Copy source
COPY croco/ /tmp/croco/

WORKDIR /tmp/croco/models/curope

# Build the wheel
RUN python -m build --no-isolation

# Install the freshly built wheel
RUN pip install dist/*.whl

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Copy asmk source
COPY asmk/ /tmp/asmk/

# Cythonize the .pyx files in the cython directory
WORKDIR /tmp/asmk/cython/
RUN cythonize *.pyx

WORKDIR /tmp/asmk/

# Build the wheel
RUN python -m build --no-isolation

# Install the freshly built wheel
RUN pip install dist/*.whl

# ── Copy dependency definition first (for caching) ────────────────────────────
COPY api/pyproject.toml .

# ── Copy full project source ──────────────────────────────────────────────────
COPY scripts/             ./scripts/
COPY pipelines/           ./pipelines/
COPY clusterings/         ./clusterings/
COPY features/            ./features/
COPY global_descriptors/  ./global_descriptors/
COPY localizers/          ./localizers/
COPY matchers/            ./matchers/
COPY models/              ./models/
COPY postprocesses/       ./postprocesses/
COPY preprocesses/        ./preprocesses/
COPY retrievers/          ./retrievers/
COPY shortlists/          ./shortlists/
COPY core/                ./core/
COPY utils/               ./utils/
COPY conf/                ./conf/
COPY extra/               ./extra/

# ── Copy API ──────────────────────────────────────────────────────────────────
COPY api/serve_app.py ./serve_app.py

# Ensure package root
RUN touch /app/__init__.py

# ── Install project ───────────────────────────────────────────────────────────
RUN pip install -e .

# ── Expose API ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start Ray Serve ───────────────────────────────────────────────────────────
CMD ["serve", "run", "serve_app:api_node"]