Installation Guide
==================

This guide covers two installation paths: the recommended **Docker-based** setup and the
**native Python** setup for development.

----

Prerequisites
-------------

- **Docker** >= 24.0 and **Docker Compose** >= 2.20
- **NVIDIA GPU** with CUDA 12.6 support (required for inference)
- **NVIDIA Container Toolkit** installed on the host
- **Git** and **Git LFS**
- At least **16 GB GPU VRAM** and **32 GB system RAM** recommended

----

Step 1 — Clone the Repository
-------------------------------

.. code-block:: bash

   git clone https://github.com/your-org/MLOps-Project-ME22B214.git
   cd MLOps-Project-ME22B214
   git lfs pull          # Downloads pre-trained model weights

----

Step 2 — Download the Dataset
-------------------------------

.. code-block:: bash

   kaggle competitions download -c image-matching-challenge-2025
   unzip image-matching-challenge-2025.zip -d data/
   mv data/image-matching-challenge-2025/* data/
   rm -r data/image-matching-challenge-2025

The ``data/`` directory should now contain ``train/``, ``test/``, ``train_labels.csv``,
and ``train_thresholds.csv``.

----

Docker Setup (Recommended)
---------------------------

This is the standard production-ready setup. All services run as Docker containers
on a shared network (``mlops_net``).

Step 3a — Configure the Environment File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the interactive setup script to generate your ``.env`` file:

.. code-block:: bash

   ./setup_env.sh

The script will prompt you for the following values:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``AIRFLOW_UID``
     - Your host user ID (run ``id -u`` to find it)
   * - ``DOCKER_GID``
     - Docker group ID (run ``getent group docker | cut -d: -f3``)
   * - ``FERNET_KEY``
     - Airflow encryption key (generate with ``python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`` )
   * - ``SMTP_USER``
     - Gmail address for Airflow email alerts
   * - ``SMTP_PASSWORD``
     - Gmail app password (not your regular password)
   * - ``HOST_PROJECT_ROOT``
     - Absolute path to this repository on the host machine

Step 3b — Generate Docker Secrets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./generate_secrets.sh
   chmod 644 ./secrets/*

This creates two files under ``secrets/``:

- ``secrets/jwt_secret`` — used to sign API JWT tokens
- ``secrets/grafana_admin_password`` — used for the Grafana admin account

.. note::
   The ``secrets/`` directory is listed in ``.gitignore`` and will never be committed.

Step 3c — Generate TLS Certificates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ./generate-certs.sh

This creates self-signed certificates under ``certs/`` for nginx TLS termination.
For production, replace these with certificates from a trusted CA.

Step 3d — Clone External Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd extra/
   git clone https://github.com/jenicek/asmk
   git clone https://github.com/naver/croco
   cd ..

Step 3e — Launch the Full Stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker compose --profile inference up --build -d

This starts the following services:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Service
     - Port
     - Description
   * - ``scene3d-ui``
     - 5173, 443
     - React + Three.js frontend
   * - ``ray-serve``
     - 8000, 8265
     - FastAPI gateway + GPU inference worker
   * - ``mlflow``
     - 5000
     - Experiment tracking server
   * - ``airflow-apiserver``
     - 8080
     - Airflow web UI and REST API
   * - ``prometheus``
     - 9090
     - Metrics scraping
   * - ``grafana``
     - 3001
     - Monitoring dashboards
   * - ``postgres``
     - (internal)
     - Airflow metadata database

.. code-block:: bash

   # Verify all containers are healthy
   docker compose ps

Wait for the ``ray-serve`` healthcheck to pass — this can take up to 5 minutes as
MASt3R model weights are loaded into GPU memory.

----

Native Python Setup (Developer Mode)
--------------------------------------

Use this path if you need to develop or debug outside Docker.

Step 3a — Build ASMK
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd extra/
   git clone https://github.com/jenicek/asmk
   cd asmk/cython/
   cythonize *.pyx
   cd ..
   python -m build --no-isolation
   pip install dist/*.whl
   cd ../../

Step 3b — Build CroCo / DUSt3R Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DUSt3R relies on RoPE positional embeddings, which require compiled CUDA kernels:

.. code-block:: bash

   cd extra/
   git clone https://github.com/naver/croco.git
   cd croco/models/curope/
   python -m build --no-isolation
   pip install dist/*.whl
   cd ../../

Step 3c — Build Remaining Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build any additional packages in ``bundle/oss/`` as ``.whl`` files using
``python -m build --no-isolation`` in their respective directories, then move
the compiled ``.whl`` files to ``bundle/oss/``.

Step 3d — Create the Python Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install uv
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

The project requires **Python 3.11** exactly (``requires-python = "==3.11.*"``).

----

Pre-trained Model Weights
--------------------------

Model weights are stored under ``extra/pretrained_models/`` via Git LFS. If you
need to download them manually:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Model
     - Download URL
   * - ALIKED ``aliked-n16.pth``
     - https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth
   * - ISC ``isc_ft_v107.pth.tar``
     - https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar
   * - MASt3R main weights
     - https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
   * - MASt3R retrieval weights
     - https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
   * - MASt3R codebook
     - https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

----

Verifying the Installation
---------------------------

Once all services are running, verify the stack is healthy:

.. code-block:: bash

   # API health check
   curl http://localhost:8000/health

   # GPU worker readiness
   curl http://localhost:8000/ready

   # Obtain a JWT token
   curl -X POST http://localhost:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin"}'

A successful ``/health`` response looks like:

.. code-block:: json

   {
     "status": "ok",
     "version": "2.0.0",
     "timestamp": 1714300000.0
   }

----

Troubleshooting Installation
------------------------------

**``ray-serve`` container exits immediately**
   Check that the NVIDIA Container Toolkit is installed and that
   ``docker run --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`` succeeds.

**Port conflicts**
   If ports 8000, 5000, or 8080 are in use on your host, edit the ``ports:``
   mappings in ``docker-compose.yaml`` before launching.

**Airflow DB migration fails**
   Ensure ``postgres`` is healthy before running ``airflow-init``:
   ``docker compose logs postgres``.

**Git LFS quota exceeded**
   Download model weights manually using the URLs above and place them under
   ``extra/pretrained_models/``.
