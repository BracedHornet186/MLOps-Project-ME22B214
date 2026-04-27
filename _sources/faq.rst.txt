Frequently Asked Questions
==========================

----

General
--------

**What does this system actually do?**
   It takes a collection of photos of a scene (e.g., a building, a room, an
   archaeological site) and reconstructs a 3D point cloud from them. For each
   image it also estimates where the camera was positioned and which direction it
   was pointing.

**What kind of images can I use?**
   The system accepts ``.jpg``, ``.jpeg``, ``.png``, ``.tif``, ``.tiff``,
   ``.bmp``, and ``.webp`` files, packaged into a single ZIP archive. Images
   should be taken with a real camera or phone; synthetic renders or heavily
   edited images may produce poor results.

**How many images do I need?**
   A minimum of 3 images is required to form a reconstruction. In practice,
   at least 10–20 overlapping images of a scene will give usable results.
   More images (50–200) generally improve accuracy and coverage.

**Do the images need to be ordered?**
   No. The system handles unordered collections. However, every image must share
   some visual overlap with at least one other image in the set.

----

Installation & Setup
---------------------

**I get "CUDA error" when starting ray-serve. What should I do?**
   Ensure the NVIDIA Container Toolkit is installed on your host:

   .. code-block:: bash

      sudo apt-get install -y nvidia-container-toolkit
      sudo systemctl restart docker

   Then verify GPU access:

   .. code-block:: bash

      docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi

**The ``generate_secrets.sh`` script fails.**
   Ensure you have ``openssl`` installed (``sudo apt install openssl``). The
   script generates random secret strings using ``openssl rand``.

**I see "AIRFLOW_UID not set" warnings.**
   Add your user ID to ``.env``:

   .. code-block:: bash

      echo "AIRFLOW_UID=$(id -u)" >> .env

**Git LFS download fails.**
   You can download model weights manually using the URLs listed in
   :doc:`installation`. Place the files in ``extra/pretrained_models/``.

----

Using the API
--------------

**My JWT token keeps expiring mid-workflow.**
   Tokens are valid for 15 minutes. For long-running automation scripts, refresh
   the token proactively by calling ``POST /auth/token`` before each request, or
   increase ``JWT_EXPIRY_SECONDS`` in the environment configuration.

**I get HTTP 503 on ``/ready``.**
   The GPU worker is still loading. MASt3R model weights take 1–3 minutes to
   load into VRAM at container startup. Wait for the ``ray-serve`` healthcheck
   to pass before sending inference requests.

**``POST /upload`` returns HTTP 413.**
   Your ZIP file exceeds the 500 MB default limit. Either reduce the dataset
   size or increase ``SCENE3D_MAX_UPLOAD_MB`` in ``docker-compose.yaml``.

**How do I run multiple jobs in parallel?**
   The system is currently configured for one concurrent job (``MAX_CONCURRENT_JOBS=1``)
   to prevent GPU memory exhaustion. Additional uploads will be queued and
   processed in order.

----

Reconstruction Quality
-----------------------

**My registration rate is below 50%. What went wrong?**
   Low registration rates are usually caused by one or more of:

   - Images without sufficient overlap (each image should share at least 20–30%
     of its view with neighbouring images).
   - Images that are too blurry (Laplacian variance below threshold).
   - Scenes with repetitive textures where feature matching produces false positives.
   - Very few images (fewer than 10 in a connected scene).

**Some images appear in the point cloud viewer but others don't.**
   Images that appear are those successfully registered by COLMAP. Excluded images
   did not have enough verified matches to determine their pose. Check the
   ``registration_rate`` value in the Stats Table for the proportion registered.

**The point cloud looks very sparse.**
   The displayed point cloud is voxel-downsampled to at most 500,000 points for
   browser performance. The original full-density PLY files are available for
   download and will be much denser.

**I uploaded 200 images but got only 1 cluster with 30 images.**
   COLMAP may have produced multiple disconnected sub-models and selected only the
   largest. This can happen when images fall into groups with little overlap between
   them. Try ensuring all images share some visual context, or increase the number
   of images from each viewpoint.

----

MLOps / DVC / MLflow
---------------------

**How do I compare two experiment runs?**
   Open MLflow at http://localhost:5000, navigate to the
   ``scene_reconstruction_dvc`` experiment, and select multiple runs to compare.
   You can plot ``mAA_overall``, ``registration_rate``, and per-dataset metrics
   side by side.

**How is the best config selected?**
   ``scripts/select_best_run.py`` queries the MLflow tracking server for the run
   with the highest ``mAA_overall`` metric in the experiment. It copies that run's
   config YAML to ``conf/best_config.yaml``. The ``ray-serve`` container reads
   this file at startup.

**DVC repro says "nothing changed". How do I force a re-run?**
   .. code-block:: bash

      dvc repro --force

   Or invalidate a specific stage:

   .. code-block:: bash

      dvc repro --force run_pipeline

**Where are MLflow artifacts stored?**
   Artifacts are stored in the ``mlflow-artifacts`` Docker volume, mounted at
   ``/opt/mlflow/artifacts`` inside the ``mlflow`` container.

----

Monitoring & Alerts
--------------------

**I'm not receiving drift alert emails.**
   Check that your SMTP credentials are correctly set in ``.env``
   (``SMTP_USER``, ``SMTP_PASSWORD``, ``SMTP_MAIL_FROM``). Verify the Airflow
   connection is active at http://localhost:8080/connection/list.

**Grafana shows "No data" for most panels.**
   The ``ray-serve`` service must be running (``inference`` Docker Compose profile)
   for Prometheus to scrape metrics. Start it with:

   .. code-block:: bash

      docker compose --profile inference up -d ray-serve

**Alertmanager is firing but Airflow retraining DAG is not triggered.**
   Check Alertmanager logs: ``docker compose logs alertmanager``.
   Verify the Airflow API is reachable from within the ``mlops_net`` network:

   .. code-block:: bash

      docker compose exec alertmanager wget -qO- http://airflow-apiserver:8080/api/v2/monitor/health

----

Data & Drift
-------------

**What does "drift detected" mean for my results?**
   It means the statistical properties of your uploaded images differ from the
   training dataset. The reconstruction will still run, but accuracy may be lower.
   High-severity drift triggers an automatic Airflow retraining job.

**How do I update the drift baselines?**
   Re-run the ``eda_baselines`` DVC stage with your new dataset:

   .. code-block:: bash

      dvc repro eda_baselines --force

   This regenerates ``data/baselines/eda_baselines.json``, which the drift monitor
   uses as the new reference.
