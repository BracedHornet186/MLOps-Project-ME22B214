Usage Guide
===========

This guide explains how to run the system, submit data for reconstruction, monitor jobs,
and call the API programmatically.

----

Running the System
------------------

Docker (Production)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker compose --profile inference up -d

All services start automatically. Wait until ``ray-serve`` reports healthy:

.. code-block:: bash

   docker compose ps
   # ray-serve should show "healthy"

Native (Development)
~~~~~~~~~~~~~~~~~~~~~

Open two terminals:

**Terminal 1 — Backend**

.. code-block:: bash

   source .venv/bin/activate
   export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
   cd api
   ray start --head --dashboard-host=0.0.0.0
   serve run serve_app:api_node

**Terminal 2 — Frontend**

.. code-block:: bash

   cd frontend
   npm install
   npm run dev
   # → http://localhost:5173

----

Running the Offline DVC Pipeline
----------------------------------

The DVC pipeline is used for training, evaluation, and experiment tracking.
Run it manually or trigger it via Airflow.

**Manual execution**

.. code-block:: bash

   # Start an MLflow parent run to group all DVC child runs
   PARENT_ID=$(python scripts/start_parent_dvc_run.py | head -n 1)
   MLFLOW_PARENT_RUN_ID="$PARENT_ID" dvc repro

   # Promote the best-performing run to production config
   python scripts/select_best_run.py

This writes the winning configuration to ``conf/best_config.yaml``, which the
``ray-serve`` container reads at startup.

**Via Airflow**

Navigate to http://localhost:8080, find the ``experiment_pipeline_dag``, and
trigger it manually. The DAG runs ``dvc repro`` inside an ephemeral Docker
container and then calls ``select_best_run.py`` automatically.

----

Uploading Data via the UI
--------------------------

1. Open http://localhost:5173 in your browser.
2. Log in with username ``admin`` and password ``admin`` (change in production).
3. Click **Upload Images** and select a ZIP archive containing your images.
4. Monitor the processing stages in the Stage Tracker panel.
5. Once complete, the 3D point cloud appears in the viewer on the right.

See :doc:`ui_guide` for a full step-by-step walkthrough with all panels explained.

----

API Authentication
-------------------

All data endpoints require a JWT Bearer token. Obtain one at ``/auth/token``:

.. code-block:: bash

   TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin"}' \
     | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

   echo $TOKEN   # store this for subsequent requests

Tokens expire after **15 minutes**. Re-request a new token when expired.

----

Uploading Data via the API
---------------------------

.. code-block:: bash

   # Upload a ZIP of images and start reconstruction
   curl -X POST http://localhost:8000/upload \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@/path/to/images.zip" \
     -F "dataset_name=my_scene" \
     -F "scene_name=scene_01"

Response:

.. code-block:: json

   {
     "job_id": "3f7a91b2-...",
     "message": "Pipeline started."
   }

----

Polling Job Status
-------------------

.. code-block:: bash

   curl http://localhost:8000/jobs/<job_id> \
     -H "Authorization: Bearer $TOKEN"

Response fields:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - ``stage``
     - Current pipeline stage: ``queued``, ``extracting``, ``matching``, ``triangulating``, ``decimating``, ``success``, or ``failed``
   * - ``progress``
     - Integer 0–100 indicating overall completion percentage
   * - ``message``
     - Human-readable status message
   * - ``registration_rate``
     - Fraction of images successfully registered into the 3D model
   * - ``has_drift``
     - Whether input data drift was detected
   * - ``drift_severity``
     - ``low``, ``moderate``, or ``high``
   * - ``download_url``
     - Populated with a download path once the job succeeds

----

Downloading Results
--------------------

**Download point cloud (ZIP of PLY files)**

.. code-block:: bash

   curl -O http://localhost:8000/download/jobs/<job_id> \
     -H "Authorization: Bearer $TOKEN"

**Download submission CSV**

.. code-block:: bash

   curl -O http://localhost:8000/download/jobs/<job_id>/csv \
     -H "Authorization: Bearer $TOKEN"

----

Checking Data Drift
--------------------

You can check a new image set for drift before triggering reconstruction:

.. code-block:: bash

   curl -X POST http://localhost:8000/drift \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@/path/to/images.zip"

The response contains statistics on brightness, contrast, sharpness, and resolution
compared to the training baseline.

----

Triggering Retraining
----------------------

If drift is detected and you want to retrain the model:

.. code-block:: bash

   curl -X POST http://localhost:8000/drift/trigger-retrain \
     -H "Authorization: Bearer $TOKEN"

This calls the Airflow API to trigger ``experiment_pipeline_dag`` programmatically.

----

Example End-to-End Workflow
----------------------------

.. code-block:: bash

   # 1. Authenticate
   TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"admin"}' \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

   # 2. Upload images
   JOB=$(curl -s -X POST http://localhost:8000/upload \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@my_photos.zip" \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

   # 3. Poll until done
   while true; do
     STATUS=$(curl -s http://localhost:8000/jobs/$JOB \
       -H "Authorization: Bearer $TOKEN" \
       | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['stage'])")
     echo "Stage: $STATUS"
     [ "$STATUS" = "success" ] || [ "$STATUS" = "failed" ] && break
     sleep 10
   done

   # 4. Download results
   curl -O http://localhost:8000/download/jobs/$JOB \
     -H "Authorization: Bearer $TOKEN"

----

Disabling Authentication (Development Only)
--------------------------------------------

To disable JWT checks during local development, set ``AUTH_ENABLED=false`` in the
``ray-serve`` service environment in ``docker-compose.yaml``:

.. code-block:: yaml

   environment:
     AUTH_ENABLED: "false"

.. warning::
   Never disable authentication in production. All data endpoints will be
   publicly accessible without a token.
