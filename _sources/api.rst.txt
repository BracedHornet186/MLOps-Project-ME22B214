API Reference
=============

The API Gateway is a **FastAPI** application served via **Ray Serve** on port ``8000``.
It provides endpoints for authentication, job management, inference, drift monitoring,
and system health.

Base URL: ``http://localhost:8000``

----

Authentication
--------------

Most endpoints require a **JWT Bearer token**. Obtain one via ``POST /auth/token``
and pass it in the ``Authorization: Bearer <token>`` header.

Tokens expire after **15 minutes** (configurable via ``JWT_EXPIRY_SECONDS`` environment
variable). Requests with expired or missing tokens receive ``HTTP 401``.

The following endpoints are **unauthenticated** (infrastructure probes):

- ``GET /health``
- ``GET /ready``
- ``GET /metrics``

----

Endpoint Reference
------------------

POST /auth/token
~~~~~~~~~~~~~~~~~

Obtain a JWT access token.

**Request Body (JSON)**

.. code-block:: json

   {
     "username": "admin",
     "password": "admin"
   }

**Response 200**

.. code-block:: json

   {
     "access_token": "eyJhbGciOiJIUzI1NiIs...",
     "token_type": "bearer",
     "expires_in": 900
   }

**Response 401**

.. code-block:: json

   { "detail": "Invalid credentials" }

**Example**

.. code-block:: bash

   curl -X POST http://localhost:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin"}'

----

GET /health
~~~~~~~~~~~

Basic liveness probe. No authentication required.

**Response 200**

.. code-block:: json

   {
     "status": "ok",
     "version": "2.0.0",
     "timestamp": 1714300000.123
   }

----

GET /ready
~~~~~~~~~~

Readiness probe that pings the GPU worker. No authentication required.

**Response 200**

.. code-block:: json

   {
     "status": "ready",
     "device": "NVIDIA A100 (40.0 GB)"
   }

**Response 503** — returned when the GPU worker has not finished loading model weights.

----

GET /metrics
~~~~~~~~~~~~~

Prometheus metrics endpoint (text/plain). No authentication required.
Scraped automatically by Prometheus every 10 seconds.

Key metrics exposed:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Metric Name
     - Description
   * - ``api_requests_total``
     - Total HTTP requests labelled by method, endpoint, and status
   * - ``api_errors_total``
     - Total 4xx/5xx responses labelled by endpoint
   * - ``inference_latency_seconds``
     - Histogram of end-to-end reconstruction wall-clock time
   * - ``registered_images_ratio``
     - Fraction of images placed in the last reconstruction
   * - ``active_jobs_total``
     - Number of currently running reconstruction jobs
   * - ``model_server_ready``
     - 1 if the GPU worker is ready, 0 otherwise
   * - ``data_valid_images_total``
     - Number of valid images in the current dataset

----

POST /upload
~~~~~~~~~~~~~

Upload a ZIP archive and start a reconstruction job. **Auth required.**

**Request** — ``multipart/form-data``

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``file``
     - File
     - ZIP archive containing images (.jpg, .jpeg, .png, .tif, .tiff, .bmp, .webp)
   * - ``dataset_name``
     - string
     - Logical dataset name (default: ``"custom"``)
   * - ``scene_name``
     - string
     - Logical scene name (default: ``"scene_01"``)

**Response 202**

.. code-block:: json

   {
     "job_id": "3f7a91b2-1234-5678-abcd-ef0123456789",
     "message": "Pipeline started."
   }

**Response 400** — invalid or non-ZIP file.

**Response 413** — upload exceeds the configured size limit.

**Example**

.. code-block:: bash

   curl -X POST http://localhost:8000/upload \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@photos.zip" \

----

GET /status/{job_id}  |  GET /jobs/{job_id}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the status of a reconstruction job. Both paths return identical responses.
**Auth required.**

**Path Parameters**

- ``job_id`` — UUID returned from ``POST /upload``

**Response 200**

.. code-block:: json

   {
     "job_id": "3f7a91b2-...",
     "stage": "matching",
     "status": "matching",
     "progress": 30,
     "message": "Running MASt3R feature matching on GPU …",
     "created_at": 1714300000.0,
     "started_at": 1714300005.0,
     "finished_at": null,
     "n_images": 45,
     "n_points": 0,
     "registration_rate": null,
     "error": null,
     "download_url": null,
     "has_drift": false,
     "drift_severity": "low"
   }

**Stage values** and their progress percentages:

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Stage
     - Progress
     - Meaning
   * - ``queued``
     - 0%
     - Waiting for the pipeline semaphore
   * - ``extracting``
     - 10%
     - Unpacking the ZIP archive
   * - ``matching``
     - 30%
     - Running MASt3R + ALIKED + SuperPoint on GPU
   * - ``triangulating``
     - 70%
     - COLMAP incremental SfM in progress
   * - ``decimating``
     - 85%
     - Voxel downsampling of point cloud
   * - ``success``
     - 100%
     - Reconstruction complete
   * - ``failed``
     - 0%
     - Pipeline error — see ``error`` field

**Response 404** — job ID not found.

----

GET /download/jobs/{job_id}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download all PLY files for a completed job as a ZIP archive. **Auth required.**

**Response 200** — ``application/zip`` containing one or more ``.ply`` files.

**Response 409** — job is not yet complete.

**Response 404** — PLY file not available (reconstruction produced no 3D points).

----

GET /download/jobs/{job_id}/csv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the raw submission CSV in IMC2025 format. **Auth required.**

**Response 200** — ``text/csv``

Columns: ``dataset``, ``scene``, ``image``, ``rotation_matrix``, ``translation_vector``.
Images that could not be registered have semicolon-separated ``nan`` values.

----

GET /download/jobs/{job_id}/{filename}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download a single named PLY file from a completed job. **Auth required.**

- ``filename`` — e.g. ``cluster0_decimated_model0_3f7a91b2.ply``

**Response 200** — ``application/octet-stream``

----

GET /clusters/{job_id}
~~~~~~~~~~~~~~~~~~~~~~~

Retrieve per-cluster reconstruction statistics. **Auth required.**

**Response 200**

.. code-block:: json

   {
     "clusters": [
       {
         "id": 0,
         "name": "cluster0_model0",
         "num_points3D": 124532,
         "filename": "cluster0_decimated_model0_3f7a91b2.ply"
       }
     ]
   }

----

GET /jobs/{job_id}/insights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve consolidated reconstruction and drift insights. **Auth required.**

**Response 200**

.. code-block:: json

   {
     "registration_rate": 0.9333,
     "n_points": 124532,
     "has_drift": false,
     "drift_severity": "low",
     "drift_report": {
       "drift_detected": false,
       "severity": "low",
       "checks": {}
     },
     "recommendation": "No action needed."
   }

The ``recommendation`` field provides a plain-language action suggestion based on
drift severity.

----

POST /drift
~~~~~~~~~~~~

Check a ZIP archive for data drift without starting a reconstruction. **Auth required.**

**Request** — ``multipart/form-data``

- ``file`` — ZIP archive of images

**Response 200** — drift report JSON with per-feature drift flags and severity.

----

POST /drift/trigger-retrain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually trigger the Airflow ``experiment_pipeline_dag`` retraining DAG. **Auth required.**

**Response 200**

.. code-block:: json

   { "status": "triggered" }

**Response 502** — Airflow API is unreachable.

----

Error Responses
---------------

All error responses follow FastAPI's standard format:

.. code-block:: json

   {
     "detail": "Human-readable error message"
   }

Common HTTP status codes:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Code
     - Meaning
   * - 400
     - Bad request (invalid file, malformed input)
   * - 401
     - Missing or expired JWT token
   * - 404
     - Resource not found (job ID, PLY file)
   * - 409
     - Conflict (e.g., download requested before job is done)
   * - 413
     - Upload too large
   * - 500
     - Internal server error in the pipeline
   * - 502
     - Upstream service (Airflow) unreachable
   * - 503
     - GPU worker not ready
