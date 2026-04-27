Pipeline Documentation
======================

The reconstruction pipeline transforms a collection of unordered images into a
sparse 3D point cloud and a set of camera poses. This page explains each stage
in the pipeline, both the **offline DVC training pipeline** and the **online
inference pipeline**.

----

Pipeline Overview
-----------------

.. code-block:: text

   ┌──────────────┐    ┌───────────────┐    ┌─────────────┐    ┌────────────────┐
   │   validate   │ →  │ eda_baselines │ →  │   preprocess│ →  │    prepare     │
   │  (data QC)   │    │ (EDA + stats) │    │  (images)   │    │ (input CSV)    │
   └──────────────┘    └───────────────┘    └─────────────┘    └────────────────┘
                                                                        │
                                                                        ▼
                                                               ┌────────────────┐
                                                               │  run_pipeline  │
                                                               │  (MASt3R +     │
                                                               │   COLMAP SfM)  │
                                                               └────────────────┘
                                                                        │
                                                                        ▼
                                                               ┌────────────────┐
                                                               │    evaluate    │
                                                               │  (mAA + MLflow)│
                                                               └────────────────┘

Each stage is defined in ``dvc.yaml`` and tracked by DVC. Metrics and artifacts
from each run are logged to MLflow.

----

Stage 1 — Data Validation
--------------------------

**DVC stage**: ``validate``

**Script**: ``scripts/validate_data.py``

**What it does**

Reads ``data/train_labels.csv`` and verifies that every image listed in the CSV
exists on disk under ``data/train/``. It reports:

- Total rows in the labels file
- Number of distinct images and scenes
- Missing files
- Duplicate image entries
- Malformed rotation matrices or translation vectors

**Outputs**

- ``data/validation/validation_report.json`` — full issue report
- ``data/validation/validation_metrics.json`` — DVC metric file with ``issue_count``
  and ``status_code``

**Acceptance threshold**

A ``status_code`` of ``0`` means all files are present and valid. A value of ``1``
means warnings exist (e.g., missing files) but the pipeline can continue.
A value of ``2`` indicates a critical error that halts downstream stages.

----

Stage 2 — Exploratory Data Analysis and Baselines
---------------------------------------------------

**DVC stage**: ``eda_baselines``

**Script**: ``scripts/eda_baselines.py``

**What it does**

Computes image statistics across the training dataset to establish the **drift
baseline**. These baselines are later used by the drift monitor to detect when
production images differ from training data. Statistics computed include:

- Image resolution distribution (width, height histograms)
- Pairwise image similarity matrix (using global descriptors)
- Sharpness distribution (Laplacian variance)
- Brightness and contrast statistics

**Outputs**

- ``data/baselines/resolution_hist.png``
- ``data/baselines/similarity_matrix.png``
- ``data/baselines/sharpness_hist.png``
- ``data/baselines/eda_baselines.json`` — raw baseline statistics
- ``data/baselines/eda_metrics.json`` — DVC metric summary

----

Stage 3 — Image Preprocessing
-------------------------------

**DVC stage**: ``image_preprocess``

**Script**: ``scripts/image_processing.py``

**Config**: ``conf/preprocess.yaml``

**What it does**

Applies a configurable preprocessing pipeline to each training image:

- **Deblurring** — images with Laplacian variance below ``blurry_threshold`` are
  sharpened or excluded depending on configuration.
- **Orientation normalisation** — corrects image rotation based on EXIF metadata
  or a learned orientation estimator, so all images are upright before matching.

The preprocessing module is designed to be pluggable. Only stages listed in
``conf/preprocess.yaml`` are applied.

**Outputs**

- ``data/processed/images/`` — preprocessed image tree mirroring ``data/train/``
- ``data/processed/preprocess_report.json``
- ``data/processed/preprocess_metrics.json``

----

Stage 4 — Data Preparation
----------------------------

**DVC stage**: ``prepare``

**Script**: ``scripts/prepare_submission.py``

**What it does**

Reads the preprocessed image paths and ``data/train_labels.csv`` to build
``data/prepared/prepared_input.csv``. This CSV is in the IMC2025 submission format
with ``nan`` placeholder values for rotation and translation — these are populated
by the reconstruction stage.

Columns: ``image_id``, ``dataset``, ``scene``, ``image``, ``rotation_matrix``,
``translation_vector``.

----

Stage 5 — Scene Reconstruction (Core Pipeline)
-----------------------------------------------

**DVC stage**: ``run_pipeline``

**Script**: ``scripts/reconstruct_scenes.py``

**Config**: ``conf/mast3r.yaml`` (or ``conf/best_config.yaml`` in production)

This is the main computational stage. It implements the full
``IMC2025Pipeline.run()`` loop.

Shortlist Generation
~~~~~~~~~~~~~~~~~~~~~

Before matching, the pipeline generates a **shortlist** of candidate image pairs
to match. Matching all N×N pairs is computationally infeasible for large datasets,
so the shortlist generator selects the most promising pairs using an **ensemble**
of global descriptor retrievers:

1. **MASt3R-ASMK** — vocabulary-tree-based retrieval using MASt3R's dense
   descriptors and the ASMK aggregation method. This is the primary retriever.
2. **MASt3R-SPoC** — an alternative global descriptor from the MASt3R retrieval
   head.
3. **DINOv2** — a general-purpose vision transformer used as a secondary global
   descriptor for cross-domain robustness.
4. **ISC** — a descriptor trained specifically for image copy detection, effective
   for repeated structures.

Each retriever proposes its top-K most similar images per query. The union of all
proposals forms the final shortlist.

Feature Extraction and Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each pair in the shortlist, the pipeline runs matching via the **MASt3R Hybrid
Matcher** (``type: mast3r_hybrid``), which combines:

- **Dense matching** — MASt3R's end-to-end dense correspondence network operates
  at 512 px resolution and produces dense pixel-level matches.
- **Sparse matching** — two local feature detectors provide complementary keypoints:

  - **ALIKED** (with LightGlue) — a learned keypoint detector with
    up to 4096 keypoints per image at 1280 px resolution.
  - **MagicLeap SuperPoint** — a classical-style detector with up to 4096
    keypoints at 1600 px resolution.

Dense and sparse matches are **fused** late in the pipeline to maximise coverage.

COLMAP Incremental SfM
~~~~~~~~~~~~~~~~~~~~~~~

Fused matches are imported into a **COLMAP** database. COLMAP's incremental
Structure-from-Motion mapper then:

1. Selects an initial image pair with good homography overlap.
2. Triangulates an initial 3D point set.
3. Registers remaining images one by one via PnP.
4. Runs bundle adjustment after each batch of registrations.
5. Filters outlier points by reprojection error.

Key COLMAP parameters (from config):

- ``mapper_min_model_size: 3`` — minimum images to form a valid reconstruction.
- ``mapper_max_num_models: 25`` — maximum number of disconnected sub-models.

**Outputs**

- ``data/reconstruction/eval_prediction.csv`` — IMC2025 format poses
- ``data/reconstruction/sparse_reconstruction.ply`` — point cloud
- ``data/reconstruction/reconstruction_metrics.json``

----

Stage 6 — Evaluation
---------------------

**DVC stage**: ``evaluate``

**Script**: ``scripts/evaluate.py``

**What it does**

Computes the **mAA (mean Average Accuracy)** metric, which is the primary quality
measure for the IMC2025 competition. mAA measures the fraction of camera poses
registered within a set of angular and translation error thresholds.

It also computes:

- Per-dataset scores and mAA values
- Clusterness score (how well images cluster geometrically)
- Registration rate

All metrics are logged as a child MLflow run under the parent DVC run.

**Outputs**

- ``data/evaluation/metrics.json``
- ``data/evaluation/git_status.txt``

----

Online Inference Pipeline
--------------------------

The online pipeline (triggered via ``POST /upload``) mirrors the DVC pipeline but
runs directly without DVC:

1. ZIP extraction → temporary workspace
2. MASt3R hybrid matching on GPU worker (``GPUModelWorker.reconstruct()``)
3. COLMAP SfM in the same temporary workspace
4. PLY export via ``pycolmap.Reconstruction.export_PLY()``
5. Voxel downsampling (``utils/decimate.py``) to ≤500,000 points
6. Results persisted to ``/app/results/``

The pipeline configuration is loaded from ``conf/best_config.yaml`` (if present)
with fallback to ``conf/mast3r.yaml``.

----

Model Selection and Promotion
------------------------------

After each DVC experiment run, ``scripts/select_best_run.py`` queries MLflow for
the run with the highest ``mAA_overall`` metric in the
``scene_reconstruction_dvc`` experiment. It copies that run's configuration to
``conf/best_config.yaml``, which becomes the active production config on the
next ``ray-serve`` restart.

----

Drift Monitoring and Retraining
---------------------------------

The ``DriftMonitor`` class (``scripts/drift_monitor.py``) compares production image
statistics to the baselines in ``data/baselines/eda_baselines.json``. It checks:

- Mean brightness
- Mean contrast
- Mean sharpness
- Aspect ratio

If any metric drifts beyond the configured threshold, an alert is raised. The
Airflow ``drift_detection_dag`` polls Prometheus every 30 minutes for the
``feature_drift_status`` metric. If drift is detected, it sends an email alert
to the configured ``SMTP_USER``. High-severity drift additionally triggers
``experiment_pipeline_dag`` automatically via the Alertmanager webhook.
