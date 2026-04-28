Business Understanding
======================

This page defines the problem the system solves, the metrics used to measure
success, and the operational targets that govern production deployment.

----

Problem Statement
------------------

Given a set of **unordered, unstructured multi-view images** captured by handheld
phones, warehouse drones, or vehicle-mounted cameras, reconstruct the 3D environment
by estimating the **camera pose** (rotation matrix **R** and translation vector
**t**) for each image.

This is the core computer vision task of **Structure-from-Motion (SfM)**. Traditional
approaches rely on hand-crafted features (SIFT, ORB) and geometric verification
(RANSAC). This system replaces the feature extraction and matching steps with
state-of-the-art neural networks (MASt3R, ALIKED, DINOv2) to achieve higher
accuracy on challenging real-world scenes.

----

Use Cases
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Domain
     - Application
   * - **AR / VR**
     - Reconstruct real environments for immersive mixed-reality experiences
   * - **Robotics**
     - Generate scene maps for autonomous navigation and pick-and-place tasks
   * - **Autonomous Driving**
     - 3D mapping and localisation from dashcam imagery
   * - **Cultural Heritage**
     - Digital preservation of historical monuments and artefacts
   * - **Surveying & Topography**
     - Generate georeferenced point clouds from drone surveys
   * - **Inspection**
     - Structural inspection of infrastructure from photo collections

----

ML Metric
----------

The primary quality metric is **mAA (mean Average Accuracy)**:

   The fraction of camera poses registered within a set of angular and translation
   error thresholds defined per scene in ``data/train_thresholds.csv``.

A pose is counted as "accurate" if both its rotation error (in degrees) and its
translation error (normalised by scene scale) fall below the threshold.

**Target**: mAA ≥ 50%

The mAA metric is computed by ``scripts/evaluate.py`` using the official IMC2025
scoring function and logged to MLflow after every experiment run.

----

Business and Operational Metrics
----------------------------------

In addition to model accuracy, the system must meet the following operational targets:

.. list-table::
   :widths: 45 30 25
   :header-rows: 1

   * - Metric
     - Target
     - Measured By
   * - End-to-end latency per scene (dual GPU)
     - ≤ 5 minutes
     - ``inference_latency_seconds`` (Prometheus)
   * - API ``/health`` response time
     - ≤ 200 ms
     - Prometheus scrape
   * - Registration rate (images placed in model)
     - ≥ 90%
     - ``registered_images_ratio`` (Prometheus)

These metrics are monitored continuously via the Grafana dashboard. Alerts fire
in Prometheus and Alertmanager when thresholds are breached, triggering automated
retraining or human review.

----

Data Source
-----------

The system is trained and evaluated on the **IMC 2025 Kaggle dataset**, provided
by the Computer Vision Group (CVG). It is supplemented by an internal
``custom_warehouse`` dataset for industrial use-case validation.

See :doc:`datasources` for full details on dataset composition, biases, and
licensing.

----

Stakeholders
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Role
     - Interest
   * - **ML Engineers**
     - Experiment tracking, model improvement, DVC pipeline management
   * - **Platform Engineers**
     - Uptime, latency, GPU utilisation, Docker deployments
   * - **End Users**
     - Easy upload workflow, fast results, accurate 3D models
   * - **Data Scientists**
     - mAA scores, per-dataset breakdowns, drift monitoring

----

Definition of Done
-------------------

A model version is considered **production-ready** when:

1. ``mAA_overall`` ≥ 0.50 on the training evaluation split.
2. ``registration_rate`` ≥ 0.90 on the standard test scenes.
3. End-to-end inference latency ≤ 5 minutes for a typical 50-image scene.
4. All Trivy and ``pip-audit`` CI checks pass with no CRITICAL/HIGH findings.
5. The configuration is committed to ``conf/best_config.yaml`` and tagged in Git.
