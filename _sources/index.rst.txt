.. MLOps 3D Scene Reconstruction documentation master file

MLOps 3D Scene Reconstruction
==============================

.. image:: _static/pipeline.png
   :alt: Model Pipeline Overview
   :align: center

|

**MLOps 3D Scene Reconstruction** is a production-grade AI system that recovers the 3D structure
of real-world environments from a collection of multi-view images. Given a set of photos captured
from different angles, the system estimates each camera's rotation matrix **R** and translation
vector **t** with high accuracy, and renders an interactive 3D point cloud.

The system is built on top of state-of-the-art foundation models (**MASt3R**, **DUSt3R**),
orchestrated through a full MLOps stack comprising **DVC**, **MLflow**, **Airflow**,
**Prometheus**, **Grafana**, and **Docker**.

----

Model Pipeline Overview
------------------------

The end-to-end pipeline transforms a ZIP archive of images into a navigable 3D point cloud:

.. code-block:: text

   ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────┐
   │  Image ZIP  │ →  │ Preprocessing│ →  │  Matching  │ →  │ Triangulation│ →  │ 3D Model │
   │  (Upload)   │    │  + Filtering │    │  (MASt3R)  │    │  (COLMAP)    │    │  (.ply)  │
   └─────────────┘    └──────────────┘    └────────────┘    └──────────────┘    └──────────┘

Each stage is tracked via MLflow, versioned with DVC, and monitored with Prometheus.

----

Downstream Applications
------------------------

- Augmented and Virtual Reality (AR/VR)
- Robotics and Autonomous Driving
- Cultural Heritage Digitization
- Surveying and Topography

----

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: User Interface

   ui_guide

.. toctree::
   :maxdepth: 2
   :caption: System Reference

   architecture
   pipeline
   api

.. toctree::
   :maxdepth: 2
   :caption: MLOps & Operations

   security
   datasources

.. toctree::
   :maxdepth: 2
   :caption: Project Context

   business_understanding
   faq

----

Quick Links
-----------

- **Frontend UI**: http://localhost:5173
- **API Gateway**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3001
- **Prometheus**: http://localhost:9090
