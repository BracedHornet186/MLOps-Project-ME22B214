## Problem statement
Reconstruct the 3D environment (camera poses R, t) from
unstructured multi-view images taken by handheld phones,
warehouse drones, or vehicle-mounted cameras.

## ML metric
mAA (mean Average Accuracy) — fraction of cameras
registered within translation thresholds.  Target ≥ 50%.

## Business / operational metrics
- End-to-end latency per scene  ≤ 5 min (dual GPU)
- API /health response time      ≤ 200 ms
- Registration rate              ≥ 90 % of images placed

## Data source
IMC 2025 Kaggle dataset + custom mobile-camera images.