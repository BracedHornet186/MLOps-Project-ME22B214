"""
utils/decimate.py
─────────────────────────────────────────────────────────────────────────────
Point cloud decimation utility for optimising .ply payloads before
sending them to the client's browser (WebGL).

Supports two back‑ends:
  1. Open3D  — preferred (fast C++ voxel grid)
  2. trimesh — fallback (pure‑Python random sampling)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PointCloudStats:
    """Lightweight summary of a point cloud file."""
    n_points: int
    bbox_min: list[float]
    bbox_max: list[float]
    has_colors: bool
    file_size_bytes: int


def get_point_cloud_stats(ply_path: str | Path) -> PointCloudStats:
    """Return basic statistics about a .ply point cloud file."""
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    if ply_path.suffix.lower() != ".ply":
        raise ValueError(f"Expected a .ply file, got: {ply_path.suffix}")

    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points)
        has_colors = pcd.has_colors()
    except ImportError:
        import trimesh

        mesh = trimesh.load(str(ply_path))
        pts = np.asarray(mesh.vertices)
        has_colors = hasattr(mesh, "colors") and mesh.colors is not None

    return PointCloudStats(
        n_points=len(pts),
        bbox_min=pts.min(axis=0).tolist() if len(pts) else [0, 0, 0],
        bbox_max=pts.max(axis=0).tolist() if len(pts) else [0, 0, 0],
        has_colors=has_colors,
        file_size_bytes=ply_path.stat().st_size,
    )


# ──────────────────────────────────────────────────────────────────────────
# Core decimation
# ──────────────────────────────────────────────────────────────────────────

def voxel_downsample_ply(
    input_path: str | Path,
    output_path: str | Path,
    voxel_size: float = 0.02,
    max_points: Optional[int] = 500_000,
) -> Path:
    """
    Downsample a .ply point cloud via voxel grid filtering.

    Parameters
    ----------
    input_path : path to the source .ply file
    output_path : path where the decimated .ply will be written
    voxel_size : edge length of each voxel cell (metres).
                 Larger → fewer points, faster browser rendering.
    max_points : hard cap on the number of output points.
                 After voxel downsampling, if the cloud is still larger
                 than this, a uniform random subsample is applied.

    Returns
    -------
    Path to the output file.

    Raises
    ------
    FileNotFoundError  if the input file does not exist
    ValueError         if the input file is not .ply
    RuntimeError       if both open3d *and* trimesh are missing
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")
    if input_path.suffix.lower() != ".ply":
        raise ValueError(f"Expected .ply file, got: {input_path.suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Strategy 1: Open3D (preferred) ────────────────────────────────
    try:
        return _downsample_open3d(input_path, output_path, voxel_size, max_points)
    except ImportError:
        log.info("Open3D not available — falling back to trimesh")

    # ── Strategy 2: trimesh fallback ──────────────────────────────────
    try:
        return _downsample_trimesh(input_path, output_path, max_points)
    except ImportError:
        raise RuntimeError(
            "Neither open3d nor trimesh is installed. "
            "Install at least one: pip install open3d  OR  pip install trimesh"
        )


def _downsample_open3d(
    src: Path, dst: Path, voxel_size: float, max_points: Optional[int]
) -> Path:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(src))
    n_before = len(pcd.points)
    log.info("Open3D: loaded %d points from %s", n_before, src.name)

    # Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size)
    n_after_voxel = len(pcd.points)
    log.info("Open3D: voxel(%.4f) → %d points", voxel_size, n_after_voxel)

    # Hard‑cap random subsample
    if max_points and n_after_voxel > max_points:
        indices = np.random.choice(n_after_voxel, max_points, replace=False)
        pcd = pcd.select_by_index(indices.tolist())
        log.info("Open3D: random subsample → %d points", max_points)

    o3d.io.write_point_cloud(str(dst), pcd, write_ascii=False)
    log.info("Open3D: wrote decimated cloud to %s", dst)
    return dst


def _downsample_trimesh(
    src: Path, dst: Path, max_points: Optional[int]
) -> Path:
    import trimesh

    cloud = trimesh.load(str(src))
    pts = np.asarray(cloud.vertices)
    n_before = len(pts)
    log.info("trimesh: loaded %d points from %s", n_before, src.name)

    target = max_points or 500_000
    if n_before > target:
        idx = np.random.choice(n_before, target, replace=False)
        pts = pts[idx]
        colors = None
        if hasattr(cloud, "colors") and cloud.colors is not None:
            colors = np.asarray(cloud.colors)[idx]
        cloud = trimesh.PointCloud(pts, colors=colors)
        log.info("trimesh: random subsample → %d points", target)

    cloud.export(str(dst))
    log.info("trimesh: wrote decimated cloud to %s", dst)
    return dst
