import pytest
from pathlib import Path
from utils.decimate import get_point_cloud_stats, voxel_downsample_ply

def test_get_point_cloud_stats_file_not_found():
    with pytest.raises(FileNotFoundError, match="PLY file not found"):
        get_point_cloud_stats("non_existent_file.ply")

def test_get_point_cloud_stats_invalid_extension(tmp_path):
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("dummy content")
    with pytest.raises(ValueError, match="Expected a .ply file"):
        get_point_cloud_stats(invalid_file)

def test_voxel_downsample_ply_file_not_found():
    with pytest.raises(FileNotFoundError, match="Input PLY not found"):
        voxel_downsample_ply("non_existent_file.ply", "output.ply")

def test_voxel_downsample_ply_invalid_extension(tmp_path):
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("dummy content")
    with pytest.raises(ValueError, match="Expected .ply file"):
        voxel_downsample_ply(invalid_file, "output.ply")
