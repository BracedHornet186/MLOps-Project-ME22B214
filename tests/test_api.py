import pytest
import os
import io
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile, HTTPException

from api.serve_app import fastapi_app, APIGateway, JobStage

client = TestClient(fastapi_app)

def test_login_success(monkeypatch):
    # Mock Docker secrets load
    monkeypatch.setenv("API_USERNAME", "admin")
    monkeypatch.setenv("API_PASSWORD_HASH", "") # Default logic hashes 'admin'
    
    response = client.post("/auth/token", json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_credentials():
    response = client.post("/auth/token", json={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

@pytest.mark.asyncio
async def test_upload_endpoint():
    # Unit test the upload method without starting Ray Serve
    mock_gpu_worker = AsyncMock()
    mock_gpu_worker.process_reconstruction.remote.return_value = "dummy_ref"
    
    gateway = APIGateway(gpu_worker_handle=mock_gpu_worker)
    
    # Mock an uploaded zip file
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "test_images.zip"
    
    # Create a valid zip file in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image1.jpg", b"fake_image_data")
    mock_file.read.return_value = buf.getvalue()
    
    response = await gateway.upload(file=mock_file, user="admin")
    
    assert "job_id" in response
    job_id = response["job_id"]
    
    assert job_id in gateway.job_manager.jobs
    assert gateway.job_manager.jobs[job_id].stage == JobStage.QUEUED
    
    mock_gpu_worker.process_reconstruction.remote.assert_called_once()

@pytest.mark.asyncio
async def test_download_job_ply(tmp_path, monkeypatch):
    mock_gpu_worker = AsyncMock()
    gateway = APIGateway(gpu_worker_handle=mock_gpu_worker)
    
    job_id = "test-job-123"
    gateway.job_manager.create_job(job_id)
    
    import api.serve_app
    monkeypatch.setattr(api.serve_app, "RESULTS_DIR", tmp_path)
    
    test_file = tmp_path / "test_model.ply"
    test_file.write_text("dummy ply content")
    
    # Call download directly
    response = await gateway.download_job_ply(job_id=job_id, filename="test_model.ply", user="admin")
    
    # FileResponse handles paths directly in fastapi
    assert response.path == str(test_file)
    
    with pytest.raises(HTTPException) as excinfo:
        await gateway.download_job_ply(job_id=job_id, filename="missing.ply", user="admin")
    assert excinfo.value.status_code == 404

@pytest.mark.asyncio
async def test_insights_endpoint(tmp_path, monkeypatch):
    mock_gpu_worker = AsyncMock()
    gateway = APIGateway(gpu_worker_handle=mock_gpu_worker)
    
    import api.serve_app
    monkeypatch.setattr(api.serve_app, "RESULTS_DIR", tmp_path)
    
    # Create dummy drift history JSONL
    drift_dir = tmp_path / "monitoring"
    drift_dir.mkdir(parents=True)
    drift_file = drift_dir / "drift_history.jsonl"
    
    import json
    drift_file.write_text(json.dumps({"job_id": "job1", "has_drift": False, "registration_rate": 0.85}) + "\n")
    
    response = await gateway.insights(user="admin")
    
    assert "metrics" in response
    assert "history" in response
    assert len(response["history"]) == 1
    assert response["history"][0]["job_id"] == "job1"
