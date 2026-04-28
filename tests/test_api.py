import pytest
import os
import io
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile, HTTPException

from api.serve_app import fastapi_app, APIGateway, JobStage

OriginalAPIGateway = APIGateway.func_or_class.__bases__[0]

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

    gateway = OriginalAPIGateway.__new__(OriginalAPIGateway)
    gateway.gpu_worker = mock_gpu_worker
    gateway.job_manager = MagicMock()
    gateway.api_requests_total = MagicMock()
    gateway._analyze_drift = AsyncMock(return_value={"has_drift": False, "severity": "low"})

    # Mock an uploaded zip file
    mock_file = AsyncMock(spec=UploadFile)
    mock_file.filename = "test_images.zip"

    # Create a valid zip file in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image1.jpg", b"fake_image_data")
    mock_file.read.return_value = buf.getvalue()

    from fastapi import BackgroundTasks
    bg_tasks = BackgroundTasks()
    response = await gateway.upload_zip(background_tasks=bg_tasks, file=mock_file, user="admin")

    assert "job_id" in response
    job_id = response["job_id"]

    gateway.job_manager.create_job.assert_called_once_with(job_id)
    gateway.job_manager.update_job.assert_called_with(
        job_id, JobStage.QUEUED, "Waiting in queue …",
        has_drift=False,
        drift_severity="low",
        drift_report={"has_drift": False, "severity": "low"},
    )

@pytest.mark.asyncio
async def test_download_job_ply(tmp_path, monkeypatch):
    import json as _json

    mock_gpu_worker = AsyncMock()
    gateway = OriginalAPIGateway.__new__(OriginalAPIGateway)
    gateway.gpu_worker = mock_gpu_worker
    gateway.api_errors_total = MagicMock()

    job_id = "test-job-123"

    import api.serve_app
    monkeypatch.setattr(api.serve_app, "RESULTS_DIR", tmp_path)

    test_file = tmp_path / "test_model.ply"
    test_file.write_text("dummy ply content")

    mock_rec = MagicMock()
    mock_rec.stage = JobStage.DONE
    mock_rec.ply_path = _json.dumps([str(test_file)])

    gateway.job_manager = MagicMock()
    gateway.job_manager.jobs = {job_id: mock_rec}
    gateway.job_manager.get_job.return_value = mock_rec

    # Call download directly
    response = await gateway.download_single_ply(job_id=job_id, filename="test_model.ply", user="admin")

    # FileResponse stores the path on .path
    assert response.path == str(test_file)

    with pytest.raises(HTTPException) as excinfo:
        await gateway.download_single_ply(job_id=job_id, filename="missing.ply", user="admin")
    assert excinfo.value.status_code == 404

@pytest.mark.asyncio
async def test_insights_endpoint(tmp_path, monkeypatch):
    mock_gpu_worker = AsyncMock()
    gateway = OriginalAPIGateway.__new__(OriginalAPIGateway)
    gateway.gpu_worker = mock_gpu_worker
    gateway.job_manager = MagicMock()
    gateway.reconstruction_maa = MagicMock()
    
    import api.serve_app
    monkeypatch.setattr(api.serve_app, "RESULTS_DIR", tmp_path)
    
    # Mock the job manager to return a job record
    mock_job = MagicMock()
    mock_job.has_drift = False
    mock_job.drift_severity = "low"
    mock_job.registration_rate = 0.85
    mock_job.n_points = 1000
    mock_job.drift_report = {}
    gateway.job_manager.jobs = {"job1": mock_job}
    gateway.job_manager.get_job.return_value = mock_job
    
    response = await gateway.get_job_insights(job_id="job1", user="admin")
    
    assert "recommendation" in response
    assert response["registration_rate"] == 0.85
    assert response["has_drift"] is False
