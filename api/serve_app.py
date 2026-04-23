import os
import shutil
import time
import yaml
import zipfile
from pathlib import Path
from typing import Optional

import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import torch

# Import the actual pipeline logic (from reconstruct_scenes.py)
from scripts.config import PipelineConfig
from scripts.pipeline import create_pipeline

app = FastAPI(title="Scene3D Ray Serve API")

# ─────────────────────────────────────────────────────────────────────────────
# 1. The GPU Inference Worker (Replaces server.py)
# ─────────────────────────────────────────────────────────────────────────────
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 2} # Locks 1 GPU exclusively
)
class GPUModelWorker:
    def __init__(self):
        """Runs ONCE at startup. Keeps models permanently hot in VRAM."""
        print("Initializing ML Pipeline and loading weights into VRAM...")
        config_path = Path("conf/best_config.yaml")
        
        with open(config_path) as f:
            self.pipeline_conf = PipelineConfig(**yaml.safe_load(f))
            
        # This matches reconstruct_scenes.py exactly
        self.pipe = create_pipeline(self.pipeline_conf)
        print("Models loaded successfully!")

    def reconstruct(self, scene_dir: str) -> Optional[str]:
        """Runs the pure compute math on the GPU."""
        scene_path = Path(scene_dir)
        
        try:
            # Run the pipeline just like reconstruct_scenes.py
            pipe_out = self.pipe.run(scene_dir=scene_path)
            
            # Export PLY if available
            persist_ply = scene_path / "sparse_reconstruction.ply"
            if "reconstruction" in pipe_out and pipe_out["reconstruction"]:
                best_rec_dir = pipe_out["reconstruction"][0]
                from scripts.colmap import pycolmap
                pycolmap.Reconstruction(str(best_rec_dir)).export_PLY(str(persist_ply))
                return str(persist_ply)
            return None
            
        finally:
            # Clean up GPU memory safely since the process stays alive
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 2. The CPU API Gateway (Replaces main.py)
# ─────────────────────────────────────────────────────────────────────────────
@serve.deployment(
    num_replicas=2, # Scale up API workers without needing more GPUs
    ray_actor_options={"num_gpus": 0, "num_cpus": 1}
)
@serve.ingress(app)
class APIGateway:
    def __init__(self, gpu_worker_handle):
        # Ray injects the connection to the GPU worker here
        self.gpu_worker = gpu_worker_handle

    @app.post("/infer")
    async def infer_endpoint(self, file: UploadFile = File(...)):
        """Handles I/O, unzipping, and delegates to the GPU worker."""
        
        # 1. CPU Task: Setup directories and Unzip
        job_id = "job_" + str(time.time()).replace(".", "")
        extract_dir = Path(f"/tmp/scene3d_jobs/{job_id}/images")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = extract_dir.parent / file.filename
        content = await file.read()
        zip_path.write_bytes(content)
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # 2. GPU Task: Call the GPU worker over Ray RPC
        # The .remote() call dispatches the work to the GPU container
        ply_result_path = await self.gpu_worker.reconstruct.remote(str(extract_dir))
        
        if not ply_result_path:
            return {"status": "failed", "message": "No reconstruction generated."}

        # 3. CPU Task: Post-process (Decimation)
        # You can run your voxel_downsample_ply here on the CPU
        
        return {"status": "success", "ply_path": ply_result_path}

# ─────────────────────────────────────────────────────────────────────────────
# Bind the Graph Together
# ─────────────────────────────────────────────────────────────────────────────
gpu_node = GPUModelWorker.bind()
api_node = APIGateway.bind(gpu_node)