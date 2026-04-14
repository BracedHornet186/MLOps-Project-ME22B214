from __future__ import annotations

from pathlib import Path

import torch
from mast3r.model import AsymmetricMASt3R

global_model_caches = {}


def get_mast3r_model(model_path: str | Path, device: torch.device) -> AsymmetricMASt3R:
    if "mast3r" in global_model_caches:
        print("Return MASt3R model from the cache")
        return global_model_caches["mast3r"]
    model = AsymmetricMASt3R.from_pretrained(str(model_path)).eval().to(device)
    global_model_caches["mast3r"] = model
    return model
