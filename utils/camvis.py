from pathlib import Path
from typing import Optional
import json
import shutil

import numpy as np
from pipelines.scene import Scene


def save_camera_debug_info(pose_dict: dict, scene: Scene, save_dir: Path,
                           prefix_dict: Optional[dict] = None):
    prefix_dict = prefix_dict or {}
    save_dir.mkdir(parents=True, exist_ok=True)
    im_dir = save_dir / "images"
    if im_dir.exists():
        for f in im_dir.glob('*'):
            if f.is_file():
                f.unlink()
        im_dir.rmdir()
    im_dir.mkdir(parents=True, exist_ok=True)

    imgpaths = []
    for key in pose_dict.keys():
        path = scene.short_key_to_image_path(key)   # original file location
        imgpaths.append(im_dir / to_debug_image_name(key, prefix_dict, scene))
        shutil.copy(path, imgpaths[-1])

    json_file = save_dir / "poses.json"
    with open(json_file, "w") as fp:
        fp.write(
            json.dumps(
                {
                    "type": "c2w",
                    "frames": [
                        {
                            "image_name": path.name,
                            "pose": to_trf(
                                pose_dict[key]["R"], pose_dict[key]["t"]
                            ).tolist(),
                        }
                        for key, path in zip(pose_dict.keys(), imgpaths)
                    ],
                }
            )
        )


def to_debug_image_name(output_key: str, prefix_dict: dict, scene: Scene) -> str:
    path = scene.short_key_to_image_path(output_key)
    prefix = prefix_dict.get(output_key)
    if prefix is None:
        return Path(path).name
    debug_name = f"{prefix}_{Path(path).name}"
    return debug_name


def to_trf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    trf = np.eye(4)
    trf[:3, :3] = R
    trf[:3, 3] = t
    return trf
