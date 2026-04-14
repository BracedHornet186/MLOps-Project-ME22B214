from pathlib import Path
from typing import Any, Callable, Optional
import copy

import cv2
import shutil
import numpy as np
import torch
import tqdm
import pycolmap

from models.mickey.models.builder import build_model
from models.mickey.datasets.utils import correct_intrinsic_scale
from models.mickey.models.MicKey.modules.utils.training_utils import colorize, generate_heat_map
from models.mickey.config.default import cfg

from data import FilePath,resolve_model_path
from localizers.base import PostLocalizer, Localizer
from localizers.common import get_default_K
from localizers.config import MicKeyConfig
from pipelines.scene import Scene
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
)
from utils.camvis import save_camera_debug_info



class MicKeyLocalizer(Localizer):
    def __init__(self, conf: MicKeyConfig, device: torch.device) -> None:
        self.device = device
        self.conf = conf
        cfg.merge_from_file(str(resolve_model_path(conf.model.config_path)))
        model = build_model(cfg, checkpoint=str(resolve_model_path(conf.model.weight_path)))
        self.model = model.eval().to(device)
    
    @torch.inference_mode()
    def infer(self, qpath: FilePath, xpath: FilePath, scene: Scene):
        shape0 = scene.get_image_shape(qpath)
        shape1 = scene.get_image_shape(xpath)
        resize = (self.conf.model.size, self.conf.model.size)
        im0 = read_color_image(str(qpath), resize).to(self.device)
        im1 = read_color_image(str(xpath), resize).to(self.device)
        K0 = get_default_K(shape0)
        K1 = get_default_K(shape1)
        if resize is not None:
            K0 = correct_intrinsic_scale(K0, resize[0] / shape0[1], resize[1] / shape0[0]).numpy()
            K1 = correct_intrinsic_scale(K1, resize[0] / shape1[1], resize[1] / shape1[0]).numpy()
        data = {}
        data['image0'] = im0
        data['image1'] = im1
        data['K_color0'] = torch.from_numpy(K0).unsqueeze(0).to(self.device)
        data['K_color1'] = torch.from_numpy(K1).unsqueeze(0).to(self.device)

        self.model(data)
        R = data['R']
        t = data['t']
        inliers = data['inliers']
        return R, t, inliers

    @torch.inference_mode()
    def localize(
        self,
        scene: Scene,
        pairs: list[tuple[int, int]]
    ) -> dict:
        poses = {}
        for pair in pairs:
            i, j = pair
            print(f"pair=({pair})")

            if len(poses) == 0:
                poses = {i: (np.eye(3), np.zeros(3))}
            
            if i is poses and j in poses:
                print(f' -> Skip pair: {pair}')
                continue

            if i not in poses and j not in poses:
                print(f' -> Skip pair: {pair}')
                continue

            if i in poses:
                qid, xid = (j, i)
            else:
                qid, xid = (i, j)

            qpath = scene.image_paths[qid]
            xpath = scene.image_paths[xid]

            R, t, inliers = self.infer(qpath, xpath, scene)
            R = R.cpu().numpy()[0]
            t = t.reshape(-1).cpu().numpy()
            # w2c?

            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                continue

            Rx, tx = poses[xid]
            print(R.shape, Rx.shape, t.shape, tx.shape)
            Rq = np.linalg.inv(R) @ Rx
            tq = (-np.linalg.inv(R) @ t) + tx

            poses[qid] = (Rq, tq)
    
        outputs = {}
        for i, pose in poses.items():
            name = Path(scene.image_paths[i]).name
            output_key = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, name 
            )
            R, t = pose
            print(R, t)
            print(R.shape, t.shape)
            T = np.zeros((3, 4))
            T[:3, :3] = R
            T[:3, 3] = t
            r3 = pycolmap.Rigid3d(T)
            print(r3.rotation.matrix(), r3.translation)
            outputs[output_key] = {
                #'R': copy.deepcopy(R),
                #'t': copy.deepcopy(t)
                'R': copy.deepcopy(r3.rotation.matrix()),
                't': copy.deepcopy(r3.translation)
            }
        
        return outputs



def prepare_score_map(scs, img, temperature=0.5):

    score_map = generate_heat_map(scs, img, temperature)

    score_map = 255 * score_map.permute(1, 2, 0).numpy()

    return score_map

def colorize_depth(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(0, 0, 0, 255), gamma_corrected=False, value_transform=None):

    img = colorize(value, vmin, vmax, cmap, invalid_val, invalid_mask, background_color, gamma_corrected, value_transform)

    shape_im = img.shape
    img = np.asarray(img, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = cv2.resize(img, (shape_im[1]*14, shape_im[0]*14), interpolation=cv2.INTER_LINEAR)

    return img

def read_color_image(path, resize):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
    Returns:
        image (torch.tensor): (3, h, w)
    """
    # read and resize image
    cv_type = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), cv_type)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize is not None:
        image = cv2.resize(image, resize)

    # (h, w, 3) -> (3, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255

    return image.unsqueeze(0)

def read_intrinsics(path_intrinsics, resize):
    Ks = {}
    with Path(path_intrinsics).open('r') as f:
        for line in f.readlines():
            if '#' in line:
                continue

            line = line.strip().split(' ')
            img_name = line[0]
            fx, fy, cx, cy, W, H = map(float, line[1:])

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            if resize is not None:
                K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H).numpy()
            Ks[img_name] = K
    return Ks