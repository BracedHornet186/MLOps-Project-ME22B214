from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import tqdm
import yaml
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from models.fsnet.FSNet_model import FSNet_model, FSNet_model_handler
from models.fsnet.modules.utils import ImagePairLoader
from postprocesses.config import FSNetConfig, RANSACConfig


class PILCompatLoader:
    def __init__(self, reader: Callable):
        self.reader = reader

    def __call__(self, path: str) -> Image.Image:
        img = self.reader(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class FSNet:
    def __init__(self,
                 conf: FSNetConfig,
                 image_reader: Callable,
                 device: torch.device):
        model_config_path = resolve_model_path(conf.config_path)
        weight_path = resolve_model_path(conf.weight_path)
        print(f'FSNet: config={model_config_path}, weights={weight_path}')

        with open(model_config_path, 'r') as fp:
            model_conf = yaml.safe_load(fp)
        
        fsnet = FSNet_model_handler(
            model_conf, weight_path, device
        )

        im_size = model_conf['ENCODER']['IM_SIZE']
        loader = ImagePairLoader(im_size, device)
        loader.loader = PILCompatLoader(image_reader)

        if conf.f_path:
            fmats_path = resolve_model_path(conf.f_path)
            fmats = np.load(fmats_path)
        else:
            fmats = None

        self.fsnet = fsnet
        self.loader = loader
        self.fmats = fmats
        self.device = device
    
    def __call__(self,
                 path1: str,
                 path2: str,
                 fmats: List[np.ndarray]):
        data_pair = self.loader.prepare_im_pair(path1, path2)
        if len(fmats) > 0:
            _fmats = np.stack(fmats)    # Shape(#fmats, 3, 3)
            if self.fmats is not None:
                _fmats = np.concatenate([_fmats, self.fmats])
        else:
            if self.fmats is not None:
                _fmats = self.fmats
            else:
                raise ValueError
        
        F_mats_torch = torch.Tensor(_fmats).to(self.device)
        best_fmat_id, errors = self.fsnet(data_pair, F_mats_torch)
        return best_fmat_id
