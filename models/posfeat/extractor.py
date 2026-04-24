import datetime
import importlib
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import skimage.io
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image as Im
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import models.posfeat.losses.preprocess_utils as putils
from scripts.data import FilePath
from models.config import PosFeatDetectorConfig, PosFeatModelConfig
from models.posfeat.losses.preprocess_utils import (denormalize_coords,
                                                    sample_feat_by_coord)
from models.posfeat.networks.PoSFeat_model import PoSFeat


def get_resized_wh(
    w: int, h: int,
    resize: Optional[int] = None
) -> Tuple[int, int]:
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(
    w: int, h: int, df: int
) -> Tuple[int, int]:
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(
    inp: np.ndarray,
    pad_size: int,
    ret_mask: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    #assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        # inp: Shape(H, W, C)
        padded = np.zeros((pad_size, pad_size, inp.shape[2]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    else:
        raise NotImplementedError()
    return padded, mask


class PosFeatExtractor:
    def __init__(self,
                 weight_path: str,
                 model_name: str,
                 detector_name: str,
                 detector_conf: PosFeatDetectorConfig,
                 loss_distance: str = 'cos',
                 device: Optional[torch.device] = None):
        assert model_name == 'PoSFeat'

        if detector_conf.use_nms in ('True', 'true', True):
            detector_conf.use_nms = True
        elif detector_conf.use_nms in ('False', 'false', False):
            detector_conf.use_nms = False

        ckpt_path = Path(weight_path)
        cfg_path = ckpt_path.parent / 'config.yaml'
        with open(cfg_path, 'r') as f:
            pre_conf = yaml.load(f, Loader=yaml.FullLoader)
        
        model_conf = PosFeatModelConfig(**pre_conf['model_config'])

        self.detector_conf = detector_conf
        self.model_conf = model_conf
        self.loss_distance = loss_distance
        print(self.detector_conf)
        print(self.model_conf)

        # model
        self.model = PoSFeat(model_conf.dict(), device)

        self.model.load_checkpoint(weight_path)
        self.model.set_eval()

        self.detector = getattr(putils, detector_name)
        print('Use {} to detect keypoints'.format(detector_name))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def read_image(
        self, path: FilePath,
        image_size: Optional[int] = None,
        padding_to_square: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        img_orig = skimage.io.imread(str(path))

        h, w, _ = img_orig.shape
        # FIXME
        new_w, new_h = get_resized_wh(w, h, resize=image_size)
        new_w, new_h = get_divisible_wh(new_w, new_h, 16)
        scale = np.array([w / new_w, h / new_h])
        im = cv2.resize(img_orig.copy(), (new_w, new_h))

        if padding_to_square:
            pad_to = max(new_h, new_w)
            im, mask = pad_bottom_right(im, pad_to, ret_mask=True)
        else:
            mask = None

        x = self.transform(im)
        x = x.unsqueeze(0)
        return x, img_orig, scale, mask

    def read_image_with_precomputed_scale(
        self, path: FilePath,
        scale: float,
        padding_to_square: bool = False
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        img_orig = skimage.io.imread(str(path))
        h, w, _ = img_orig.shape
        # FIXME
        image_size = max(int(w * scale), int(h * scale))
        new_w, new_h = get_resized_wh(w, h, resize=image_size)
        new_w, new_h = get_divisible_wh(new_w, new_h, 16)
        final_scale = np.array([w / new_w, h / new_h])
        im = cv2.resize(img_orig.copy(), (new_w, new_h))

        if padding_to_square:
            pad_to = max(new_h, new_w)
            im, mask = pad_bottom_right(im, pad_to, ret_mask=True)
        else:
            mask = None

        x = self.transform(im)
        x = x.unsqueeze(0)
        return x, img_orig, final_scale, mask

    def process(
        self, inputs: dict, outputs: dict, remove_pad: bool = False
    ) -> dict:
        desc_f = outputs['local_map']

        if remove_pad:
            b,c,h,w = inputs['im1_ori'].shape
            pad = inputs['pad1']
            desc_f = desc_f[:,:,:-(pad[3]//4),:-(pad[0]//4)]
            outputs['local_point'] = outputs['local_point'][:,:,:-(pad[3]//4),:-(pad[0]//4)]
        else:
            b,c,h,w = inputs['im1'].shape

        coord_n, kp_score = self.detector(outputs['local_point'],
                                          **self.detector_conf.dict())
        coords = denormalize_coords(coord_n, h, w)

        feat_f = sample_feat_by_coord(desc_f, coord_n,
                                      self.loss_distance == 'cos')
        kpt = coords.cpu().numpy().squeeze(0)

        # scale for inloc
        kpt = kpt * inputs['scale']

        return {'kpt': kpt,
                'desc': feat_f,
                'kp_score': kp_score}

    @torch.no_grad()
    def extract(self, inputs: dict) -> dict:
        for key, val in inputs.items():
            if key in ('name1', 'pad1', 'scale'):
                continue
            inputs[key] = val.cuda(non_blocking=True)
        outputs = self.model.extract(inputs['im1'])
        processed = self.process(inputs, outputs)
        torch.cuda.empty_cache()
        return processed

    def process_batch(
        self, inputs: dict, outputs: dict, remove_pad: bool = False
    ) -> dict:
        desc_f = outputs['local_map']

        if remove_pad:
            b,c,h,w = inputs['im1_ori'].shape
            pad = inputs['pad1']
            desc_f = desc_f[:,:,:-(pad[3]//4),:-(pad[0]//4)]
            outputs['local_point'] = outputs['local_point'][:,:,:-(pad[3]//4),:-(pad[0]//4)]
        else:
            b,c,h,w = inputs['im1'].shape

        coord_n, kp_score = self.detector(outputs['local_point'],
                                          **self.detector_conf.dict())
        coords = denormalize_coords(coord_n, h, w)

        feat_f = sample_feat_by_coord(desc_f, coord_n,
                                      self.loss_distance == 'cos')
        kpt = coords

        # scale for inloc
        #kpt = kpt * inputs['scale']

        return {'kpt': kpt,
                'desc': feat_f,
                'kp_score': kp_score}

    @torch.no_grad()
    def extract_batch(self, inputs: dict) -> dict:
        for key, val in inputs.items():
            if key in ('name1', 'pad1', 'scale'):
                continue
            inputs[key] = val.cuda(non_blocking=True)
        outputs = self.model.extract(inputs['im1'])
        processed = self.process_batch(inputs, outputs)
        torch.cuda.empty_cache()
        return processed
