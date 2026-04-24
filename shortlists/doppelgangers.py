import argparse
import copy
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
import torchvision
import tqdm
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

from scripts.data import FilePath, resolve_model_path
from models.doppelgangers.models.cnn_classifier import decoder as DoppelGangers
from models.doppelgangers.utils.dataset import read_loftr_matches
from models.doppelgangers.third_party.loftr import LoFTR, default_cfg
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import ShortlistGeneratorConfig
from shortlists.global_descriptor import (
    GlobalDescriptorShortlistGenerator,
    create_global_descriptor_extractor,
)
from scripts.workspace import log


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_ckpt(weight_path: str | Path) -> dict:
    ckpt = torch.load(weight_path, map_location="cpu")
    new_ckpt = copy.deepcopy(ckpt["dec"])
    for key, value in ckpt["dec"].items():
        if "module." in key:
            new_ckpt[key[len("module.") :]] = new_ckpt.pop(key)
    return new_ckpt


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new


def read_image(img_pth, img_size, df, padding):
    if str(img_pth).endswith("gif"):
        pil_image = ImageOps.grayscale(Image.open(str(img_pth)))
        img_raw = np.array(pil_image)
    else:
        img_raw = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

    w, h = img_raw.shape[1], img_raw.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        mask = np.zeros((1, pad_to, pad_to), dtype=bool)
        mask[:, :h_new, :w_new] = True
        mask = mask[:, ::8, ::8]

    image = cv2.resize(img_raw, (w_new, h_new))
    pad_image = np.zeros((1, 1, pad_to, pad_to), dtype=np.float32)
    pad_image[0, 0, :h_new, :w_new] = image / 255.0

    return pad_image, mask


class DoppelGangersShortlistGenerator(ShortlistGenerator):
    def __init__(
        self, conf: ShortlistGeneratorConfig, device: Optional[torch.device] = None
    ):
        assert conf.doppelgangers
        self.conf = conf
        self.device = device
        self.global_desc_shortlist_generator = GlobalDescriptorShortlistGenerator(
            conf, device=device
        )

        cfg = dict2namespace(
            {
                "data": {"test": {"img_size": 1024}},
                "models": {"decoder": {"input_dim": conf.doppelgangers.input_dim}},
            }
        )
        ckpt = load_ckpt(resolve_model_path(conf.doppelgangers.weight_path))
        self.classifier = DoppelGangers(cfg.models.decoder)
        self.classifier.load_state_dict(ckpt)
        print(f"[DoppelGangers] Load: {conf.doppelgangers.weight_path}")
        self.classifier = self.classifier.eval().to(self.device)

        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(
            torch.load(
                resolve_model_path(conf.doppelgangers.loftr_weight_path),
                map_location="cpu",
            )["state_dict"]
        )
        self.matcher = matcher.eval().to(self.device)

    @torch.inference_mode()
    def __call__(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None, **kwargs
    ) -> List[Tuple[int, int]]:
        assert self.conf.doppelgangers
        pairs = self.global_desc_shortlist_generator(
            scene, progress_bar=progress_bar, **kwargs
        )

        pairs_list = []
        for i, (idx1, idx2) in enumerate(pairs):
            path1 = scene.image_paths[idx1]
            path2 = scene.image_paths[idx2]

            img1_raw, mask1 = read_image(
                path1,
                self.conf.doppelgangers.loftr_img_size,
                self.conf.doppelgangers.loftr_df,
                self.conf.doppelgangers.loftr_padding,
            )
            img2_raw, mask2 = read_image(
                path2,
                self.conf.doppelgangers.loftr_img_size,
                self.conf.doppelgangers.loftr_df,
                self.conf.doppelgangers.loftr_padding,
            )
            img1 = torch.from_numpy(img1_raw).cuda()    # Shape(1, 1, 1024, 1024)
            img2 = torch.from_numpy(img2_raw).cuda()    # Shape(1, 1, 1024, 1024)
            mask1 = torch.from_numpy(mask1).cuda()      # Shape(1, 128, 128)
            mask2 = torch.from_numpy(mask2).cuda()      # Shape(1, 128, 128)

            batch = {"image0": img1, "image1": img2, "mask0": mask1, "mask1": mask2}

            self.matcher(batch)
            kpts1 = batch["mkpts0_f"].cpu().numpy()
            kpts2 = batch["mkpts1_f"].cpu().numpy()
            conf = batch["mconf"].cpu().numpy()

            if np.sum(conf > 0.8) == 0:
                matches = None
            else:
                F, mask = cv2.findFundamentalMat(
                    kpts1[conf > 0.8], kpts2[conf > 0.8], cv2.FM_RANSAC, 3, 0.99
                )
                if mask is None or F is None:
                    matches = None
                else:
                    matches = np.array(
                        np.ones((kpts1.shape[0], 2))
                        * np.arange(kpts2.shape[0]).reshape(-1, 1)
                    ).astype(int)[conf > 0.8][mask.ravel() == 1]

            image = read_loftr_matches(
                path1,
                path2,
                self.conf.doppelgangers.loftr_img_size,
                self.conf.doppelgangers.loftr_df,
                self.conf.doppelgangers.loftr_padding,
                kpts1,
                kpts2,
                matches,
                warp=True,
                conf=conf,
            )   # Shape(10, 1024, 1024)

            image = image.unsqueeze(0)  # Shape(1, 10, 1024, 1024)
            image = image.to(self.device, non_blocking=True)
            scores = torch.softmax(self.classifier(image), dim=1)
            assert scores.shape == (1, 2)

            score = scores[0, 1].item()
            if score >= self.conf.doppelgangers.threshold:
                pairs_list.append((idx1, idx2))
            
            if progress_bar:
                progress_bar.set_postfix_str(
                    f"DoppelGangers classification ({i}/{len(pairs)})"
                )

        # TODO: Support re-computing topk dists and ranks
        scene.update_shortlist(pairs_list).release_topk_dists_and_ranks()
        return pairs_list
