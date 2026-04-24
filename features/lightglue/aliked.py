from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import kornia
import numpy as np
import torch
import torch.nn as nn
from kornia.color import grayscale_to_rgb
from lightglue import ALIKED as _ALIKED
from lightglue.aliked import DKD, SDDH, ConvBlock, Extractor, resnet
from lightglue.utils import ImagePreprocessor, numpy_image_to_torch

from scripts.data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    create_rotator,
    keypoints_to_lafs,
    lafs_to_keypoints,
    postprocess,
    read_image,
)
from features.config import LightGlueALIKEDConfig
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import Cropper


class ALIKED(_ALIKED):
    def __init__(self, weight_path: str, **conf):
        Extractor.__init__(self, **conf)  # Update with default configuration.
        conf = self.conf
        c1, c2, c3, c4, dim, K, M = self.cfgs[conf.model_name]
        conv_types = ["conv", "conv", "dcn", "dcn"]
        conv2D = False
        mask = False

        # build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = self.get_resblock(c1, c2, conv_types[1], mask)
        self.block3 = self.get_resblock(c2, c3, conv_types[2], mask)
        self.block4 = self.get_resblock(c3, c4, conv_types[3], mask)

        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )
        self.score_head = nn.Sequential(
            resnet.conv1x1(dim, 8),
            self.gate,
            resnet.conv3x3(8, 4),
            self.gate,
            resnet.conv3x3(4, 4),
            self.gate,
            resnet.conv3x3(4, 1),
        )
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask)
        self.dkd = DKD(
            radius=conf.nms_radius,
            top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
            scores_th=conf.detection_threshold,
            n_limit=conf.max_num_keypoints
            if conf.max_num_keypoints > 0
            else self.n_limit_max,
        )

        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        print(f"[ALIKEDLightGlue] weight={weight_path}")

    def sample_descriptors(self, data: dict) -> dict:
        image = data["image"]
        pre_sampled_keypoints = data["pre_sampled_keypoints"][0]  # Shape(N, 2)

        if image.shape[1] == 1:
            image = grayscale_to_rgb(image)
        feature_map, score_map = self.extract_dense_map(image)
        b, c, h, w = score_map.shape

        # keypoints, kptscores, scoredispersitys = self.dkd(
        #    score_map, image_size=data.get("image_size")
        # )

        wh = torch.tensor([w - 1, h - 1], device=score_map.device)
        pre_sampled_keypoints = pre_sampled_keypoints / wh * 2 - 1
        pre_sampled_keypoint_scores = torch.nn.functional.grid_sample(
            score_map[0].unsqueeze(0),
            pre_sampled_keypoints.view(1, 1, -1, 2),
            mode="bilinear",
            align_corners=True,
        )[0, 0, 0, :]  # CxN
        descriptors, offsets = self.desc_head(feature_map, [pre_sampled_keypoints])

        _, _, h, w = image.shape
        wh = torch.tensor([w - 1, h - 1], device=image.device)
        # no padding required
        # we can set detection_threshold=-1 and conf.max_num_keypoints > 0
        return {
            "keypoints": wh
            * (torch.stack([pre_sampled_keypoints]) + 1)
            / 2.0,  # B x N x 2
            "descriptors": torch.stack(descriptors),  # B x N x D
            "keypoint_scores": torch.stack([pre_sampled_keypoint_scores]),  # B x N
        }


class LightGlueALIKEDHandler(LocalFeatureHandler):
    def __init__(
        self, conf: LightGlueALIKEDConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device
        model = (
            ALIKED(
                weight_path=str(resolve_model_path(conf.weight_path)),
                model_name=conf.model_name,
                max_num_keypoints=conf.max_num_keypoints,
                detection_threshold=conf.detection_threshold,
            )
            .eval()
            .to(self.device, dtype=torch.float32)
        )
        self.model = model

    @torch.inference_mode()
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        img = image_reader(str(path))

        ori_normalizer = None
        if orientation is not None:
            ori_normalizer = OrientationNormalizer(degree=orientation)

        img = img[..., ::-1]  # BGR->RGB
        img = numpy_image_to_torch(img)
        img = img.to(self.device, non_blocking=True)
        img = img[None]

        if cropper:
            cropper.set_original_image(img[0])
            img = cropper.crop_tensor_image(img[0])
            img = img[None, ...]  # Shape(1, 3, H', W')

        if ori_normalizer:
            ori_normalizer.set_original_image(img[0])
            img = ori_normalizer.get_upright_image_tensor()
            img = img[None, ...]  # Shape(1, 3, H'', W'')

        rotator = None
        if self.conf.implement_version == "v1":
            if rotation:
                raise NotImplementedError(
                    "Implement version v1 does not support rotation"
                )

            x = img
            if resize:
                preds = self.model.extract(x, resize=resize.lg_resize)
            else:
                preds = self.model.extract(x)
        elif self.conf.implement_version == "v2":
            if resize:
                resize_conf = {
                    "resize": resize.lg_resize,  # target edge length, None for no resizing
                    "side": "long",
                    "interpolation": "bilinear",
                    "align_corners": None,
                    "antialias": True,
                }
                x = kornia.geometry.transform.resize(
                    img,
                    resize_conf["resize"],
                    side=resize_conf["side"],
                    antialias=resize_conf["antialias"],
                    align_corners=resize_conf["align_corners"],
                )
            else:
                x = img

            if rotation:
                rotator = create_rotator(
                    rotation, x.shape[-2], x.shape[-1], device=self.device
                )
                x = rotator.transform_homography_variants_tensor(x)

            preds = self.model.extract(x, resize=None)
        else:
            raise ValueError(self.conf.implement_version)

        kpts = preds["keypoints"].reshape(-1, 2)
        scores = preds["keypoint_scores"].reshape(-1)
        descs = preds["descriptors"].reshape(len(kpts), -1)

        if rotator:
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        if self.conf.implement_version == "v1":
            # Rescaling has been done in model.extract()
            pass
        elif self.conf.implement_version == "v2":
            # Original image size
            H, W = img.shape[-2:]

            # Resized image size
            h, w = x.shape[-2:]

            # Rescale lafs
            lafs[:, 0, :] *= float(W) / float(w)
            lafs[:, 1, :] *= float(H) / float(h)
        else:
            raise ValueError

        if ori_normalizer:
            kpts = lafs_to_keypoints(lafs)
            kpts = ori_normalizer.keypoints_to_original_coords_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs, scores, descs)
        return outputs

    @torch.inference_mode()
    def extract_by_keypoints(
        self,
        path: FilePath,
        pre_sampled_keypoints: np.ndarray,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        img = image_reader(str(path))

        ori_normalizer = None
        if orientation is not None:
            ori_normalizer = OrientationNormalizer(degree=orientation)

        img = img[..., ::-1]  # BGR->RGB
        img = numpy_image_to_torch(img)
        img = img.to(self.device, non_blocking=True)
        img = img[None]

        if cropper:
            cropper.set_original_image(img[0])
            img = cropper.crop_tensor_image(img[0])
            img = img[None, ...]  # Shape(1, 3, H', W')
            pre_sampled_keypoints = cropper.crop_keypoints_ndarray(
                pre_sampled_keypoints
            )

        if ori_normalizer:
            raise NotImplementedError

        rotator = None
        if self.conf.implement_version == "v1":
            raise NotImplementedError
        elif self.conf.implement_version == "v2":
            kpts = torch.from_numpy(pre_sampled_keypoints).float().to(self.device)

            if resize:
                resize_conf = {
                    "resize": resize.lg_resize,  # target edge length, None for no resizing
                    "side": "long",
                    "interpolation": "bilinear",
                    "align_corners": None,
                    "antialias": True,
                }
                x = kornia.geometry.transform.resize(
                    img,
                    resize_conf["resize"],
                    side=resize_conf["side"],
                    antialias=resize_conf["antialias"],
                    align_corners=resize_conf["align_corners"],
                )
                H, W = img.shape[-2:]
                h, w = x.shape[-2:]

                # Scale keypoints
                kpts[:, 0] *= float(w) / float(W)
                kpts[:, 1] *= float(h) / float(H)
            else:
                x = img

            if rotation:
                rotator = create_rotator(
                    rotation, x.shape[-2], x.shape[-1], device=self.device
                )
                x = rotator.transform_homography_variants_tensor(x)

            shape = x.shape[-2:][::-1]
            _x, scales = ImagePreprocessor(resize=None)(x)

            preds = self.model.sample_descriptors(
                {"image": _x, "pre_sampled_keypoints": [kpts]}
            )
            preds["image_size"] = torch.tensor(shape)[None].to(img).float()
            preds["keypoints"] = (preds["keypoints"] + 0.5) / scales[None] - 0.5
        else:
            raise ValueError(self.conf.implement_version)

        kpts = preds["keypoints"].reshape(-1, 2)
        scores = preds["keypoint_scores"].reshape(-1)
        descs = preds["descriptors"].reshape(len(kpts), -1)

        if rotator:
            raise NotImplementedError

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        if self.conf.implement_version == "v1":
            # Rescaling has been done in model.extract()
            pass
        elif self.conf.implement_version == "v2":
            # Original image size
            H, W = img.shape[-2:]

            # Resized image size
            h, w = x.shape[-2:]

            # Rescale lafs
            lafs[:, 0, :] *= float(W) / float(w)
            lafs[:, 1, :] *= float(H) / float(h)
        else:
            raise ValueError

        if ori_normalizer:
            raise NotImplementedError

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs, scores, descs)
        return outputs
