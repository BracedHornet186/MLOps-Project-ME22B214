from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
from kornia.feature import (HardNet, KeyNetDetector, LAFAffNetShapeEstimator,
                            LAFDescriptor, LAFOrienter, LocalFeature, OriNet,
                            PassLAF, extract_patches_from_pyramid)
from kornia.utils.image import image_to_tensor
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from models.openglue.models.features.opencv import DoGOpenCVAffNetHardNet
from features.base import (LocalFeatureHandler, LocalFeatureOutputs,
                           postprocess, read_image)
from features.config import HardNetConfig
from scripts.preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class HardNetHandler(LocalFeatureHandler):
    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        x = image_to_tensor(img, False).float() / 255.
        x = kornia.color.bgr_to_rgb(x)
        return x
    
    def preprocess(
        self,
        x: torch.Tensor,
        resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        if resize is not None:
            x = resize_image_tensor(x, resize)
        # To grayscale
        x = kornia.color.rgb_to_grayscale(x)
        return x
    

class KeyNetAffNetHardNetHandler(HardNetHandler):
    def __init__(self,
                 conf: HardNetConfig,
                 device: Optional[torch.device] = None):
        assert device is not None
        self.conf = conf
        self.device = device
        self.feature = KeyNetAffNetHardNet(
            str(resolve_model_path(conf.orinet_weight_path)),
            str(resolve_model_path(conf.keynet_weight_path)),
            str(resolve_model_path(conf.affnet_weight_path)),
            str(resolve_model_path(conf.hardnet_weight_path)),
            num_features=conf.num_features,
            device=device
        )

    @torch.inference_mode()
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError

        img = image_reader(str(path))
        x = self.to_torch_image(img)
        x = self.preprocess(x, resize=resize)
        lafs, scores, descs = self.extract(x)
        lafs = lafs[0]
        scores = scores[0]
        descs = descs[0]
        outputs = postprocess((lafs, scores, descs), img, x)
        return outputs

    def extract(self, x: torch.Tensor, *args, **kwargs) -> LocalFeatureOutputs:
        x = x.to(self.device)
        return self.feature(x)


class DoGAffNetHardNetHandler(HardNetHandler):
    def __init__(self,
                 conf: HardNetConfig,
                 device: Optional[torch.device] = None):
        model = DoGOpenCVAffNetHardNet(max_keypoints=conf.num_features)

        orinet_weights = torch.load(resolve_model_path(conf.orinet_weight_path))['state_dict']
        model.orinet.angle_detector.load_state_dict(orinet_weights)

        affnet_weights = torch.load(resolve_model_path(conf.affnet_weight_path))['state_dict']
        model.affnet.load_state_dict(affnet_weights)
        
        hn_weights = torch.load(resolve_model_path(conf.hardnet_weight_path))['state_dict']
        model.hardnet.load_state_dict(hn_weights)

        self.conf = conf
        self.device = device
        self.model = model.eval().to(device)

    @torch.inference_mode()
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError

        img = image_reader(str(path))
        x = self.to_torch_image(img)
        x = self.preprocess(x, resize=resize)
        x = x.to(self.device, non_blocking=True)

        lafs, scores, descs = self.model(x)

        lafs = lafs[0]
        scores = scores[0]
        descs = descs[0]
        outputs = postprocess((lafs, scores, descs), img, x)
        return outputs


class KeyNetAffNetHardNet(LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.
    .. image:: _static/img/keynet_affnet.jpg
    """
    def __init__(
        self,
        orinet_weight_path: str,
        keynet_weight_path: str,
        affnet_weight_path: str,
        hardnet_weight_path: str,
        num_features: int = 5000,
        upright: bool = False,
        device = torch.device('cpu'),
        scale_laf: float = 1.0,
    ):
        ori_module = PassLAF() if upright else LAFOrienter(angle_detector=OriNet(False)).eval()
        if not upright:
            weights = torch.load(orinet_weight_path)['state_dict']
            ori_module.angle_detector.load_state_dict(weights)

        detector = KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=LAFAffNetShapeEstimator(False).eval()
        ).to(device)
        kn_weights = torch.load(keynet_weight_path)['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load(affnet_weight_path)['state_dict']
        detector.aff.load_state_dict(affnet_weights)
        
        hardnet = HardNet(False).eval()
        hn_weights = torch.load(hardnet_weight_path)['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)

