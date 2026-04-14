from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from core import ConfigBlock
from models.config import (
    MTLDescModelConfig,
    PosFeatDetectorConfig,
    PosFeatModelConfig,
    RELFModelConfig,
)
from postprocesses.config import NMSConfig
from preprocesses.config import ResizeConfig, RotationConfig


class ALIKEConfig(BaseModel):
    weight_path: str
    model_type: str = "alike-l"

    top_k: int = 500
    scores_th: float = 0.5
    n_limit: int = 5000
    sub_pixel: bool = False

    def __str__(self) -> str:
        return f"alike(topk={self.top_k}, th={self.scores_th}, n_limit={self.n_limit})"


class ALIKEDConfig(BaseModel):
    weight_path: str

    model_name: str = "aliked-n32"
    top_k: int = -1  # -1 for threshold based mode, >0 for top K mode.
    scores_th: float = 0.2
    n_limit: int = 5000  # Maximum number of keypoints to be detected

    def __str__(self) -> str:
        return f"aliked(topk={self.top_k}, th={self.scores_th}, n_limit={self.n_limit})"


class DaDConfig(BaseModel):
    weight_path: str
    num_keypoints: int = 512


class DarkFeatConfig(BaseModel):
    weight_path: str


class DELFPytorchConfig(BaseModel):
    weight_path: str

    def get_extractor_config(self, path: str) -> dict:
        return {
            # params for feature extraction.
            "MODE": "delf",
            "GPU_ID": 0,
            #'IOU_THRES': 0.98,
            "IOU_THRES": 0.9,
            #'ATTN_THRES': 0.17,
            "ATTN_THRES": 0.3,
            "TOP_K": 4096,
            "PCA_DIMS": 40,
            "USE_PCA": False,
            "SCALE_LIST": [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            # params for model load.
            "LOAD_FROM": path,
            "ARCH": "resnet50",
            "EXPR": "dummy",
            "TARGET_LAYER": "layer3",
        }


class EdgePoint2Config(BaseModel):
    model_type: str
    weight_path: str
    top_k: int
    score: float = -5.0


class GIMSuperPointConfig(BaseModel):
    weight_path: str

    max_num_keypoints: int = 2048

    def __str__(self) -> str:
        return "gim()"


class LightGlueALIKEDConfig(BaseModel):
    weight_path: str

    model_name: str = "aliked-n16"
    max_num_keypoints: int = 4096
    detection_threshold: float = 0.01
    resize: int = 1024

    implement_version: str = "v1"

    def __str__(self) -> str:
        return f"lightglue_aliked(num={self.max_num_keypoints}, th={self.detection_threshold}, resize={self.resize}, impl_ver={self.implement_version})"


class LightGlueALIKEDDeDoDeV2Config(BaseModel):
    weight_path: str
    weight_path_dedode_detector: str

    max_num_keypoints: int = 4096

    implement_version: str = "v2"

    def __str__(self) -> str:
        return f"lightglue_aliked(num={self.max_num_keypoints}, th={self.detection_threshold}, resize={self.resize}, impl_ver={self.implement_version})"


class LightGlueALIKEDTileConfig(BaseModel):
    weight_path: str

    max_num_keypoints: int = 4096
    detection_threshold: float = 0.01
    resize: int = 1024  # not used

    max_tile_size: int = 1024

    topk_after_merge: int = 4096
    threshold_after_merge: float = 0.05

    implement_version: str = "v2"

    def __str__(self) -> str:
        return f"lightglue_aliked_tile(num={self.max_num_keypoints}, th={self.detection_threshold}, resize={self.resize}, tile_size={self.max_tile_size}, topk={self.topk_after_merge}, th={self.threshold_after_merge})"


class LightGlueDISKConfig(BaseModel):
    weight_path: str
    max_num_keypoints: int = 4096

    def __str__(self) -> str:
        return f"lightglue_doghardnet(num={self.max_num_keypoints})"


class LightGlueDoGHardNetConfig(BaseModel):
    hardnet_weight_path: str
    max_num_keypoints: int = 4096

    def __str__(self) -> str:
        return f"lightglue_doghardnet(num={self.max_num_keypoints})"


class LightGlueSIFTConfig(BaseModel):
    max_num_keypoints: int = 4096

    def __str__(self) -> str:
        return f"lightglue_sift(num={self.max_num_keypoints})"


class LightGlueSuperPointConfig(BaseModel):
    weight_path: str

    max_num_keypoints: Optional[int] = 2048
    detection_threshold: float = 0.0005
    nms_radius: int = 4
    remove_borders: int = 4

    def __str__(self) -> str:
        return f"lightglue_superpoint(num={self.max_num_keypoints}, th={self.detection_threshold}, nms={self.nms_radius}, remove_borders={self.remove_borders})"


class DeDoDeConfig(BaseModel):
    detector_weight_path: str
    descriptor_weight_path: str
    dinov2_weight_path: Optional[str] = None

    detector_model_name: str = "L-upright"
    descriptor_model_name: str = "B-upright"

    num_features: int = 10000

    def __str__(self) -> str:
        return f"dedode(num={self.num_features}, detector={self.detector_model_name}, descriptor={self.descriptor_model_name})"


class DISKConfig(BaseModel):
    num_features: int = 2048
    weight_path: Optional[str] = None

    def __str__(self) -> str:
        return f"disk(num={self.num_features})"


class FeatureBoosterConfig(BaseModel):
    weight_path: str
    model_type: str  # ALIKE+Boost-F | ALIKE+Boost-B
    descriptor: str

    alike: Optional[ALIKEConfig] = None

    def __str__(self) -> str:
        return f"featurebooster(model_type={self.model_type}, descriptor={self.descriptor})"


class LANetConfig(BaseModel):
    weight_path: str
    model_version: str = "v1"

    threshold: float = 0.0
    topk: Optional[int] = None

    def __str__(self) -> str:
        return f"lanet(topk={self.topk}, th={self.threshold}, ver={self.model_version})"


class MagicLeapSuperPointConfig(BaseModel):
    weight_path: str

    nms_radius: int = 4
    keypoint_threshold: float = 0.005
    max_keypoints: int = -1
    remove_borders: int = 4

    fix_sampling: bool = False

    def __str__(self) -> str:
        return f"magicleap_superpoint(num={self.max_keypoints}, th={self.keypoint_threshold}, nms={self.nms_radius}, remove_border={self.remove_borders})"


class MTLDescConfig(BaseModel):
    model: MTLDescModelConfig
    topk: int = 10000

    def __str__(self) -> str:
        return f"mtldesc(topk={self.topk}, th={self.model.detection_threshold})"


class FeatureSetConfig(BaseModel):
    local_features: list[LocalFeatureConfig] = []

    nms: Optional[NMSConfig] = None
    topk: Optional[int] = None

    def __str__(self) -> str:
        fs = [
            f"feat({str(getattr(f, f.type))}, rot={f.rotation.angles if f.rotation else None})"
            for f in self.local_features
        ]
        fstr = ", ".join(fs)
        return f"featureset(topk={self.topk}, features=[{fstr}])"


class HardNetConfig(BaseModel):
    orinet_weight_path: str
    keynet_weight_path: str
    affnet_weight_path: str
    hardnet_weight_path: str

    detector: str = "keynet"  # keynet | sift
    num_features: int = 2048

    def __str__(self) -> str:
        return f"hardnet(num={self.num_features}, detector={self.detector})"


class KeyNetAffNetHardNet8Config(BaseModel):
    orinet_weight_path: str
    keynet_weight_path: str
    affnet_weight_path: str
    hardnet_weight_path: str

    detector: str = "keynet"  # keynet | sift
    num_features: int = 2048

    def __str__(self) -> str:
        return (
            f"keynetaffnethardnet8(num={self.num_features}, detector={self.detector})"
        )


class PosFeatConfig(BaseModel):
    weight_path: str

    model: str = "PoSFeat"
    model_config: PosFeatModelConfig = PosFeatModelConfig()

    detector: str = "generate_kpts_single"
    detector_config: PosFeatDetectorConfig = PosFeatDetectorConfig()

    loss_distance: str = "cos"

    # TODO
    remove_border_pad_size: int = 0

    def __str__(self) -> str:
        return f"posfeat(model={self.model}, detector={self.detector})"


class RELFConfig(BaseModel):
    weight_path: str
    descriptor: RELFModelConfig
    detector: LocalFeatureConfig

    def __str__(self) -> str:
        return f"relf(desc={str(self.descriptor)}, detector={str(self.detector)})"


class SFD2Config(BaseModel):
    weight_path: str

    model_name: str = "ressegnetv2"
    use_stability: bool = True
    max_keypoints: int = 4096
    conf_th: float = 0.001
    multiscale: bool = False
    scales: list[float] = [1.0]

    def __str__(self) -> str:
        return f"sfd2(num={self.max_keypoints}, th={self.conf_th}, multiscale={self.multiscale}, scales={self.scales})"


class SiLKConfig(BaseModel):
    weight_path: str

    nms_dist: int = 0
    border_dist: int = 0
    threshold: float = 1.0  # 1.0: disabled
    topk: int = 10000

    nms: Optional[NMSConfig] = None

    def __str__(self) -> str:
        return f"silk(topk={self.topk}, th={self.threshold}, nms={self.nms_dist}, border_dist={self.border_dist})"


class SuperPointConfig(BaseModel):
    weight_path: str

    max_keypoints: int = -1
    nms_kernel: int = 9
    remove_borders_size: int = 4
    keypoint_threshold: float = 0.0

    def __str__(self) -> str:
        return f"superpoint(num={self.max_keypoints}, th={self.keypoint_threshold}, nms={self.nms_kernel})"


class XFeatConfig(BaseModel):
    weight_path: str

    topk: int = 4096
    multiscale: bool = True

    def __str__(self) -> str:
        return f"xfeat(topk={self.topk})"


class LocalFeatureConfig(ConfigBlock):
    # alike | aliked | disk | featurebooster | hardnet |
    # magicleap_superpoint | mtldesc | posfeat | sfd2 | silk | superpoint
    # lightglue_aliked | lightglue_doghardnet | lightglue_sift | lightglue_superpoint | relf | xfeat | dad | edgepoint2
    type: str = "disk"

    resize: Optional[ResizeConfig] = None
    rotation: Optional[RotationConfig] = None
    ignore_cropper: bool = False
    extract_from_pre_sampled_keypoints: bool = False
    pre_sampled_keypoints_interpolation_num: Optional[int] = None

    alike: Optional[ALIKEConfig] = None
    aliked: Optional[ALIKEDConfig] = None
    dad: Optional[DaDConfig] = None
    darkfeat: Optional[DarkFeatConfig] = None
    lightglue_aliked: Optional[LightGlueALIKEDConfig] = None
    lightglue_aliked_dedode_v2: Optional[LightGlueALIKEDDeDoDeV2Config] = None
    lightglue_aliked_tile: Optional[LightGlueALIKEDTileConfig] = None
    dedode: Optional[DeDoDeConfig] = None
    delf_pytorch: Optional[DELFPytorchConfig] = None
    disk: Optional[DISKConfig] = None
    edgepoint2: Optional[EdgePoint2Config] = None
    featurebooster: Optional[FeatureBoosterConfig] = None
    featureset: Optional[FeatureSetConfig] = None
    gim_superpoint: Optional[GIMSuperPointConfig] = None
    hardnet: Optional[HardNetConfig] = None
    hardnet8: Optional[KeyNetAffNetHardNet8Config] = None
    lanet: Optional[LANetConfig] = None
    lightglue_disk: Optional[LightGlueDISKConfig] = None
    lightglue_doghardnet: Optional[LightGlueDoGHardNetConfig] = None
    lightglue_sift: Optional[LightGlueSIFTConfig] = None
    mtldesc: Optional[MTLDescConfig] = None
    multiscale: Optional[FeatureSetConfig] = None  # compatible
    posfeat: Optional[PosFeatConfig] = None
    relf: Optional[RELFConfig] = None
    sfd2: Optional[SFD2Config] = None
    superpoint: Optional[SuperPointConfig] = None
    xfeat: Optional[XFeatConfig] = None

    # Research Only
    # --------------
    magicleap_superpoint: Optional[MagicLeapSuperPointConfig] = None
    lightglue_superpoint: Optional[LightGlueSuperPointConfig] = None
    silk: Optional[SiLKConfig] = None


FeatureSetConfig.update_forward_refs()
RELFConfig.update_forward_refs()


class LIMAPConfig(BaseModel):
    detector_method: str
    extractor_method: str

    detector_weight_path: Optional[str] = None
    extractor_weight_path: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"limap(detector={self.detector_method}, extractor={self.extractor_method})"
        )


class Line2DFeatureConfig(BaseModel):
    type: Literal["limap"]

    resize: Optional[ResizeConfig] = None

    limap: Optional[LIMAPConfig] = None

    def __str__(self) -> str:
        return f"line2d({self.type}={getattr(self, self.type)})"
