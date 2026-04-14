from __future__ import annotations

from argparse import Namespace
from typing import Literal, Optional, Union

from pydantic import BaseModel

from core import ConfigBlock


class HuggingFaceModelConfig(BaseModel):
    pretrained_model: str
    options: Optional[dict[str, str]] = None


class APGeMConfig(BaseModel):
    weight_path: str

    resize_max: int = 1024

    whiten_name: str = "Landmarks_clean"
    whiten_p: float = 0.25
    whiten_v: Optional[float] = None
    whiten_m: Optional[float] = 1.0

    def get_whiten_params(self) -> dict:
        return {
            "whitenp": self.whiten_p,
            "whitenv": self.whiten_v,
            "whitenm": self.whiten_m,
        }


class CVNetConfig(BaseModel):
    weight_path: str

    depth: int = 101  # 50 | 101
    reduction_dim: int = 2048


class DINOv2SALADConfig(BaseModel):
    weight_path: str

    image_size: list[int] = [322, 322]


class MASt3RModelConfig(ConfigBlock):
    weight_path: str
    use_amp: bool = False

    retrieval_weight_path: Optional[str] = None
    retrieval_codebook_path: Optional[str] = None


class MASt3RRetrievalModelConfig(ConfigBlock):
    mast3r: MASt3RModelConfig

    global_desc_type: Literal["spoc", "retrieval_spoc", "retrieval_asmk"] = "spoc"


class MoGeModelConfig(ConfigBlock):
    weight_path: str


class MTLDescModelConfig(BaseModel):
    """Adapted from https://github.com/vignywang/MTLDesc/blob/master/configs/MTLDesc_eva.yaml"""

    weight_path: str
    weights_id: str = "29"
    ckpt_name: str = "mtldesc"

    name: str = "MTLDesc"
    backbone: str = "models.mtldesc.network.MTLDesc"
    detection_threshold: float = 0.9
    nms_dist: float = 4
    nms_radius: int = 4
    border_remove: float = 4


class PosFeatModelConfig(BaseModel):
    align_local_grad: bool = False
    backbone: Optional[str] = None
    backbone_config: Optional[dict] = None
    local_input_elements: list[str] = ["local_map", "local_map_small"]
    local_with_img: bool = True
    localheader: Optional[str] = None
    localheader_config: Optional[dict] = None


class PosFeatDetectorConfig(BaseModel):
    num_pts: int = 8192
    stable: bool = True
    use_nms: Union[str, bool] = "True"  # softnms, True, False
    nms_radius: int = 1
    thr: float = 0.9  # False or a float
    thr_mod: str = "abs"  # max mean abs


class DoppelGangersModelConfig(BaseModel):
    weight_path: str
    loftr_weight_path: str

    # Smaller means more pairs will be included, larger means more pairs will be filtered out
    threshold: float = 0.8

    input_dim: int = 10

    loftr_img_size: int = 1024
    loftr_df: int = 8
    loftr_padding: bool = True

    def __str__(self) -> str:
        return f"doppelgangers(th={self.threshold})"


class PatchNetVLADModelConfig(BaseModel):
    weight_path: str
    weight_path_vgg: Optional[str] = None

    resize_height: int = 480
    resize_width: int = 640

    num_clusters: int = 16  # ?
    pooling: str = "patchnetvlad"
    patch_sizes: str = "2,5,8"
    strides: str = "1,1,1"
    num_pcs: int = 4096
    vladv2: bool = False


class RELFModelConfig(BaseModel):
    name: str
    load_dir: Optional[str] = None  # weight file path, not directory

    num_group: int = 16
    channels: int = 64
    candidate: str = "top1"
    multi_gpu: str = "-1"

    def __str__(self) -> str:
        return f"relf(name={self.name})"

    def to_args(self, weight_path: str) -> Namespace:
        args = Namespace()
        args.model = self.name
        args.load_dir = weight_path
        args.num_group = self.num_group
        args.channels = self.channels
        args.candidate = self.candidate
        args.multi_gpu = self.multi_gpu
        return args


class CheckOrientationModelConfig(BaseModel):
    weight_path: str
    config_path: str

    def __str__(self) -> str:
        return f"check_ori({self.weight_path})"


class FFTformerModelConfig(BaseModel):
    weight_path: str

    def __str__(self) -> str:
        return f"fftformer({self.weight_path})"


class MicKeyModelConfig(BaseModel):
    weight_path: str
    config_path: str
    size: int


class GroundedSAMConfig(BaseModel):
    detector: HuggingFaceModelConfig
    segmentator: HuggingFaceModelConfig

    mode: Literal["neg-and-pos", "pos-only"] = "neg-and-pos"

    threshold: float = 0.35
    labels: list[str] = []

    positive_labels: list[str] = []
    negative_labels: list[str] = []

    def __str__(self) -> str:
        return f"grounded_sam(th={self.threshold}, mode={self.mode})"


class ISCModelConfig(BaseModel):
    weight_path: str
