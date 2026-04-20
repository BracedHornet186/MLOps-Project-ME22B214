from typing import List, Literal, Optional

from pydantic import BaseModel

from models.config import (
    CheckOrientationModelConfig,
    FFTformerModelConfig,
    GroundedSAMConfig,
)


class ResizeConfig(BaseModel):
    func: str = "kornia"  # kornia | opencv | superpoint | magicleap | lightglue | gim

    size: Optional[List[int]] = None  # (H, W)
    small_edge_length: Optional[int] = None  # Smaller edge length
    long_edge_length: Optional[int] = None  # Longer edge length
    antialias: bool = False

    # For "opencv"
    divisible_factor: Optional[int] = None
    pad_bottom_right: bool = False
    cv2_interpolation: Optional[str] = None  # LANCZOS4 | None: cv2.INTER_LINEAR

    # For "SuperPoint"
    target_size_height: Optional[int] = None
    target_size_width: Optional[int] = None

    # For "magicleap"
    ml_resize: Optional[int] = None

    # For "lightglue"
    lg_resize: Optional[int] = None

    # For "gim"
    gim_resize: Optional[int] = None

    def get_cv2_interp(self):
        import cv2

        if self.cv2_interpolation is None:
            return cv2.INTER_LINEAR
        elif self.cv2_interpolation == "LANCZOS4":
            return cv2.INTER_LANCZOS4
        elif self.cv2_interpolation == "CUBIC":
            return cv2.INTER_CUBIC
        else:
            raise ValueError(self.cv2_interpolation)

    def __str__(self) -> str:
        kwargs = {}
        if self.lg_resize:
            kwargs["size"] = self.lg_resize
        elif self.gim_resize:
            kwargs["size"] = self.gim_resize
        elif self.ml_resize:
            kwargs["size"] = self.ml_resize
        elif self.long_edge_length:
            kwargs["long"] = self.long_edge_length
        elif self.small_edge_length:
            kwargs["small"] = self.small_edge_length
        elif self.size:
            kwargs["size"] = self.size

        param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"{self.func}({param_str})"


class PairedPreResizeConfig(BaseModel):
    func: str = "same_smaller_width"
    img_object_type: str = "cv2"

    limit_size: Optional[int] = None
    cv2_interpolation: Optional[str] = None

    def get_cv2_interp(self):
        import cv2

        if self.cv2_interpolation is None:
            return cv2.INTER_LINEAR
        elif self.cv2_interpolation == "LANCZOS4":
            return cv2.INTER_LANCZOS4
        elif self.cv2_interpolation == "CUBIC":
            return cv2.INTER_CUBIC
        else:
            raise ValueError(self.cv2_interpolation)


class RotationConfig(BaseModel):
    angles: List[int]  # Clockwise


class OverlapRegionEstimatorConfig(BaseModel):
    method: str = "meanshift"  # meanshift | lof

    min_matches_required: int = 50
    min_bbox_length: int = 80

    bidirectional: bool = True
    bidirectional_merging: Optional[str] = None  # None | intersection | union

    bbox_expansion: int = 8
    border_padding: int = 8

    ms_bin_seeding: bool = True
    ms_cluster_all: bool = True

    lof_n_neighbors: int = 100

    def __str__(self) -> str:
        return (
            f"oe(method={self.method}, "
            f"min_matches={self.min_matches_required}, "
            f"min_bbox={self.min_bbox_length}, "
            f"bbox_exp={self.bbox_expansion}, "
            f"border_pad={self.border_padding}, "
            f"lof_n={self.lof_n_neighbors})"
        )


class MaskingConfig(BaseModel):
    make_watermark_masks: bool = False
    watermark_overlap_delta: int = 10
    watermark_border_delta: int = 30

    rerun_overlap_estimation: bool = False

    def __str__(self) -> str:
        return (
            f"masking("
            f"watermark={self.make_watermark_masks}, "
            f"overlap_delta={self.watermark_overlap_delta}, "
            f"border_delta={self.watermark_border_delta}, "
            f"rerun_oe={self.rerun_overlap_estimation})"
        )


class OrientationNormalizationConfig(BaseModel):
    type: Literal["check_orientation"] = "check_orientation"

    check_orientation: Optional[CheckOrientationModelConfig] = None

    def __str__(self) -> str:
        return f"ori(check_ori={self.check_orientation})"


class DeblurringConfig(BaseModel):
    type: Literal["fftformer"]
    blurry_threshold: float = 100.0

    fftformer: Optional[FFTformerModelConfig] = None

    def __str__(self) -> str:
        c = getattr(self, self.type)
        return f"deblur({self.type}={str(c)})"


class DepthEstimationConfig(BaseModel):
    type: Literal["depth_anything"]

    hf_weight_path: Optional[str] = None
    keep_raw_prediction: bool = False

    def __str__(self) -> str:
        return f"depth(type={self.type})"


class SegmentationConfig(BaseModel):
    type: Literal["grounded_sam"]
    resize: int = 800
    skip_when_identical_camera_scene: bool = False
    dilation: Optional[int] = None

    grounded_sam: Optional[GroundedSAMConfig] = None

    def __str__(self) -> str:
        return f"segmentation(type={self.type}, resize={self.resize}, skip_identical_camera={self.skip_when_identical_camera_scene}, dilation={self.dilation}, gsam={self.grounded_sam})"