from pathlib import Path
from typing import Any, Callable, Optional

import limap.base
import limap.line2d
import numpy as np
import torch
from numpy import ndarray

from scripts.data import resolve_model_path
from features.base import Line2DFeatureHandler, read_image
from features.config import LIMAPConfig
from preprocess import _get_resized_wh
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class LIMAPDetectionHandler(Line2DFeatureHandler):
    def __init__(self, conf: LIMAPConfig):
        self.conf = conf

        detector_weight_path = None
        if conf.detector_weight_path:
            detector_weight_path = str(resolve_model_path(conf.detector_weight_path))
        detector = limap.line2d.get_detector(
            {"method": conf.detector_method, "skip_exists": False},
            weight_path=detector_weight_path,
        )

        extractor_weight_path = None
        if conf.extractor_weight_path:
            extractor_weight_path = str(resolve_model_path(conf.extractor_weight_path))
        extractor = limap.line2d.get_extractor(
            {"method": conf.extractor_method, "skip_exists": False},
            weight_path=extractor_weight_path,
        )
        self.detector = detector
        self.extractor = extractor
        print(f"[LIMAPDetectionHandler] {detector}, {extractor}")
        print(
            f"[LIMAPDetectionHandler] {detector_weight_path}, {extractor_weight_path}"
        )

    @torch.inference_mode()
    def __call__(
        self,
        path: str | Path,
        shape: tuple[int, int],
        resize: ResizeConfig | None = None,
        rotation: RotationConfig | None = None,
        cropper: Cropper | None = None,
        orientation: int | None = None,
        image_reader: Optional[Callable] = None,
    ) -> tuple[ndarray, Any]:
        H, W = shape
        h, w = None, None
        if resize:
            assert resize.long_edge_length
            w, h = _get_resized_wh(W, H, resize.long_edge_length)

            view = limap.base.CameraView(
                limap.base.Camera("SIMPLE_RADIAL", hw=(h, w)), str(path)
            )
        else:
            view = limap.base.CameraView(str(path))

        segs = self.detector.detect(view)
        descinfos = self.extractor.extract(view, segs)

        # Postprocess
        if resize:
            assert h is not None
            assert w is not None

            segs[:, 0] *= float(W) / float(w)
            segs[:, 1] *= float(H) / float(h)
            segs[:, 2] *= float(W) / float(w)
            segs[:, 3] *= float(H) / float(h)

        segs[:, 0] = np.clip(segs[:, 0], 0, W - 1)
        segs[:, 1] = np.clip(segs[:, 1], 0, H - 1)
        segs[:, 2] = np.clip(segs[:, 2], 0, W - 1)
        segs[:, 3] = np.clip(segs[:, 3], 0, H - 1)

        return segs, descinfos
