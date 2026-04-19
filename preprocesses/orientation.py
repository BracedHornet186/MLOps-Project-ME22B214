"""Adapted from https://qiita.com/Klein/items/a04cf1a6c94d6f03846e"""

import copy
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from PIL import Image
from timm import create_model as timm_create_model
from torch import nn

from data import resolve_model_path
from models.config import CheckOrientationModelConfig
from preprocesses.config import OrientationNormalizationConfig
from pipelines.scene import Scene

image_rotation_func = {
    1: lambda img: img,
    2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    3: lambda img: img.transpose(Image.ROTATE_180),
    4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
    6: lambda img: img.transpose(Image.ROTATE_270),
    7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
    8: lambda img: img.transpose(Image.ROTATE_90),
}


def get_orientation_of_photo(img: Image.Image) -> int:
    exif = img.getexif()
    ori = exif.get(0x112, 1)
    return ori


def to_upright(img: Image.Image) -> Image.Image:
    ori = get_orientation_of_photo(img)
    return image_rotation_func[ori](img)


def compute_and_register_orientations(
    scene: Scene,
    conf: OrientationNormalizationConfig,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> Scene:
    device = torch.device("cuda:0")
    if conf.type == "check_orientation":
        assert conf.check_orientation
        handler = CheckOrientationHandler(conf.check_orientation, device=device)
    else:
        raise ValueError(conf.type)
    
    for i, path in enumerate(scene.image_paths):
        try:
            img = scene.get_image(path)
            deg = int(handler(img) * 90)
        except Exception as e:
            print(f"Check orientation error: {e}")
            deg = 0
        scene.update_orientation(path, deg)
        if progress_bar:
            progress_bar.set_postfix_str(
                f"Checking orientation ({i + 1}/{len(scene.image_paths)})"
            )
    return scene


def create_check_orientation_model(
    weight_path: str, activation: Optional[str] = "softmax"
) -> nn.Module:
    model = timm_create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)

    if activation == "softmax":
        return nn.Sequential(model, nn.Softmax(dim=1))

    return model


class CheckOrientationHandler:
    def __init__(self, conf: CheckOrientationModelConfig, device: torch.device):
        self.conf = conf
        self.device = device
        self.transforms = T.Compose(
            [
                T.Resize((224, 224)),
                T.ConvertImageDtype(torch.float),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        model = create_check_orientation_model(
            str(resolve_model_path(conf.weight_path))
        )
        model = model.eval().to(self.device)
        self.model = model

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> int:
        """
        Args
        ----
        img : np.ndarray
            Shape(H, W, C), uint8, BGR order

        Return
        ------
        rot : int
            Degrees (anti-clockwise)
                0 -> 0
                1 -> 90
                2 -> 180
                3 -> 270
        """
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        x = self.transforms(torch.from_numpy(img).permute(2, 0, 1))
        x = x.to(torch.float32).to(self.device).unsqueeze(0)
        preds = self.model(x).cpu().numpy()
        rot = preds[0].argmax()
        return int(rot.item())


class OrientationNormalizer:
    """
    Parameters
    ----------
    img: np.ndarray
        A source image with shape(H, W, C)

    degree : int
        Orientation of the source image (anti-clockwise)
    """

    orientation_to_upright_mappings = {0: 0, 1: 3, 2: 2, 3: 1}

    def __init__(self, degree: int):
        assert degree in (0, 90, 180, 270)  # anti-clockwise
        self.img = None
        self.h = None
        self.w = None
        self.degree = degree
        self.rot_k = int(degree // 90)
    
    @classmethod
    def create_if_needed(cls, degree: Optional[int] = None) -> Optional["OrientationNormalizer"]:
        if degree is None:
            return None
        return OrientationNormalizer(degree)
    
    def set_original_image(self, img: np.ndarray | torch.Tensor) -> "OrientationNormalizer":
        """
        np.ndarray, Shape(H, W, 3)
        or 
        torch.Tensor, Shape(3, H, W)
        """
        assert self.img is None
        if isinstance(img, np.ndarray):
            h = img.shape[0]
            w = img.shape[1]
        elif isinstance(img, torch.Tensor):
            h = img.shape[-2]
            w = img.shape[-1]
        else:
            raise TypeError
        self.img = copy.deepcopy(img)
        self.h = h
        self.w = w
        return self

    def get_upright_rotation_params(self) -> dict:
        k = self.orientation_to_upright_mappings[self.rot_k]
        return {"degree": int(k * 90), "k": k}

    def get_original_image_ndarray(self) -> np.ndarray:
        assert isinstance(self.img, np.ndarray)
        return self.img.copy()

    def get_original_image_tensor(self) -> torch.Tensor:
        assert isinstance(self.img, torch.Tensor)
        return self.img.clone()

    def get_upright_image_ndarray(self) -> np.ndarray:
        assert isinstance(self.img, np.ndarray)
        params = self.get_upright_rotation_params()
        return np.rot90(self.img, k=params["k"]).copy()

    def get_upright_image_tensor(self) -> torch.Tensor:
        assert isinstance(self.img, torch.Tensor)
        params = self.get_upright_rotation_params()
        return torch.rot90(self.img, k=params["k"], dims=(1, 2)).clone()
    
    def keypoints_to_original_coords_ndarray(self, kpts: np.ndarray) -> np.ndarray:
        assert self.w is not None
        assert self.h is not None
        params = self.get_upright_rotation_params()
        k = params["k"]

        if k == 0:
            return kpts
        elif k == 1:
            _kpts = kpts.copy()
            _kpts[:, 0] = (self.w - 1) - kpts[:, 1].copy()
            _kpts[:, 1] = kpts[:, 0].copy()
            return _kpts
        elif k == 2:
            _kpts = kpts.copy()
            _kpts[:, 0] = (self.w - 1) - kpts[:, 0].copy()
            _kpts[:, 1] = (self.h - 1) - kpts[:, 1].copy()
            return _kpts
        elif k == 3:
            _kpts = kpts.copy()
            _kpts[:, 0] = kpts[:, 1].copy()
            _kpts[:, 1] = (self.h - 1) - kpts[:, 0].copy()
            return _kpts
        else:
            raise ValueError(k)

    @torch.inference_mode()
    def keypoints_to_original_coords_torch(self, kpts: torch.Tensor) -> torch.Tensor:
        _kpts = self.keypoints_to_original_coords_ndarray(kpts.cpu().numpy())
        return torch.from_numpy(_kpts)
