from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from PIL import Image

from data import resolve_model_path
from models.config import FFTformerModelConfig
from models.fftformer.fftformer_arch import fftformer
from pipelines.scene import Scene
from preprocesses.config import DeblurringConfig


class DeblurHandler:
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args
        ----
        img : Image.Image
            RGB order
        """
        raise NotImplementedError


class FFTformerDeblurHandler(DeblurHandler):
    def __init__(self, conf: FFTformerModelConfig, device: torch.device):
        self.conf = conf
        self.device = device
        model = fftformer()
        state_dict = torch.load(
            resolve_model_path(conf.weight_path), map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=True)
        model = model = model.eval().to(self.device)
        self.model = model

    @torch.inference_mode()
    def __call__(self, img: Image.Image) -> Image.Image:
        x = F.to_tensor(img)
        x = x.unsqueeze(0)
        x = x.to(self.device, non_blocking=True)
        b, c, h, w = x.shape
        h_n = (32 - h % 32) % 32
        w_n = (32 - w % 32) % 32
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode="reflect")
        pred = self.model(x)
        pred = pred[:, :, :h, :w]
        pred_clip = torch.clamp(pred, 0, 1)
        pred_clip += 0.5 / 255
        result_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")
        return result_img


def create_deblur_handler(
    conf: DeblurringConfig, device: torch.device
) -> DeblurHandler:
    if conf.type == "fftformer":
        assert conf.fftformer
        handler = FFTformerDeblurHandler(conf.fftformer, device)
    else:
        raise ValueError(conf.type)

    print(f"[create_deblur_handler] Created: {handler}")
    return handler


@torch.inference_mode()
def run_deblurring(scene: Scene, conf: DeblurringConfig, device: torch.device,
                   progress_bar: Optional[tqdm.tqdm] = None) -> Scene:
    handler = create_deblur_handler(conf, device)
    for i, path in enumerate(scene.image_paths):
        img = scene.get_image(path, use_original=True)

        # Check blurry
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        v = cv2.Laplacian(gray, cv2.CV_64F).var()

        if v > conf.blurry_threshold:
            # This image is not blurred
            continue

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Deblurring: v={v} vs th={conf.blurry_threshold} ({i}/{len(scene.image_paths)})"
            )

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            deblurred_image = handler(image)
        except Exception as e:
            print(f"Deblurring error: {e}")
            continue

        save_path = scene.deblur_image_dir / Path(path).name
        deblurred_image.save(save_path)

        dimg = np.array(deblurred_image)
        dimg = cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR)
        scene.deblurred_images[str(path)] = dimg
        print(f"[run_deblurring] Saved and cached: {save_path}")
