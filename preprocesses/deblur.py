from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from PIL import Image

from scripts.data import resolve_model_path
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
        x_orig = F.to_tensor(img).unsqueeze(0).to(self.device, non_blocking=True)
        scale = 1.0
        
        while scale >= 0.25:
            if scale < 1.0:
                h, w = x_orig.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                x = torch.nn.functional.interpolate(x_orig, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                x = x_orig
                
            try:
                # Use mixed precision for a massive VRAM reduction!
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    b, c, h, w = x.shape
                    h_n = (32 - h % 32) % 32
                    w_n = (32 - w % 32) % 32
                    x_pad = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode="reflect")
                    
                    pred = self.model(x_pad)
                    pred = pred[:, :, :h, :w]
                    
                # Inference succeeded
                pred_clip = torch.clamp(pred, 0, 1) + (0.5 / 255.0)
                
                # Restore original resolution output if downscaling was triggered
                if scale < 1.0:
                    orig_h, orig_w = x_orig.shape[2:]
                    pred_clip = torch.nn.functional.interpolate(pred_clip.to(torch.float32), size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    
                return F.to_pil_image(pred_clip.squeeze(0).cpu(), mode="RGB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"CUDA OOM caught at scale factor {scale}. Halving resolution and retrying.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    scale *= 0.5
                else:
                    raise e
                    
        raise RuntimeError("Deblurring failed due to persistent PyTorch CUDA OOM even at scale 0.25.")


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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        save_path = scene.deblur_image_dir / Path(path).name
        deblurred_image.save(save_path)

        dimg = np.array(deblurred_image)
        dimg = cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR)
        scene.deblurred_images[str(path)] = dimg
        print(f"[run_deblurring] Saved and cached: {save_path}")
