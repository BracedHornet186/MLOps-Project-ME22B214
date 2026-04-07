from typing import Optional

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from data import FilePath, resolve_model_path
from pipelines.scene import Scene
from preprocesses.config import DepthEstimationConfig


class DepthEstimationModel:
    @torch.inference_mode()
    def __call__(self, img: Image.Image, return_raw: bool) -> Image.Image:
        raise NotImplementedError


class HFDepthEstimationModel(DepthEstimationModel):
    def __init__(self, weight_path: str, device: torch.device):
        self.weight_path = resolve_model_path(weight_path)
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(self.weight_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.weight_path).to(
            device
        )

    @torch.inference_mode()
    def __call__(self, img: Image.Image, return_raw: bool = False) -> Image.Image:
        """https://huggingface.co/docs/transformers/model_doc/depth_anything"""
        inputs = self.image_processor(images=img, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        if return_raw:
            return prediction.squeeze().cpu().numpy()   # Shape(H, W)

        output = prediction.squeeze().cpu().numpy()
        normalize = (output.max() - output.min()) or 1
        formatted = (255 * (output - output.min()) / normalize).astype("uint8")
        depth = Image.fromarray(formatted)
        return depth


class DepthImageRetriever:
    def __init__(self, scene: Scene):
        self.scene = scene
        #self.extractor = cv2.img_hash.PHash_create()
        self.hashes = {}
    
    def build(self) -> "DepthImageRetriever":
        for path in self.scene.image_paths:
            img = self.scene.get_depth_image(path)
            #h = self.extractor.compute(img)
            f = cv2.resize(img, (30, 30)).flatten() / 255.0
            self.hashes[str(path)] = f
            print(f'[DepthImageRetriever] Register hash: {path}')
        return self
    
    def search(self, path: FilePath) -> list[tuple[str, float]]:
        q = self.hashes.get(str(path))
        if q is None:
            return []
        
        results = []
        for xpath, x in self.hashes.items():
            if str(xpath) == str(path):
                continue
            #dist = self.extractor.compare(q, x)
            dist = ((q - x) ** 2).sum()
            results.append((str(xpath), float(dist)))
        
        results = sorted(results, key=lambda x: x[1])
        return results


def create_depth_estimation_model(
    conf: DepthEstimationConfig, device: torch.device
) -> DepthEstimationModel:
    if conf.type == "depth_anything":
        assert conf.hf_weight_path
        model = HFDepthEstimationModel(conf.hf_weight_path, device=device)
    else:
        raise ValueError(conf.type)
    return model


@torch.inference_mode()
def run_depth_estimation(
    scene: Scene,
    conf: DepthEstimationConfig,
    device: torch.device,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> Scene:
    model = create_depth_estimation_model(conf, device)
    for i, path in enumerate(scene.image_paths):
        img = scene.get_image(path, use_original=True)
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        depth_image = model(image, return_raw=conf.keep_raw_prediction)

        if conf.keep_raw_prediction:
            assert isinstance(depth_image, np.ndarray)
            scene.update_depth_image(path, depth_image)
        else:
            scene.update_depth_image(path, np.array(depth_image))

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Depth estimation: ({i}/{len(scene.image_paths)})"
            )

    return scene
