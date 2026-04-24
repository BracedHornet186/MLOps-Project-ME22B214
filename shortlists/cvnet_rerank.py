from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from scripts.data import FilePath, resolve_model_path
from models.config import CVNetConfig
from models.cvnet.model.CVNet_Rerank_model import CVNet_Rerank
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import ShortlistGeneratorConfig
from scripts.workspace import log


class CVNetRerankShortlistGenerator(ShortlistGenerator):
    def __init__(self,
                 conf: ShortlistGeneratorConfig,
                 device: Optional[torch.device] = None):
        assert conf.cvnet
        assert conf.cvnet_rerank_batch_size == 1
        self.conf = conf
        self.device = device
        image_size = None
        if conf.cvnet_rerank_image_size_height and conf.cvnet_rerank_image_size_width:
            image_size = (
                conf.cvnet_rerank_image_size_height,
                conf.cvnet_rerank_image_size_width
            )
        model, transforms = load_cvnet(conf.cvnet,
                                       device=device,
                                       image_size=image_size)
        self.model = model
        self.transforms = transforms
        log(f'[CVNetRerankShortlistGenerator] conf={conf}')

    @torch.inference_mode()
    def __call__(self,
                 scene: Scene,
                 progress_bar: Optional[tqdm.tqdm] = None,
                 **kwargs) -> List[Tuple[int, int]]:
        image_paths = scene.image_paths
        if len(image_paths) <= self.conf.global_desc_fallback_threshold:
            log(f'# of images is less than '
                f'{self.conf.global_desc_fallback_threshold}')
            log(f'-> Use all pairs')
            pairs = get_all_pairs(image_paths)
            scene.update_shortlist(pairs)
            return pairs

        dataset = ImageDataset(image_paths, self.transforms)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.conf.cvnet_rerank_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.conf.cvnet_rerank_num_workers
        )

        inputs = []
        for i, x in enumerate(loader, start=1):
            inputs.append(x)
            if progress_bar:
                progress_bar.set_postfix_str(
                    f'[CVNetRerankShortlistGenerator] Caching images ({i}/{len(loader)})'
                )
        
        feats = []
        for i, x in enumerate(inputs, start=1):
            x: torch.Tensor = x.clone()
            if self.device:
                x = x.to(self.device, non_blocking=True)
            f = self.model.extract_global_descriptor(x)
            feats.append(f)
            if progress_bar:
                progress_bar.set_postfix_str(
                    f'[CVNetRerankShortlistGenerator] Global descriptors extraction ({i}/{len(loader)})'
                )
        feats = torch.cat(feats)
        # NOTE: Assume that each feature is L2 normalized
        sims = feats @ feats.T

        torch.cuda.synchronize()
        feats = None
        torch.cuda.empty_cache()

        corrs = torch.zeros_like(sims)
        all_pairs = get_all_pairs(image_paths)
        for i, (q, r) in enumerate(all_pairs, start=1):
            xq: torch.Tensor = inputs[q].clone()
            xr: torch.Tensor = inputs[r].clone()
            if self.device:
                xq = xq.to(self.device, non_blocking=True)
                xr = xr.to(self.device, non_blocking=True)
            corr: float = self.model(xq, xr).item()
            corrs[q, r] = corr
            if progress_bar:
                progress_bar.set_postfix_str(
                    f'[CVNetRerankShortlistGenerator] Correlation prediction ({i}/{len(all_pairs)})'
                )

        # np.save('sims.npy', sims.cpu().numpy())
        # np.save('corrs.npy', corrs.cpu().numpy())

        scores = []
        for q, r in all_pairs:
            sg = sims[q, r].item()
            sr = corrs[q, r].item()
            scores.append(sg + self.conf.cvnet_rerank_alpha * sr)
        
        pairs_list = []
        for score, (q, r) in zip(scores, all_pairs):
            if score >= self.conf.cvnet_rerank_threshold:
                pairs_list.append((q, r))
        
        pairs_list = sorted(list(set(pairs_list)))
        scene.update_shortlist(pairs_list)
        return pairs_list


def load_cvnet(
    conf: CVNetConfig,
    device: Optional[torch.device] = None,
    image_size: Optional[Tuple[int, int]] = None
) -> Tuple[CVNet_Rerank, T.Compose]:
    model = CVNet_Rerank(
        conf.depth,
        conf.reduction_dim
    )
    state_dict = torch.load(
        resolve_model_path(conf.weight_path),
        map_location='cpu'
    )['model_state']
    model_dict = model.state_dict()
    state_dict = {k : v for k, v in state_dict.items()}
    weight_dict = {k : v for k, v in state_dict.items()
                    if k in model_dict and model_dict[k].size() == v.size()}

    if len(weight_dict) != len(state_dict):
        raise AssertionError("The model is not fully loaded.")

    model_dict.update(weight_dict)
    model.load_state_dict(model_dict)
    model = model.eval().to(device)

    if image_size:
        transforms = T.Compose([
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
    else:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    return model, transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths: List[FilePath],
                 transforms: T.Compose):
        self.image_paths = image_paths
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, i: int) -> Any:
        path = self.image_paths[i]
        img = cv2.imread(str(path))
        img = Image.fromarray(img)
        x = self.transforms(img)
        return x
