from typing import Optional

import torch
import torchvision.transforms as T
from models.dinov2_salad.vpr_model import VPRModel
from models.dinov2_salad.models.backbones.dinov2 import DINOV2_ARCHS


def dinov2_salad(
    weight_path: str,
    backbone : str = "dinov2_vitb14",
    pretrained=True,
    backbone_args=None,
    agg_args=None,
) -> torch.nn.Module:
    """Return a DINOv2 SALAD model.
    
    Args:
        backbone (str): DINOv2 encoder to use ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14').
        pretrained (bool): If True, returns a model pre-trained on GSV-Cities (only available for 'dinov2_vitb14').
        backbone_args (dict): Arguments for the backbone (check models.backbones.dinov2).
        agg_args (dict): Arguments for the aggregation module (check models.aggregators.salad).
    Return:
        model (torch.nn.Module): the model.
    """
    assert backbone in DINOV2_ARCHS.keys(), f"Parameter `backbone` is set to {backbone} but it must be one of {list(DINOV2_ARCHS.keys())}"
    assert not pretrained or backbone == "dinov2_vitb14", f"Parameter `pretrained` can only be set to True if backbone is 'dinov2_vitb14', but it is set to {backbone}"


    backbone_args = backbone_args or {
        'num_trainable_blocks': 4,
        'return_token': True,
        'norm_layer': True,
    }
    agg_args = agg_args or {
        'num_channels': DINOV2_ARCHS[backbone],
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256,
    }
    model = VPRModel(
        backbone_arch=backbone,
        backbone_config=backbone_args,
        agg_arch='SALAD',
        agg_config=agg_args,
    )
    model.load_state_dict(
        torch.load(weight_path, map_location=torch.device('cpu'))
    )
    return model


def dinov2_salad_input_transform(image_size: Optional[tuple[int, ...]] = None) -> T.Compose:
    MEAN=[0.485, 0.456, 0.406]
    STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])