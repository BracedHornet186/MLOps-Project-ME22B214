from typing import Optional

import torch
import torchvision.transforms as T

from scripts.data import resolve_model_path
from models.config import PatchNetVLADModelConfig
from models.patchnetvlad.models.models_generic import get_backend, get_model


def create_patch_netvlad_model(
    conf: PatchNetVLADModelConfig, device: Optional[torch.device] = None
):
    device = device or torch.device("cpu")
    weight_path = str(resolve_model_path(conf.weight_path))
    checkpoint = torch.load(weight_path, map_location="cpu")

    config = {}
    config["global_params"] = {
        "pooling": conf.pooling,
        "patch_sizes": conf.patch_sizes,
        "strides": conf.strides,
        "num_pcs": conf.num_pcs,
        "vladv2": conf.vladv2,
    }

    if config["global_params"]["num_pcs"] != "0":
        assert checkpoint["state_dict"]["WPCA.0.bias"].shape[0] == int(
            config["global_params"]["num_pcs"]
        )
    config["global_params"]["num_clusters"] = str(
        checkpoint["state_dict"]["pool.centroids"].shape[0]
    )

    if config["global_params"]["num_pcs"] != "0":
        use_pca = True
    else:
        use_pca = False

    vgg_weight_path = None
    if conf.weight_path_vgg:
        vgg_weight_path = str(resolve_model_path(conf.weight_path_vgg))

    encoder_dim, encoder = get_backend(weight_path=vgg_weight_path)
    model = get_model(
        encoder, encoder_dim, config["global_params"], append_pca_layer=use_pca
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(device)

    return model


def create_patch_netvlad_transforms(resize: tuple[int, int] = (480, 640)) -> T.Compose:
    if resize[0] > 0 and resize[1] > 0:
        return T.Compose(
            [
                T.Resize(resize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
