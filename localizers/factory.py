import torch
from typing import Optional
from localizers.config import PostLocalizerConfig, LocalizerConfig
from localizers.base import PostLocalizer, Localizer
from localizers.two_view import TwoViewPostLocalizer


def create_localizer(
    conf: LocalizerConfig, device: Optional[torch.device] = None
) -> Localizer:
    if device is None:
        device = torch.device('cuda')

    if conf.type == "mapfree":
        from localizers.mapfree import MapFreeLocalizer
        assert conf.mapfree
        localizer = MapFreeLocalizer(conf.mapfree, device=device)
    elif conf.type == "mickey":
        from localizers.mickey import MicKeyLocalizer
        assert conf.mickey
        localizer = MicKeyLocalizer(conf.mickey, device=device)
    else:
        raise ValueError(conf.type)
    return localizer


def create_post_localizer(
    conf: PostLocalizerConfig, device: Optional[torch.device] = None
) -> PostLocalizer:
    if device is None:
        device = torch.device('cuda')

    if conf.type == "mapfree":
        from localizers.mapfree import MapFreePostLocalizer
        assert conf.mapfree
        localizer = MapFreePostLocalizer(conf.mapfree, device=device)
    elif conf.type == "two_view":
        assert conf.two_view
        localizer = TwoViewPostLocalizer(conf.two_view, device=device)
    else:
        raise ValueError(conf.type)
    return localizer
