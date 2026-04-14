#import ml_collections
from typing import Any, Optional, Tuple, List
from pydantic import BaseModel


class TransformerConfig(BaseModel):
    mlp_dim: int
    num_heads: int
    num_layers: int
    attention_dropout_rate: float
    dropout_rate: float

    def get(self, k: str) -> Any:
        return getattr(self, k)

    def __getitem__(self, k: str) -> Any:
        return self.get(k)


class ResNetConfig(BaseModel):
    num_layers: Tuple[int, ...] = (3, 4, 9)
    width_factor: int = 1


class PatchesConfig(BaseModel):
    size: Tuple[int, int] = (16, 16)
    grid: Optional[Tuple[int, int]] = None

    def get(self, k: str) -> Any:
        return getattr(self, k)

    def __getitem__(self, k: str) -> Any:
        return self.get(k)


class Config(BaseModel):
    patches: PatchesConfig
    hidden_size: int
    transformer: TransformerConfig
    resnet: Optional[ResNetConfig] = None
    classifier: str = 'seg'
    representation_size: Optional[int] = None
    resnet_pretrained_path: Optional[str] = None
    pretrained_path: Optional[str] = None
    patch_size: int = 16
    decoder_channels: Tuple[int, ...] = tuple()
    skip_channels: Optional[Tuple[int, ...]] = None
    n_classes: int = 2
    n_skip: Optional[int] = None
    activation: str = 'softmax'



def get_b16_config(
    pretrained_path: str = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
):
    """Returns the ViT-B/16 configuration."""
    config = Config(
        patches=PatchesConfig(
            size=(16, 16)
        ),
        hidden_size=768,
        transformer=TransformerConfig(
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            attention_dropout_rate=0.0,
            dropout_rate=0.1
        ),
        classifier='seg',
        representation_size=None,
        resnet_pretrained_path=None,
        pretrained_path=pretrained_path,
        patch_size=16,
        decoder_channels=(256, 128, 64, 16),
        n_classes=2,
        activation='softmax'
    )
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = Config(
        patches=PatchesConfig(
            size=(16, 16)
        ),
        hidden_size=1,
        transformer=TransformerConfig(
            mlp_dim=1,
            num_heads=1,
            num_layers=1,
            attention_dropout_rate=0.0,
            dropout_rate=0.1
        ),
        classifier='token',
        representation_size=None,
    )
    return config


def get_r50_b16_config(
    pretrained_path: str = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
):
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ResNetConfig()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = pretrained_path
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config(
    pretrained_path: str = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
):
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = pretrained_path
    return config


def get_l16_config(
    pretrained_path: str = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
):
    """Returns the ViT-L/16 configuration."""
    config = Config(
        patches=PatchesConfig(
            size=(16, 16)
        ),
        hidden_size=128,
        transformer=TransformerConfig(
            mlp_dim=2048,
            num_heads=8,
            num_layers=12,
            attention_dropout_rate=0.0,
            dropout_rate=0.1
        ),
        classifier='seg',
        representation_size=None,
        resnet_pretrained_path=None,
        pretrained_path=pretrained_path,
        patch_size=16,
        decoder_channels=(256, 128, 64, 16),
        n_classes=2,
        activation='softmax'
    )
    return config


def get_r50_l16_config(
    resnet_pretrained_path: str = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
):
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ResNetConfig()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = resnet_pretrained_path
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = Config(
        patches=PatchesConfig(
            size=(14, 14)
        ),
        hidden_size=1280,
        transformer=TransformerConfig(
            mlp_dim=5120,
            num_heads=16,
            num_layers=32,
            attention_dropout_rate=0.0,
            dropout_rate=0.1
        ),
        classifier='token',
        representation_size=None,
    )
    return config