from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead, activate_head, custom_interpolate
from vggt.heads.track_head import TrackHead
from vggt.heads.track_modules.base_track_predictor import BaseTrackerPredictor
from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from vggt.layers.vision_transformer import vit_base, vit_giant2, vit_large, vit_small
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.models.vggt import VGGT

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 518


class CustomVGGT(VGGT):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = CustomAggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1",
        )
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )
        self.track_head = CustomTrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    @torch.inference_mode()
    def extract_patch_tokens(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        B, C_in, H, W = images.shape
        assert B == 1

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        # NOTE: self.aggregator._resnet_mean: Shape(1, 1, 3, 1, 1)
        # NOTE: self.aggregator._resnet_std: Shape(1, 1, 3, 1, 1)
        images = (
            images - self.aggregator._resnet_mean[0]
        ) / self.aggregator._resnet_std[0]

        # Reshape to [B*S, C, H, W] for patch embedding
        patch_tokens = self.aggregator.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        return patch_tokens


class CustomAggregator(Aggregator):
    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        # Fixed args for CustomAggregator
        B: int = 1,
        S: int = 2,
        H: int = IMG_SIZE,
        W: int = IMG_SIZE,
    ) -> tuple[list[torch.Tensor], int]:
        """
        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S,
                H // self.patch_size,
                W // self.patch_size,
                device=patch_tokens.device,
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2)
                .to(patch_tokens.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = (
                        self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat(
                    [frame_intermediates[i], global_intermediates[i]], dim=-1
                )
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx


class CustomDPTHead(DPTHead):
    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        patch_start_idx: int,
        frames_chunk_size: int = 8,
        # Fixed args for CustomDPTHead
        B: int = 1,
        S: int = 2,
        H: int = IMG_SIZE,
        W: int = IMG_SIZE,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, patch_start_idx)
        raise NotImplementedError

    def _forward_impl(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        patch_start_idx: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
        # Fixed args for CustomDPTHead
        B: int = 1,
        S: int = 2,
        H: int = IMG_SIZE,
        W: int = IMG_SIZE,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if frames_start_idx is not None and frames_end_idx is not None:
            raise NotImplementedError

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (
                int(patch_h * self.patch_size / self.down_ratio),
                int(patch_w * self.patch_size / self.down_ratio),
            ),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(
            out, activation=self.activation, conf_activation=self.conf_activation
        )

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf


class CustomTrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = CustomDPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        )

        # Tracker module that predicts point trajectories
        # Takes feature maps and predicts coordinates and visibility
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,  # Match the output_dim of feature extractor
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx,
        query_points=None,
        iters=None,
    ):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        # Extract features from tokens
        # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
        feature_maps = self.feature_extractor(aggregated_tokens_list, patch_start_idx)

        # Use default iterations if not specified
        if iters is None:
            iters = self.iters

        # Perform tracking using the extracted features
        coord_preds, vis_scores, conf_scores = self.tracker(
            query_points=query_points,
            fmaps=feature_maps,
            iters=iters,
        )

        return coord_preds, vis_scores, conf_scores
