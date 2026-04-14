"""https://github.com/HITCSC/EdgePoint2/blob/main/edgepoint2.py"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BasicLayer, ResNetLayer


class EdgePoint2(nn.Module):
    def __init__(self, c1, c2, c3, c4, cdesc, cdetect):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        csum = c2 + c3 + c4

        self.block1 = nn.Sequential(
            BasicLayer(1, c1, 4, 2, 1), BasicLayer(c1, c2), ResNetLayer(c2, c2)
        )
        self.block2 = ResNetLayer(c2, c3)
        self.block3 = ResNetLayer(c3, c4)

        self.desc1 = nn.Identity()
        self.desc2 = nn.Identity()
        self.desc3 = nn.Identity()
        self.desc_head = nn.Sequential(
            nn.Conv2d(csum, csum, 1),
            BasicLayer(csum, csum, groups=csum // 16),
            nn.Conv2d(csum, cdesc, 1),
        )

        self.conv1 = nn.Conv2d(c2, cdetect, 1)
        self.conv2 = nn.Conv2d(c3, cdetect, 1)
        self.conv3 = nn.Conv2d(c4, cdetect, 1)

        self.score_head = nn.Sequential(
            nn.Conv2d(cdetect, cdetect, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cdetect, cdetect, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cdetect, 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        with torch.no_grad():
            if x.shape[1] > 1:
                x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        x1 = self.block1(x)
        _x2 = F.avg_pool2d(x1, 2, 2)
        x2 = F.avg_pool2d(_x2, 2, 2)
        x2 = self.block2(x2)
        x3 = F.avg_pool2d(x2, 4, 4)
        x3 = self.block3(x3)

        desc = torch.cat(
            [
                self.desc1(_x2),
                F.interpolate(
                    self.desc2(x2), scale_factor=2, mode="bilinear", align_corners=False
                ),
                F.interpolate(
                    self.desc3(x3), scale_factor=8, mode="bilinear", align_corners=False
                ),
            ],
            1,
        )
        desc = self.desc_head(desc)

        score = (
            self.conv1(x1)
            + F.interpolate(
                self.conv2(x2), scale_factor=4, mode="bilinear", align_corners=False
            )
            + F.interpolate(
                self.conv3(x3), scale_factor=16, mode="bilinear", align_corners=False
            )
        )
        score = self.score_head(score)

        return desc, score

    def sample(self, dense, kpts, *, norm=True, align_corners=False):
        desc = F.grid_sample(dense, kpts, mode="bilinear", align_corners=align_corners)
        return F.normalize(desc, 2, 1) if norm else desc


class EdgePoint2Wrapper(nn.Module):
    cfgs = {
        "T32": {"c1": 8, "c2": 8, "c3": 16, "c4": 24, "cdesc": 32, "cdetect": 8},
        "T48": {"c1": 8, "c2": 8, "c3": 16, "c4": 24, "cdesc": 48, "cdetect": 8},
        "S32": {"c1": 8, "c2": 8, "c3": 24, "c4": 32, "cdesc": 32, "cdetect": 8},
        "S48": {"c1": 8, "c2": 8, "c3": 24, "c4": 32, "cdesc": 48, "cdetect": 8},
        "S64": {"c1": 8, "c2": 8, "c3": 24, "c4": 32, "cdesc": 64, "cdetect": 8},
        "M32": {"c1": 8, "c2": 16, "c3": 32, "c4": 48, "cdesc": 32, "cdetect": 8},
        "M48": {"c1": 8, "c2": 16, "c3": 32, "c4": 48, "cdesc": 48, "cdetect": 8},
        "M64": {"c1": 8, "c2": 16, "c3": 32, "c4": 48, "cdesc": 64, "cdetect": 8},
        "L32": {"c1": 8, "c2": 16, "c3": 48, "c4": 64, "cdesc": 32, "cdetect": 8},
        "L48": {"c1": 8, "c2": 16, "c3": 48, "c4": 64, "cdesc": 48, "cdetect": 8},
        "L64": {"c1": 8, "c2": 16, "c3": 48, "c4": 64, "cdesc": 64, "cdetect": 8},
        "E32": {"c1": 16, "c2": 16, "c3": 48, "c4": 64, "cdesc": 32, "cdetect": 16},
        "E48": {"c1": 16, "c2": 16, "c3": 48, "c4": 64, "cdesc": 48, "cdetect": 16},
        "E64": {"c1": 16, "c2": 16, "c3": 48, "c4": 64, "cdesc": 64, "cdetect": 16},
    }

    def __init__(
        self,
        model_type: str,
        weight_path: str,
        top_k: int,
        k: int = 2,
        score: float = -5,
    ):
        super().__init__()
        assert top_k is None or top_k > 0
        self.top_k = top_k
        self.k = k
        self.score_thresh = score

        self.model = EdgePoint2(**self.cfgs[model_type])
        self.model.load_state_dict(torch.load(weight_path, "cpu"))

        self.mp = nn.MaxPool2d(k * 2 + 1, 1, k)

    @torch.inference_mode()
    def forward(self, x):
        B, _, oH, oW = x.shape
        nH = oH // 32 * 32
        nW = oW // 32 * 32
        size = torch.tensor([nW, nH], dtype=x.dtype, device=x.device)
        scale = torch.tensor([oW / nW, oH / nH], dtype=x.dtype, device=x.device)
        if oW != nW or oH != nH:
            x = F.interpolate(x, (nH, nW), mode="bilinear", align_corners=True)

        raw_desc, raw_detect = self.model(x)

        detect1 = raw_detect == self.mp(raw_detect)
        detect1[..., :, :4] = False
        detect1[..., :, -4:] = False
        detect1[..., :4, :] = False
        detect1[..., -4:, :] = False

        detect2 = raw_detect > self.score_thresh
        detect = torch.logical_and(detect1, detect2)[:, 0]
        H = torch.arange(detect.shape[-2], dtype=x.dtype, device=x.device)
        W = torch.arange(detect.shape[-1], dtype=x.dtype, device=x.device)
        H, W = torch.meshgrid(H, W)
        ind = torch.stack([W, H], dim=-1)
        kpts = [ind[detect[b]] for b in range(B)]
        scores = [raw_detect[b, 0, detect[b]] for b in range(B)]

        if self.top_k is not None:
            for i in range(B):
                score, idx = scores[i].topk(min(self.top_k, scores[i].shape[0]))
                scores[i] = score
                kpts[i] = kpts[i][idx]

        descs = [
            self.model.sample(
                raw_desc[b : b + 1], (kpts[b] + 0.5).reshape(1, -1, 1, 2) / size * 2 - 1
            )[0, :, :, 0].mT.contiguous()
            if kpts[b].shape[0] > 0
            else raw_desc.new_zeros([0, raw_desc.shape[1]])
            for b in range(B)
        ]

        return [
            {"keypoints": kpts[b] * scale, "scores": scores[b], "descriptors": descs[b]}
            for b in range(B)
        ]
