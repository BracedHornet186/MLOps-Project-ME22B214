from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from croco.models.curope.curope2d import _kernels
from mast3r.model import AsymmetricMASt3R


class MASt3RMatchingModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = AsymmetricMASt3R.from_pretrained(path)

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        true_shape1: torch.Tensor,
        true_shape2: torch.Tensor,
    ):
        # encode the two images --> B,S,D
        view1 = {"img": img1, "true_shape": true_shape1, "instance": "0"}
        view2 = {"img": img2, "true_shape": true_shape2, "instance": "1"}
        res1, res2 = self.model(view1, view2)

        return (
            res1["desc"],
            res2["desc"],
            res1["desc_conf"],
            res2["desc_conf"],
        )

        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(
            view1, view2
        )

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2["pts3d_in_other_view"] = res2.pop(
            "pts3d"
        )  # predict view2's pts3d in view1's frame
        return res1, res2


if __name__ == "__main__":
    model = MASt3RMatchingModel(
        "extra/pretrained_models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )
    print(type(model))
    model = model.eval().cuda()

    img1 = torch.rand((1, 3, 224, 224)).cuda()
    img2 = torch.rand((1, 3, 224, 224)).cuda()
    true_shape1 = torch.tensor([[224, 224]]).int().cuda()
    true_shape2 = torch.tensor([[224, 224]]).int().cuda()
    args = (img1, img2, true_shape1, true_shape2)
    y = model(*args)

    torch.compiler.allow_in_graph(_kernels.rope_2d)
    onnx_prog = torch.onnx.export(
        model,
        args,
        input_names=["img1", "img2", "true_shape1", "true_shape2"],
        output_names=["desc1", "desc2", "desc_conf1", "desc_conf2"],
        dynamo=True,
        # report=True,
    )
