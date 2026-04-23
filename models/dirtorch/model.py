from collections import OrderedDict
from typing import Optional
import sys
import torch
import torch.nn.functional as F
import sklearn.decomposition


import models.dirtorch.nets as nets
from models.dirtorch.utils import common
from models.dirtorch.utils.pytorch_loader import get_loader
from models.dirtorch.utils.common import tonumpy, matmul, pool
from scripts.data import FilePath

#
# Adapted from https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/extractors/dir.py
# 
# The DIR model checkpoints (pickle files) include sklearn.decomposition.pca,
# which has been deprecated in sklearn v0.24
# and must be explicitly imported with `from sklearn.decomposition import PCA`.
# This is a hacky workaround to maintain forward compatibility.
sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca


def load_model(path: FilePath, device: Optional[torch.device] = None):
    checkpoint = common.load_checkpoint(path, False)
    checkpoint = torch.load(path, map_location='cpu')
    new_dict = OrderedDict()
    for k, v in list(checkpoint['state_dict'].items()):
        if k.startswith('module.'):
            k = k[7:]
        new_dict[k] = v
    checkpoint['state_dict'] = new_dict

    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = net.eval().to(device)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


