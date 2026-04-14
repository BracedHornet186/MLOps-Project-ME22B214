""" Adapted from https://github.com/sungonce/CVNet

Test code of Correlation Verification Network """
# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch
import models.cvnet.core.checkpoint as checkpoint
from models.cvnet.core.config import cfg
from models.cvnet.model.CVNet_Rerank_model import CVNet_Rerank


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(cfg.MODEL.DEPTH, cfg.MODEL.HEADS.REDUCTION_DIM)
    print(model)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    return model
