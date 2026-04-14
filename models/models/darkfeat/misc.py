import torch
import numpy as np


def frame2tensor(frame: np.ndarray, device: torch.device):
    if len(frame.shape) == 2:
        return torch.from_numpy(frame/255.).float()[None, None].to(device)
    else:
        return torch.from_numpy(frame/255.).float().permute(2, 0, 1)[None].to(device)