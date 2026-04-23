import math
import os
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from scripts.config import DistConfig


def init_dist(ddp: bool = False, torch_dist_backend: str = "nccl") -> DistConfig:
    if not ddp:
        return DistConfig.single()

    dist.init_process_group(backend=torch_dist_backend, init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    return DistConfig(gpu=rank, rank=rank, world_size=world_size)


def all_gather_with_auto_alignment(
    x: torch.Tensor, shard_size_list: List[int]
) -> torch.Tensor:
    world_size = len(shard_size_list)
    max_num_samples = max(shard_size_list)
    aligned_shape = (max_num_samples, *x.shape[1:])

    tensor_list = [
        torch.zeros(aligned_shape).type_as(x).to(x.device) for _ in range(world_size)
    ]

    shortage = (
        torch.zeros((max_num_samples - int(x.shape[0]), *x.shape[1:]))
        .type_as(x)
        .to(x.device)
    )
    aligned_x = torch.cat([x, shortage], dim=0)

    dist.all_gather(tensor_list, aligned_x)

    tensor_list = [
        aligned_tensor[0:orig_size]
        for orig_size, aligned_tensor in zip(shard_size_list, tensor_list)
    ]

    return torch.cat(tensor_list)


class DistributedInferenceSampler(Sampler):
    def __init__(self, dataset: Union[Dataset, np.ndarray], dist_conf: DistConfig):
        self.dataset = dataset
        self.dist_conf = dist_conf
        if isinstance(dataset, Dataset):
            self.indices = np.array(list(range(len(dataset))))  # type: ignore
        elif isinstance(dataset, np.ndarray):
            self.indices = dataset
        else:
            raise TypeError
        self.num_samples = math.floor(len(self.indices) / self.dist_conf.world_size)
        assert self.num_samples >= 1

    @property
    def world_size(self) -> int:
        return self.dist_conf.world_size

    @property
    def rank(self) -> int:
        return self.dist_conf.rank

    def __iter__(self) -> Iterator:
        begin, end = self.get_begin_and_end_idx()
        indices = self.indices[begin:end]
        assert len(indices) == len(self)
        return iter(indices)

    def __len__(self) -> int:
        return self.get_shard_size(self.rank)

    def __str__(self) -> str:
        begin, end = self.get_begin_and_end_idx()
        return f"{self.__class__.__name__}(rank={self.rank}, world_size={self.world_size}, begin={begin}, end={end})"

    def get_begin_and_end_idx(self) -> Tuple[int, int]:
        if self.rank == (self.world_size - 1):
            begin = self.rank * self.num_samples
            end = len(self.indices)
        else:
            begin = self.rank * self.num_samples
            end = (self.rank + 1) * self.num_samples
        return begin, end

    def get_shard_size_list(self) -> List[int]:
        shard_size_list = [self.num_samples] * (self.world_size - 1)
        shard_size_list.append(len(self.indices) - sum(shard_size_list))
        assert sum(shard_size_list) == len(self.indices)
        assert len(shard_size_list) == self.world_size
        return shard_size_list

    def get_shard_size(self, i: int) -> int:
        return self.get_shard_size_list()[i]

    def get_entire_dataset_size(self) -> int:
        return sum(self.get_shard_size_list())
