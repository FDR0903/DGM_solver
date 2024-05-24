import random
from abc import ABCMeta
from builtins import ValueError
from dataclasses import dataclass, fields, asdict
from typing import Tuple, Generic, Union, TypeVar, List

import numpy as np
import torch
from functools import cached_property

from torch_util import DEVICE


@dataclass
class Sample:
    t_interior: torch.Tensor
    x_interior: torch.Tensor
    t_terminal: torch.Tensor
    x_terminal: torch.Tensor

    def require_grad(self, option: bool = True) -> None:
        for field in fields(self.__class__):
            getattr(self, field.name).requires_grad = option


@dataclass
class BaseBatchSize(metaclass=ABCMeta):

    def __getitem__(self, item: int) -> int:
        return self.batch_sizes[item]

    @cached_property
    def batch_sizes(self) -> List[int]:
        return list(asdict(self).values())


BatchSizeType = Union[Tuple[int, ...], BaseBatchSize]


@dataclass
class BatchSize(BaseBatchSize):
    batch_interior: int
    batch_terminal: int


@dataclass
class Sampler:

    terminal_time: float
    x_low: float
    x_high: float
    x_dim: int = 1

    def sample(self, batch_sizes: BatchSizeType, time: float) -> Sample:
        t_interior = np.random.uniform(low=time, high=self.terminal_time, size=(batch_sizes[0], 1))
        x_interior = np.random.uniform(low=self.x_low, high=self.x_high, size=(batch_sizes[0], self.x_dim))
        t_terminal = np.ones(shape=(batch_sizes[1], 1)) * self.terminal_time
        x_terminal = np.random.uniform(low=self.x_low, high=self.x_high, size=[batch_sizes[1], self.x_dim])
        return Sample(
            *[torch.tensor(x, device=DEVICE, dtype=torch.float32) for x in
              (t_interior, x_interior, t_terminal, x_terminal)]
            )

@dataclass
class MemorySampler(Sampler):

    sample_pool_size: int = 1000
    delete_pool_after: int = 10000

    def __post_init__(self):
        self.counter = 0
        self._sample_pool = []

    def sample(self, batch_sizes: BatchSizeType, time: float) -> Sample:
        if self.counter == self.delete_pool_after:
            self._sample_pool = []
            self.counter = 0
        else:
            self.counter += 1

        if len(self._sample_pool) == self.sample_pool_size:
            return random.choice(self._sample_pool)
        else:
            sample = super().sample(batch_sizes, time)
            self._sample_pool.append(sample)
            return sample





@dataclass
class HarrisonBatchSize(BaseBatchSize):
    n_interior_time: int
    n_interior_space: int
    n_terminal: int
    n_random_interior: int
    n_random_terminal: int


@dataclass
class HarrisonSampler(Sampler):

    def __post_init__(self):
        if self.x_dim > 1:
            raise ValueError("The Harrison Sampler only works in 1d.")

    def sample(self, batch_sizes: BatchSizeType, time: float) -> Sample:
        n_interior_time, n_interior_space, n_terminal, n_random_interior, n_random_terminal = batch_sizes

        t_interior = np.concatenate((
            np.linspace(time, self.terminal_time, n_interior_time).repeat(n_interior_space),
            np.random.uniform(low=time, high=self.terminal_time, size=n_random_interior),
        ), axis=0)[:, None]
        x_interior = np.concatenate((
            np.tile(np.linspace(self.x_low, self.x_high, n_interior_space), reps=n_interior_time),
            np.random.uniform(low=self.x_low, high=self.x_high, size=n_random_interior),
        ), axis=0)[:, None]

        t_terminal = np.ones(shape=(n_terminal + n_random_terminal, 1)) * self.terminal_time
        x_terminal = np.concatenate((
            torch.linspace(self.x_low, self.x_high, steps=n_terminal),
            np.random.uniform(low=self.x_low, high=self.x_high, size=n_random_terminal),
        ), axis=0)[:, None]

        return Sample(
            *[torch.tensor(x, device=DEVICE, dtype=torch.float32) for x in
              (t_interior, x_interior, t_terminal, x_terminal)]
        )
