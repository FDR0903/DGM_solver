from typing import Tuple

import torch
from torch import nn as nn

from torch_util import get_activation


class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation_1: str = 'tanh', activation_2: str = 'tanh'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_1 = get_activation(activation_1)
        self.activation_2 = get_activation(activation_2)

        self.u = LSTMParameter((self.input_dim, self.output_dim))
        self.w = LSTMParameter((self.output_dim, self.output_dim))
        self.b = LSTMParameter((1, self.output_dim))

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = self.activation_1((x @ self.u.z) + (s @ self.w.z) + self.b.z)
        g = self.activation_1((x @ self.u.g) + (s @ self.w.g) + self.b.g)
        r = self.activation_1((x @ self.u.r) + (s @ self.w.r) + self.b.r)
        h = self.activation_2((x @ self.u.h) + ((s * r) @ self.w.h) + self.b.h)

        return (torch.ones_like(g) - g) * h + (z * s)


class LSTMParameter(nn.Module):

    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.z = nn.Parameter(torch.ones(size=size))
        self.g = nn.Parameter(torch.ones(size=size))
        self.r = nn.Parameter(torch.ones(size=size))
        self.h = nn.Parameter(torch.ones(size=size))
