from typing import Optional

import torch
from torch import nn as nn

from torch_util import get_activation


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: Optional[str] = None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))
