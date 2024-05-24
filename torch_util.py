from typing import Optional, Any
from pathlib import Path

import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_activation(name: Optional[str]) -> nn.Module:
    if not name:
        return nn.Identity()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'ReLU':
        return nn.ReLU()
    raise AttributeError(f'{name} is not specified as an activation')


def save_trained_state(model: Any, directory: str, name: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory + '/' + name + '.pt')


def load_trained_state(model: Any, directory: str, name: str) -> None:
    model.load_state_dict(torch.load(directory + '/' + name + '.pt'))
