from _py_abc import ABCMeta
from abc import abstractmethod
from typing import Tuple, TypeVar
from typing import Callable, Tuple
import torch
from torch import nn
from torch import Tensor


class ImpactModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.c_time = None

    def set_current_time(self, time) -> None:
        self.c_time = time

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self._forward(torch.cat((x, u), dim=1))

    @abstractmethod
    def _forward(self, input_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def evaluate(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass


ImpactModelType = TypeVar('ImpactModelType', bound=ImpactModel)


class DeterministicImpact(ImpactModel):

    def __init__(
            self,
            impact_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.map = impact_function

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[Tensor, ...]:
        return self.map(x, u),

    def evaluate(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.forward(x, u)[0]

    def _forward(self, input_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass