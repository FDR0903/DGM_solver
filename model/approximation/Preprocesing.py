from abc import ABCMeta, abstractmethod
from typing import Callable

import torch.nn

from torch_util import DEVICE


class Transformation(torch.nn.Module, metaclass=ABCMeta):
    pass

    @abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        pass


class InputTransformation(Transformation, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class IdentityInput(InputTransformation):

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def fit(self, data: torch.Tensor) -> None:
        pass


class StandardScale(InputTransformation):

    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(0.0, device=DEVICE), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.tensor(1.0, device=DEVICE), requires_grad=False)

    def fit(self, data: torch.Tensor) -> None:
        self.mean.data = torch.mean(data, dim=0)
        self.scale.data = torch.std(data, dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) * self.scale


class OutputTransformation(Transformation, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        pass


class IdentityOutput(OutputTransformation):

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return outputs

    def fit(self, data: torch.Tensor) -> None:
        pass


class StandardBackwardScale(OutputTransformation, StandardScale):

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return (outputs + self.mean) / self.scale


class FunctionBackwardShift(OutputTransformation):

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = function

    def fit(self, data: torch.Tensor) -> None:
        pass

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return outputs + self.fn(inputs)

class FunctionBackwardShift_t(OutputTransformation):

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = function

    def fit(self, data: torch.Tensor) -> None:
        pass

    def forward(self, t: torch.Tensor,x: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return outputs + self.fn(t, x)

class FunctionBackwardShiftAndScale(FunctionBackwardShift):

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(function)
        self.scale = 1.0

    def fit(self, data: torch.Tensor) -> None:
        self.scale = torch.std(data, dim=0)

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs, outputs * self.scale)
