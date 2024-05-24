from _py_abc import ABCMeta
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn as nn

from model.approximation.Preprocesing import IdentityInput, IdentityOutput, InputTransformation, \
    OutputTransformation
from model.approximation.FF import FeedForwardBlock
from model.approximation.LSTM import LSTMLayer


class FunctionModel(nn.Module, metaclass=ABCMeta):

    def __init__(
            self,
            input_transform: Optional[InputTransformation] = None,
            output_transform: Optional[OutputTransformation] = None,
    ):
        super().__init__()
        self.input_transform = input_transform if input_transform else IdentityInput()
        self.output_transform = output_transform if output_transform else IdentityOutput()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        input_state = self.input_transform(x)
        input_state = torch.cat([t, input_state], 1)
        output_state = self._forward(input_state)
        return self.output_transform(x, output_state)

    @abstractmethod
    def _forward(self, input_state: torch.Tensor) -> torch.Tensor:
        pass


class DGM(FunctionModel):
    def __init__(
            self,
            input_dim:   int,
            n_layers:    int,
            hidden_dim:  int,
            output_dim:  int,
            final_activ: str = None,
            input_transform: Optional[InputTransformation] = None,
            output_transform: Optional[OutputTransformation] = None,
    ):
        super().__init__(input_transform, output_transform)

        self.input_dim   = input_dim
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        
        self.first_layer = FeedForwardBlock(input_dim + 1, hidden_dim)
        self.lstm_layers = nn.ModuleList([LSTMLayer(input_dim + 1, hidden_dim) for _ in range(self.n_layers)])
        self.final_layer = FeedForwardBlock(hidden_dim, output_dim, activation=final_activ)
        
        self.first_layer = FeedForwardBlock(input_dim + 1, hidden_dim)
        self.lstm_layers = nn.ModuleList([LSTMLayer(input_dim + 1, hidden_dim) for _ in range(self.n_layers)])
        self.final_layer = FeedForwardBlock(hidden_dim, output_dim, activation=final_activ)
    
    def _forward(self, input_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.first_layer(input_state)

        for layer in self.lstm_layers:
            hidden_state = layer(hidden_state, input_state)

        return self.final_layer(hidden_state)


class MLP(FunctionModel):

    def __init__(
            self,
            input_dim: int,
            n_layers: int,
            hidden_dim: int,
            output_dim: int,
            activ: str = 'tanh',
            final_activ: str = None,
            input_transform: Optional[InputTransformation] = None,
            output_transform: Optional[OutputTransformation] = None,
    ):
        super().__init__(input_transform, output_transform)

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.first_layer = FeedForwardBlock(input_dim + 1, hidden_dim, activ)
        self.ff_layers = nn.ModuleList([FeedForwardBlock(hidden_dim, hidden_dim, activ) for _ in range(self.n_layers)])
        self.final_layer = FeedForwardBlock(hidden_dim, output_dim, activation=final_activ)

    def _forward(self, input_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.first_layer(input_state)

        for layer in self.ff_layers:
            hidden_state = layer(hidden_state)

        return self.final_layer(hidden_state)

class FunctionModel_timeoutput(nn.Module, metaclass=ABCMeta):

    def __init__(
            self,
            input_transform: Optional[InputTransformation] = None,
            output_transform: Optional[OutputTransformation] = None,
    ):
        super().__init__()
        self.input_transform = input_transform if input_transform else IdentityInput()
        self.output_transform = output_transform if output_transform else IdentityOutput()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        input_state = self.input_transform(x)
        input_state = torch.cat([t, input_state], 1)
        output_state = self._forward(input_state)
        # return self.output_transform(torch.cat([t, x],1), output_state)
        return self.output_transform(t, x, output_state)

    @abstractmethod
    def _forward(self, input_state: torch.Tensor) -> torch.Tensor:
        pass


class DGM_t(FunctionModel_timeoutput):

    def __init__(
            self,
            input_dim: int,
            n_layers: int,
            hidden_dim: int,
            output_dim: int,
            final_activ: str = None,
            input_transform: Optional[InputTransformation] = None,
            output_transform: Optional[OutputTransformation] = None,
    ):
        super().__init__(input_transform, output_transform)

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.first_layer = FeedForwardBlock(input_dim + 1, hidden_dim)
        self.lstm_layers = nn.ModuleList([LSTMLayer(input_dim + 1, hidden_dim) for _ in range(self.n_layers)])
        self.final_layer = FeedForwardBlock(hidden_dim, output_dim, activation=final_activ)

    def _forward(self, input_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.first_layer(input_state)

        for layer in self.lstm_layers:
            hidden_state = layer(hidden_state, input_state)

        return self.final_layer(hidden_state)


class MLP_t(FunctionModel_timeoutput):

    def __init__(
            self,
            input_dim: int,
            n_layers: int,
            hidden_dim: int,
            output_dim: int,
            activ: str = 'tanh',
            final_activ: str = None,
            input_transform: Optional[InputTransformation] = None,
            output_transform: Optional[OutputTransformation] = None,
    ):
        super().__init__(input_transform, output_transform)

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.first_layer = FeedForwardBlock(input_dim + 1, hidden_dim, activ)
        self.ff_layers = nn.ModuleList([FeedForwardBlock(hidden_dim, hidden_dim, activ) for _ in range(self.n_layers)])
        self.final_layer = FeedForwardBlock(hidden_dim, output_dim, activation=final_activ)

    def _forward(self, input_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.first_layer(input_state)

        for layer in self.ff_layers:
            hidden_state = layer(hidden_state)

        return self.final_layer(hidden_state)
