from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Tuple, Optional, Dict, Any, TypeVar, Generic, Type, Callable

import numpy as np
import torch
from torch.autograd import grad

from model.ImpactModel import ImpactModel
from model.approximation.FunctionModel import FunctionModel
from model.pde.Sampler import Sample, Sampler, BatchSizeType


@dataclass
class PdeConfig(metaclass=ABCMeta):
    pass

    @classmethod
    @abstractmethod
    def with_default_values(cls) -> 'PdeConfig':
        pass

@dataclass
class StandardPdeConfig(PdeConfig):
    @classmethod
    def with_default_values(cls) -> 'PdeConfig':
        return cls()


PdeConfigType = TypeVar('PdeConfigType', bound=PdeConfig)


class Evaluation:

    def __init__(self, value_function: FunctionModel, control: FunctionModel, drift: ImpactModel, sample: Sample):
        sample.require_grad(True)
        self._init(value_function, control, drift, sample)
        sample.require_grad(False)
    
    def _init(self, value_function: FunctionModel, control: FunctionModel, drift: ImpactModel, sample: Sample):
        self.v = value_function(sample.t_interior, sample.x_interior)
        self.u = control(sample.t_interior, sample.x_interior)
        self.d = drift.evaluate(sample.x_interior, self.u)


class StandardEvaluation(Evaluation):

    def _init(self, value_function: FunctionModel, control: FunctionModel, drift: ImpactModel, sample: Sample):
        super()._init(value_function, control, drift, sample)

        self.v_t = grad(self.v, [sample.t_interior], grad_outputs=torch.ones_like(self.v), create_graph=True)[0]
        self.v_x = grad(self.v, [sample.x_interior], grad_outputs=torch.ones_like(self.v), create_graph=True)[0]
        self.v_xx = grad(self.v_x, [sample.x_interior], grad_outputs=torch.ones_like(self.v_x), create_graph=True)[0]
        self.d_u = grad(self.d, [self.u], grad_outputs=torch.ones_like(self.d), create_graph=True)[0]


EvalType = TypeVar('EvalType', bound=Evaluation)


@dataclass
class PdeLoss:
    hjb: torch.Tensor
    control: torch.Tensor
    terminal: torch.Tensor
    loss_weights: np.array

    @cached_property
    def hjb_mse(self) -> torch.Tensor:
        return (self.hjb ** 2).mean()

    @cached_property
    def control_mse(self) -> torch.Tensor:
        return (self.control ** 2).mean()

    @cached_property
    def terminal_mse(self) -> torch.Tensor:
        return (self.terminal ** 2).mean()

    @property
    def mse(self) -> torch.Tensor:
        return self.hjb_mse * self.loss_weights[0] + self.control_mse * self.loss_weights[1] + self.terminal_mse * \
               self.loss_weights[2]

    @property
    def mse_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'HJB': self.hjb_mse.item(),
            'Hamiltonian': self.control_mse.item(),
            'Terminal': self.terminal_mse.item(),
            'Total': self.mse.item(),
        }


class PdeSolver(Generic[EvalType, PdeConfigType], metaclass=ABCMeta):

    eval_type: EvalType = StandardEvaluation
    config_type: PdeConfigType = PdeConfig

    def __init__(
            self,
            drift: ImpactModel,
            value_function: FunctionModel,
            control: FunctionModel,
            sampler: Sampler,
            config: PdeConfigType = None,
            pde_optimizer: Optional[torch.optim.Optimizer] = None,
            loss_weights: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.drift = drift
        self.value_function = value_function
        self.control = control
        self.sampler = sampler
        self.config = config if config else self.config_type.with_default_values()
        self.pde_optimizer = pde_optimizer if pde_optimizer else torch.optim.Adam(
            [*value_function.parameters(), *control.parameters()],
        )
        self.loss_weights = loss_weights if loss_weights else np.ones(3)

    def pde_step(self, batch_sizes: BatchSizeType, time) -> PdeLoss:
        sample = self.sampler.sample(batch_sizes, time)
        

        pde_loss = self.pde_loss(sample)

        pde_loss.mse.backward()

        self.pde_optimizer.step()
        self.pde_optimizer.zero_grad()

        return pde_loss

    def pde_loss(self, sample: Sample) -> PdeLoss:
        e = self.eval_type(self.value_function, self.control, self.drift, sample)
        rc, rc_u = self.compute_running_cost_with_gradient(sample, e)
        tc = self.compute_terminal_cost(sample)
        return PdeLoss(
            hjb=self.compute_hjb_loss(e, rc, rc_u),
            control=self.compute_control_loss(e, rc, rc_u),
            terminal=self.compute_terminal_loss(sample, tc),
            loss_weights=self.loss_weights,
        )

    def compute_running_cost_with_gradient(self, sample: Sample, e: Evaluation) -> Tuple[torch.Tensor, torch.Tensor]:
        rc = self.compute_running_cost(sample, e)
        rc_u = grad(rc, [e.u], grad_outputs=torch.ones_like(rc), create_graph=True)[0]
        return rc, rc_u

    @abstractmethod
    def compute_running_cost(self, sample: Sample, e: EvalType) -> torch.Tensor:
        pass

    def compute_terminal_cost(self, sample: Sample) -> torch.Tensor:
        return self._compute_terminal_cost(sample.x_terminal, self.config)

    @staticmethod
    @abstractmethod
    def _compute_terminal_cost(x: torch.Tensor, config: PdeConfigType) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_control_loss(self, e: EvalType, rc: torch.Tensor, rc_u: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_hjb_loss(self, e: EvalType, rc: torch.Tensor, rc_u: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_terminal_loss(self, sample: Sample, tc: torch.Tensor) -> torch.Tensor:
        pass

    def fit_data_transforms(self, batch_sizes: BatchSizeType, time) -> None:
        sample = self.sampler.sample(batch_sizes, time)

        self.value_function.input_transform.fit(sample.x_interior)
        self.control.input_transform.fit(sample.x_interior)

        self.value_function.output_transform.fit(self.compute_terminal_cost(sample))
        # self.control.output_transform.fit()  # <- No output transform.

    @classmethod
    def get_terminal_condition(cls, config: Optional[PdeConfigType] = None) -> Callable[[torch.Tensor], torch.Tensor]:
        return partial(cls._compute_terminal_cost, config=config if config else cls.config_type())

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "pde_optimizer":
                setattr(result, k, deepcopy(v, memo))

        result.pde_optimizer = self.pde_optimizer.__class__(
            [*result.control.parameters(), *result.value_function.parameters()]
        )
        result.pde_optimizer.load_state_dict(self.pde_optimizer.state_dict())

        return result
