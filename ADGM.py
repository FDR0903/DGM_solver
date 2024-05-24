from dataclasses import dataclass
from decimal import Decimal
from operator import itemgetter
from typing import Optional, Any
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm


from model.approximation.FunctionModel import FunctionModel
from model.pde.PdeSolver import Evaluation
from model.pde.PdeSolver import PdeSolver
from model.pde.PdeSolver import StandardEvaluation, EvalType, StandardPdeConfig
from model.pde.Sampler import Sample, Sampler
from torch_util import save_trained_state
from model.ImpactModel import ImpactModel

@dataclass
class ADGMSolverConfig(StandardPdeConfig):
    phi:            float
    sigma:          float
    alpha:          float
    tc_exponent:    float
    rc_exponent:    float

class ADGMSolver(PdeSolver[StandardEvaluation, StandardPdeConfig]):
    def __init__(
            self,
            drift:          ImpactModel,
            value_function: FunctionModel,
            control:        FunctionModel,
            sampler:        Sampler,
            config:         ADGMSolverConfig,
            pde_optimizer:  Optional[torch.optim.Optimizer] = None,
            loss_weights:   Optional[np.ndarray[Any, float]] = None,
    ):
        super().__init__(drift, value_function, control, sampler, config, pde_optimizer, loss_weights)
        
    def compute_running_cost(self, sample: Sample, e: EvalType):
        return self.config.phi * torch.pow(torch.abs(e.u), self.config.rc_exponent)
        
    @staticmethod
    def _compute_terminal_cost(x: torch.Tensor, config: ADGMSolverConfig) -> torch.Tensor:
        return - config.alpha * x ** config.tc_exponent

    def compute_control_loss(self, e: EvalType, rc: torch.Tensor, rc_u: torch.Tensor) -> torch.Tensor:
        return e.v_x * e.d_u + e.d_u - rc_u
        
    def compute_hjb_loss(self, e: EvalType, rc: torch.Tensor, rc_u: torch.Tensor) -> torch.Tensor:
        return e.v_t + self.config.sigma ** 2 * .5 * e.v_xx\
                    + e.v_x * e.d + e.d - rc

    def compute_terminal_loss(self, sample, tc) -> torch.Tensor:
        return self.value_function(sample.t_terminal, sample.x_terminal) - tc



def train_pde_solver_adgm(
        solver: PdeSolver,
        n_training: int,
        batch_sizes: Tuple[int, ...],
        directory: str,
        save_every: int,
        weight_every: int,
        scheduler: ExponentialLR = None,
        reweight:bool=False,
) -> None:
    tqdm_training = tqdm(range(n_training))
    losses = {}
    for i in tqdm_training:
        loss = solver.pde_step(batch_sizes,  time=0)
        tqdm_training.set_postfix(loss.mse_dict)
        
        if (i + 1) % weight_every == 0:
            # sort losses & put highest weight on loss
            if reweight:
                loss_items = [x.item() for x in [loss.hjb_mse, loss.control_mse, loss.terminal_mse]]
                solver.loss_weights = np.array(loss_items)/np.sum(loss_items)

            if scheduler:
                scheduler.step()

        if (i + 1) % save_every == 0:
            save_trained_state(solver.value_function, directory, f'solution_{(i + 1) // save_every + 1:02}')
            save_trained_state(solver.control, directory, f'control_{(i + 1) // save_every + 1:02}')
        
        losses[i] = loss

    return losses