from dataclasses import dataclass
from typing import Callable, NamedTuple

from torch.utils.tensorboard import SummaryWriter

from jaxtyping import PyTree, PRNGKeyArray

from exptax.models.base import BaseExperiment
from exptax.base import ParticlesApprox


# TODO: line search SGD + preconditionned SGLD + think of flat landscape for CSLGD
# Learning with Differentiable Perturbed Optimizers
# in CSGLD, think of scaling variance of random walk with flatness of landscape

class Optim(NamedTuple):
    """
    Wrapper around key functions for an optimizer:
    - init: initialize the optimizer
    - run: run the optimizer
    - logger: log values of the optimizer
    - writer: optional tensorboard writer
    """

    init: Callable[[PRNGKeyArray, ParticlesApprox], NamedTuple]
    run: Callable[[PRNGKeyArray, NamedTuple, ParticlesApprox], PyTree]
    logger: Callable[[PyTree, PyTree, ParticlesApprox], None]
    writer: SummaryWriter = None


@dataclass
class Optimizer:
    """
    Base class for all optimizers.

    Attributes:
        exp_model (BaseExperiment): The experiment model to optimize.
        writer (SummaryWriter): The summary writer to log results.
    """

    exp_model: BaseExperiment
    writer: SummaryWriter

    def hyperparams(self):
        return [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]

    def log_hyperparams(self):
        self.writer.add_text("hyperparams", str(self.hyperparams()))
