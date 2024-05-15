from typing import Callable, NamedTuple
from exptax.base import ParticlesApprox


class InferenceAlgorithm(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable


class Inference_state(NamedTuple):
    particles: ParticlesApprox
    inference_args: dict


class inference_step(NamedTuple):
    init: Callable
    run: Callable
