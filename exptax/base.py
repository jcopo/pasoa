from typing import NamedTuple
from jaxtyping import PyTree

class ParticlesApprox(NamedTuple):
    """
    Represents particles approximation.

    Attributes:
        thetas (PyTreeDef): The particle positions.
        weights (PyTreeDef): The particle weights.
    """

    thetas: PyTree
    weights: PyTree