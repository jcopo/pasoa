from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List
import jax
import jax.numpy as np
from jaxtyping import PyTree
import matplotlib.pyplot as plt

from exptax.base import ParticlesApprox



class BaseExperiment(ABC):
    """
    Base class for experiments
    :param rng_key: random key gen
    :param params_distrib: Dic of parameters name and corresponding distribs
    """

    def __init__(self, rng_key, params_distrib):
        self.params_distrib = params_distrib
        r_keys = jax.random.split(rng_key, len(params_distrib))
        self.ground_truth = {
            key: params_distrib[key].sample(rng) for key, rng in zip(params_distrib, r_keys)
        }

    @abstractmethod
    def sample(self, args, rng_key, xi):
        pass

    @abstractmethod
    def log_prob(self, theta, y, xi):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def measure(self, rng_key, xi):
        return self.sample(self.ground_truth, rng_key, xi)

    def make_potential(self, hist:PyTree, n_meas:int)->Callable:
        """
        Helper function to build potential for MetropolisHastings kernel
        :param hist: list of Dk, ie past measurement locations and values. Format [{"xi":xi_k, "y":y_k} for k in range(K)]]
        :param rng_key: random key
        :param n_meas: current measure number
        """

        def potential(args):
            # apply p(y|args, xi) to all y_k, xi_k
            log_lik = lambda xi, y: self.log_prob(args, y, xi)

            xi_ks = hist["xi"]
            y_ks = hist["meas"]
            # return log( cumprod p(y| theta, xi)_k * prior(theta) )
            prior_term = jax.tree_map(
                lambda value, dis: dis.log_prob(value), args, self.params_distrib
            )

            def apply_cond(index, xi, y):
                # jax.debug.print("lk: {}", log_lik(xi, y))
                return jax.lax.cond(index < n_meas, log_lik, lambda *_: 0.0, xi, y)

            indices = np.arange(y_ks.shape[0])
            sums = jax.vmap(apply_cond)(indices, xi_ks, y_ks)

            return sums.sum() + jax.tree_util.tree_reduce(np.add, prior_term).squeeze()

        return potential

    def plot_energy(self, fig:plt.Figure, axs:List[plt.Axes], particles:ParticlesApprox, energy:Callable)->None:
        pass

    def plot_opt(self, fig:plt.Figure, ax:plt.Axes, positions:PyTree, energies:PyTree)->None:
        pass

    def plot_inference(self, fig:plt.Figure, ax:plt.Axes, particles:ParticlesApprox, xi_star:PyTree, n_meas:int, writer=None, title=None)->None:
        pass