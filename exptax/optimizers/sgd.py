from dataclasses import dataclass
from typing import Callable, NamedTuple

import optax
import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax._src.lib import pytree
from jax_tqdm import scan_tqdm


from exptax.optimizers.base import Optimizer
from exptax.base import ParticlesApprox

PRNGKey = jax.random.PRNGKeyArray

PyTreeDef = pytree.PyTreeDef


class SGDState(NamedTuple):
    """
    Represents the state of the SGD optimizer.

    Attributes:
        positions (PyTreeDef): The positions of the parameters.
        opt_state (PyTreeDef): The optimizer state.
        loss_val (float, optional): The loss value. Defaults to 0.
    """

    positions: PyTreeDef
    opt_state: PyTreeDef
    loss_val: float = 0.0


@dataclass
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Attributes:
    -----------
    opt_steps : int
        Number of optimization steps to perform.
    n_meas : int
        Number of measurements.
    step : Callable
        Function that performs a single optimization step.
    opt_args : dict
        Arguments to pass to the optimizer builder.
    opt_builder : Callable
        Function that builds the optimizer.

    Methods:
    --------
    run(start_point: PyTreeDef) -> PyTreeDef:
        Runs the optimizer on the given starting point and returns the optimized parameters,
        a dictionary of parameter histories, and the final loss value.
    """

    opt_steps: int
    opt_args: dict
    opt_builder: Callable
    energy: Callable[[PyTreeDef, PyTreeDef, PRNGKey], PyTreeDef]

    def __post_init__(self):
        self.optx_opt = self.opt_builder(**self.opt_args)
        self.optx_opt = optax.chain(
            optax.zero_nans(), optax.adam(**self.opt_args), optax.scale(-1)
        )

    def init(
        self, rng_key: PRNGKey, particles: ParticlesApprox, state: SGDState = None
    ) -> SGDState:
        positions = self.exp_model.xi_part(particles, rng_key)
        opt_state = self.optx_opt.init(positions)
        return SGDState(positions, opt_state)

    def step(
        self, rng_key: PRNGKey, state: SGDState, particles: ParticlesApprox
    ) -> SGDState:
        # Estimate PCE with current particle approximation
        # diff / xi to run gradient optimisation step
        # thetas = jax.tree_map(lambda leaf:jax.random.permutation(rng_key, leaf, axis=1), thetas)
        # weights = jax.random.permutation(rng_key, weights, axis=1)

        positions, opt_state, _ = state
        def loss(x):
            return self.energy(particles, x, rng_key)
        loss_value, grads = jax.value_and_grad(loss)(positions)

        updates, opt_state = self.optx_opt.update(grads, opt_state, positions)
        params = optax.apply_updates(positions, updates)
        # jax.debug.print('params: {}', params)
        n_state = SGDState(params, opt_state, loss_value)
        return n_state, n_state

    def run(
        self,
        rng_key: PRNGKey,
        state: SGDState,
        particles: ParticlesApprox,
    ) -> PyTreeDef:
        # self.log_hyperparams()

        @scan_tqdm(self.opt_steps)
        def step(state, tup):
            _, key = tup
            return self.step(key, state, particles)

        keys = jax.random.split(rng_key, self.opt_steps)
        end_state, hist = jax.lax.scan(
            step, state, (np.arange(0, self.opt_steps), keys)
        )

        return end_state.positions, hist

    def logger(
        self,
        xi_star: PyTreeDef,
        hist: PyTreeDef,
        particles: ParticlesApprox,
        n_meas: int,
        show_plot=False,
    ):
        """
        Logs the history of the optimization procedure.
        """
        fig, ax = plt.subplots()

        # ax2.scatter(xi_star["pos"], plotter_loss(xi_star, rng_key), color="red", zorder=1)
        # ax2.scatter(positions["pos"], plotter_loss(positions, rng_key) , color="green", zorder=1)
        # plt.show()
        self.exp_model.plot_energy(fig, [ax], particles, self.energy)

        positions, opt_state, energies = hist
        self.exp_model.plot_opt(fig, ax, positions, energies)
        self.exp_model.plot_inference(fig, ax, particles, xi_star, n_meas)
        # self.exp_model.plot_particles(fig, ax, particles.thetas, particles.weights)
        # ax.plot(space["pos"], np.exp(energie_domain / self.temps[i]), zorder=0)

        plt.tight_layout()
        self.writer.add_figure("xi", fig, n_meas, close=(not show_plot))
        plt.show(block=False)
        plt.pause(0.3)
        plt.close()
        for idx in range(self.opt_steps):
            self.writer.add_scalar(
                "potential", energies[idx].item(), n_meas * self.opt_steps + idx
            )
