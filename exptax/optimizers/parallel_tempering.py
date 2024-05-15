from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple
import logging

import matplotlib.pyplot as plt

import jax
from jax import Array, numpy as np
from jax._src.lib import pytree

from blackjax.base import SamplingAlgorithm

from exptax.optimizers.base import Optimizer
from exptax.base import ParticlesApprox

PRNGKey = jax.random.PRNGKeyArray

PyTreeDef = pytree.PyTreeDef



class PTState(NamedTuple):
    """
    Represents the state of the Parallel Tempering optimizer.

    Attributes:
        positions (PyTreeDef): The positions of the particles.
        temps (PyTreeDef): The temperatures of the procedure.
        accept_rate (PyTreeDef): The acceptance rates of the particles.
        energies (PyTreeDef): The energies of the particles.
    """

    positions: PyTreeDef
    temps: PyTreeDef
    accept_rate: PyTreeDef
    energies: PyTreeDef


@dataclass
class ParallelTempering(Optimizer):
    """
    Parallel Tempering optimizer for sampling from a probability distribution.
    This optimizer uses a Markov Chain Monte Carlo (MCMC) algorithm to sample from the distribution.
    The algorithm runs multiple chains in parallel, each at a different temperature,
    and swaps the chains periodically to improve mixing and convergence.

    Args:
    - temps (Array): Array of temperatures for each chain.
    - kernel (SamplingAlgorithm): MCMC kernel to use for each chain.
    - kernel_args (dict): Additional arguments to pass to the MCMC kernel.
    - energy (Callable): Function that computes the energy of a given state.
    - opt_steps (int): Number of optimization steps to run.
    """

    temps: Array
    kernel: SamplingAlgorithm
    kernel_args: dict
    opt_steps: int
    energy: Callable[[PyTreeDef, PyTreeDef, PRNGKey], PyTreeDef]

    def init(self, rng_key, *args) -> PTState:
        keys = jax.random.split(rng_key, self.temps.size)
        positions = jax.vmap(self.exp_model.xi)(keys)
        return PTState(
            positions, self.temps, np.zeros_like(self.temps), np.zeros_like(self.temps)
        )

    def kernel_step_t(
        self,
        positions: PyTreeDef,
        temp: PyTreeDef,
        key,
        kernel_args,
        particles: ParticlesApprox,
    ) -> PyTreeDef:
        """
        Moves the positions using a BlackJax Metropolis-Hastings kernel.

        Args:
            positions (PyTreeDef): The current state of the positions.
            temp (PyTreeDef): The current temperature.
            key : A JAX PRNG key.

        Returns:
            PyTreeDef: The new state of the positions after the kernel step.
        """
        thetas, weights = particles
        thetas = jax.tree_map(
            lambda leaf: jax.random.permutation(key, leaf, axis=1), thetas
        )
        # energy = functools.partial(self.type_loss, exp_model=self.exp_model, weights=weights, thetas=thetas)
        # TODO: change PTState arg to something without contours
        def loss(positions):
            return self.energy(particles, positions, key)

        mkv_kernel = self.kernel(
            lambda param: (loss(param) / temp).squeeze(),
            # num_integration_steps=50,
            **kernel_args,
        )
        state = mkv_kernel.init(positions)
        new_position, info = mkv_kernel.step(key, state)

        new_position = jax.tree_util.tree_map(
            lambda p: np.where(p < 0.0, 0.0, p), new_position
        )
        new_position = jax.tree_util.tree_map(
            lambda p: np.where(p > 100.0, 100.0, p), new_position
        )
        # jax.debug.print("{}", info)
        # pdb.set_trace()

        return new_position.position

    # change state with MCMC
    def parallel_step(
        self, key: PRNGKey, state: PTState, particles: ParticlesApprox
    ) -> PTState:
        """
        Parallel step of PT where we update all chains.

        Args:
        - state: A tuple containing the current positions, temperatures, and a PRNG key.

        Returns:
        - A tuple containing the new positions, temperatures, and a new PRNG key.
        """
        positions, temps, accept_rate, energies = state
        key, subk = jax.random.split(key, 2)

        n_t = temps.size
        keys = jax.random.split(key, n_t)

        new_positions = jax.vmap(self.kernel_step_t, in_axes=(0, 0, 0, 0, None))(
            positions, temps, keys, self.kernel_args, particles
        )

        return PTState(new_positions, temps, accept_rate, energies)

    # else propose change in T
    def swapping_step(
        self, key: PRNGKey, state: PTState, particles: ParticlesApprox
    ) -> PTState:
        """
        Performs an odd/even swap of temperatures in the given state.

        Args:
        - state: a tuple containing the current positions, temperatures, and a random key.

        Returns:
        - A tuple containing the new positions, temperatures, and a new PRNG key.
        """
        # positions, temps, key = state
        positions, temps, accept_rate, energies = state
        key, subk = jax.random.split(key, 2)
        # energy = functools.partial(self.type_loss, exp_model=self.exp_model, weights=weights, thetas=thetas)
        def loss(positions, key):
            return self.energy(particles, positions, key)

        # get proposal swap in temps
        n_t = temps.size
        u = jax.random.bernoulli(key, 0.5)
        x = np.array([i for i in range(n_t)])
        idx = np.where((x % 2) == u, x + 1, x - 1)
        swap = np.clip(idx, a_min=0, a_max=n_t)
        swap_temps = np.stack([temps, temps[swap]], axis=1)

        # calculate accept prob for each candidate swap
        def f(eng, Ts):
            return -eng * (1 / Ts[0] - 1 / Ts[1])
        keys = jax.random.split(key, n_t)
        energies = jax.vmap(loss, in_axes=(0, 0))(positions, keys)
        all_exp = jax.vmap(f, in_axes=(0, 0))(energies, swap_temps)
        accept_probs = np.clip(np.exp(all_exp + all_exp[swap]), a_min=0, a_max=1)

        # randomly accept or reject swap based on accept_probs
        # do_accept = jax.random.bernoulli(rng_key, accept_probs) doesn't work
        # as it might return a (False, True) pair for a same swap
        swap_idx = np.stack([x, swap], axis=1)
        acpt = np.tile(jax.random.uniform(key, (n_t, 1)), 2)
        do_accept = (acpt <= accept_probs[swap_idx])[:, 0]
        swapped_pos = jax.tree_util.tree_map(
            lambda arr: jax.vmap(np.where)(do_accept, arr[swap], arr), positions
        )
        swapped_eng = jax.vmap(np.where)(do_accept, energies[swap], energies)
        # pdb.set_trace()
        return PTState(swapped_pos, temps, accept_probs, swapped_eng)

    # @jax.jit
    def step(self, key: PRNGKey, state: PTState, particles: ParticlesApprox) -> Tuple:
        """
        Perform a single optimization step.

        Args:
            key (PRNGKey): The random key for generating random numbers.
            state (PTState): The current state of the  PT procedure.
            particles (ParticlesApprox): The particles approximation.

        Returns:
            Tuple: A tuple containing the updated state and the same state to accumulate with scan
        """

        state = self.parallel_step(key, state, particles)
        state = self.swapping_step(key, state, particles)

        if logging.getLogger().getEffectiveLevel() == logging.INFO:
            for i in range(state.temps.size):
                jax.debug.print(
                    "pos {}: {}, {}",
                    i,
                    np.ravel(
                        jax.tree_flatten(
                            jax.tree_map(lambda arr: arr[i], state.positions)
                        )[0][0]
                    ),
                    state.energies[i],
                )

        # TODO: don't include state.particles in accumulator
        return state, state

    def run(
        self,
        key: PRNGKey,
        state: PTState,
        particles: ParticlesApprox,
    ):
        """
        Run optimization and return the optimum and history of the procedure.

        Args:
        - key: a JAX PRNGKey used as the random key for the optimization.
        - positions: an array of initial positions for the optimization.

        Returns:
        - end_state: the final state of the optimization.
        - hist: the history of the optimization procedure.
        """
        keys = jax.random.split(key, self.opt_steps)
        end_state, hist = jax.lax.scan(
            lambda state, key: self.step(key, state, particles), state, keys
        )
        # print(end_positions)
        xi_star = jax.tree_map(lambda arr: arr[0], end_state.positions)

        return xi_star, hist

    def logger(self, xi_star, hist, particles, n_meas: int, show_plot=False):
        """
        Logs the history of the optimization procedure.
        """
        N = self.temps.size
        fig, axs = plt.subplots(nrows=N, ncols=1, figsize=(8, 17))

        # ax2.scatter(xi_star["pos"], plotter_loss(xi_star, rng_key), color="red", zorder=1)
        # ax2.scatter(positions["pos"], plotter_loss(positions, rng_key) , color="green", zorder=1)
        # plt.show()
        self.exp_model.plot_energy(fig, axs, particles, self.energy)

        for i in range(self.temps.size):
            ax = axs[i]
            hist_i = jax.tree_map(lambda arr: arr[:, i], hist)
            positions, temps, accept_rate, energies = hist_i
            self.exp_model.plot_opt(fig, ax, positions, energies)
            self.exp_model.plot_inference(fig, ax, particles, xi_star, n_meas)
            # ax.plot(space["pos"], np.exp(energie_domain / self.temps[i]), zorder=0)

            self.writer.add_figure(f"xi_{i}", fig, n_meas, close=(not show_plot))
            for idx in range(self.opt_steps):
                self.writer.add_scalar(
                    f"potential_{i}",
                    energies[idx].item(),
                    n_meas * self.opt_steps + idx,
                )
                self.writer.add_scalar(
                    f"accept_rate_{i}",
                    accept_rate[idx].item(),
                    n_meas * self.opt_steps + idx,
                )
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)
        plt.close()
