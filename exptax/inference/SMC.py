import logging
from functools import partial
from typing import Tuple
import pdb

import jax
import jax.numpy as np

import blackjax
import blackjax.smc as smc
import blackjax.smc.ess as ess
import blackjax.smc.resampling as resampling
from blackjax.smc.tempered import TemperedSMCState

from exptax.base import ParticlesApprox
from exptax.models.base import BaseExperiment

PyTreeDef = jax.tree_util.PyTreeDef
PRNGKey = jax.random.PRNGKeyArray


def SMC(
    particles: ParticlesApprox,
    rng_key: PRNGKey,
    current_hist: PyTreeDef,
    n_meas: int,
    exp_model: BaseExperiment,
    no_temp: bool,
) -> Tuple[ParticlesApprox, np.ndarray]:
    thetas, weights = particles
    outer_samples, inner_samples = weights.shape
    thetas = jax.tree_util.tree_map(lambda leaf: jax.lax.collapse(leaf, 0, 2), thetas)
    weights = jax.lax.collapse(weights, 0, 2)
    # SMC step to move from d_(k-1) to d_k
    initial_state = TemperedSMCState(thetas, weights, 0.0)

    if no_temp:
        thetas, weights = smc_no_temp(
            rng_key, current_hist, initial_state, exp_model, n_meas
        )
        n_iter = 0

    else:
        n_iter, final_state = smc_inference_loop(
            rng_key, current_hist, initial_state, exp_model, n_meas
        )
        thetas, weights = final_state.particles, final_state.weights

    thetas = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((outer_samples, inner_samples, *leaf.shape[1:])),
        thetas,
    )
    weights = weights.reshape((outer_samples, inner_samples))
    return ParticlesApprox(thetas, weights), n_iter


def _normalize(log_weights):
    """Normalize log-weights into weights and return resulting weights and log-likelihood increment."""
    n = log_weights.shape[0]
    max_logw = np.max(log_weights)
    w = np.exp(log_weights - max_logw)
    w_mean = w.mean()

    log_likelihood_increment = np.log(w_mean) + max_logw

    w = w / (n * w_mean)
    return w, log_likelihood_increment


def empirical_cov(state):
    def cov_util(particles):
        mean = particles.mean(axis=0)
        centered = (particles - mean).reshape((*particles.shape[:1], 1, -1))
        _cov = np.einsum("nxy, nyx, n->xy", centered, centered, state.weights)
        return _cov

    cov = jax.tree_map(cov_util, state.particles)
    return cov


def markov_step(particles, rng_key, scale, potential_fn):
    N = jax.flatten_util.ravel_pytree(particles)[0].size
    # dic params keys are considered independents
    # For Sources
    cov = (
        2.2
        * jax.scipy.linalg.block_diag(*jax.tree_util.tree_leaves(scale))
        / np.sqrt(N)
        / 10.0
    )

    mrkv = blackjax.additive_step_random_walk.normal_random_walk(
        lambda x: potential_fn(x, rng_key), cov
    )
    state = mrkv.init(particles)
    new_state, info = mrkv.step(rng_key + 1, state)
    # jax.debug.print("Markov Info: {}, {}", info.acceptance_rate, info.is_accepted)
    # jax.debug.print("Markov Info: {}", info.acceptance_rate)
    return new_state.position


def SMC_kernel(rng_key, log_lik, log_prob, state, resampling_fn):
    """
    Do a general SMC step on particles using log_lik to generate weights without MCMC step here
    :param Callable log_lik: Log_likelihood function to evaluate on particles
    :param Pytree particles: Set of particles
    """
    NM = jax.tree_util.tree_leaves(state.particles)[0].shape[0]

    # resample
    scan_key, resampling_key = jax.random.split(rng_key, 2)
    resampling_index = resampling_fn(resampling_key, state.weights, NM)
    n_particles = jax.tree_map(lambda x: x[resampling_index], state.particles)

    # move particles with mcmc step
    keys = jax.random.split(rng_key, NM)
    step = partial(markov_step, potential_fn=log_prob)
    scale = empirical_cov(state)
    # jax.debug.print("cov: {} \n", scale)
    moved_particles = jax.vmap(step, in_axes=(0, 0, None))(n_particles, keys, scale)

    # calculate weights and normalize
    weights = jax.vmap(log_lik, in_axes=(0, 0))(moved_particles, keys)
    n_wei, _ = _normalize(weights)

    return moved_particles, n_wei


def tempered_SMC_kernel(rng_key, state, log_lik, potential):
    target_ess = 0.9
    max_delta = 1 - state.lmbda
    root_solver = smc.solver.dichotomy

    def compute_delta(state: TemperedSMCState) -> float:
        lmbda = state.lmbda
        max_delta = 1 - lmbda
        delta = ess.ess_solver(
            lambda x: log_lik(x, rng_key),
            state.particles,
            target_ess,
            max_delta,
            root_solver,
        )
        delta = np.clip(delta, 0.0, max_delta)
        return delta

    delta = compute_delta(state)
    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        jax.debug.print(
            "delta: {delta}, current tempering:{lmbda}", delta=delta, lmbda=state.lmbda
        )
    # logging.info(f"delta tempering:{delta}, current tempering:{state.lmbda}")
    lmbda = delta + state.lmbda

    def tempered_log_lik(particles, rng_key):
        return delta * log_lik(particles, rng_key)

    def tempered_log_prob(particles, rng_key):
        logprior = potential(particles)
        tempered_loglikelihood = state.lmbda * log_lik(particles, rng_key)
        return logprior + tempered_loglikelihood

    n_particles, weights = SMC_kernel(
        rng_key, tempered_log_lik, tempered_log_prob, state, resampling.systematic
    )

    return TemperedSMCState(n_particles, weights, lmbda)


def smc_inference_loop(rng_key, hist, initial_state, exp_model, n_meas):
    """
    Run the temepered SMC algorithm.
    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    :param params: Pytree of the design and hist of the model
    :param hist: Pytree of hist of the model
    :param initial_state: TemperedSMCState
    :param exp_model: class BaseExperiment
    :param n_meas: current measure number
    """

    last_data = jax.tree_map(lambda x: x[n_meas], hist)
    xi_star, y_measured = last_data["xi"], last_data["meas"]
    potential = exp_model.make_potential(hist, n_meas)
    smc_kernel = partial(
        tempered_SMC_kernel,
        log_lik=lambda theta, rng_key: exp_model.log_prob(theta, y_measured, xi_star),
        potential=potential,
    )

    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state = smc_kernel(rng_key=subk, state=state)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state


@partial(jax.jit, static_argnums=(4,))
def smc_no_temp(rng_key, hist, initial_state, exp_model, n_meas):
    last_data = jax.tree_map(lambda x: x[n_meas], hist)
    xi_star, y_measured = last_data["xi"], last_data["meas"]
    log_lik = lambda theta, rng_key: exp_model.log_prob(theta, y_measured, xi_star)
    potential = partial(
        exp_model.make_potential(hist, rng_key, n_meas),
        params_distrib=exp_model.params_distrib,
    )
    thetas, weights = SMC_kernel(
        rng_key, log_lik, potential, initial_state, resampling.systematic
    )
    return thetas, weights
