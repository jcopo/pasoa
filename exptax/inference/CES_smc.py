import logging
from functools import partial
from typing import Tuple

import jax
import jax.numpy as np
import blackjax
import blackjax.smc as smc
import blackjax.smc.resampling as resampling
import blackjax.smc.ess as ess
from blackjax.smc.tempered import TemperedSMCState

from exptax.base import ParticlesApprox
from exptax.models.base import BaseExperiment

PyTreeDef = jax.tree_util.PyTreeDef
PRNGKey = jax.random.PRNGKeyArray


def SMC_CES(
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

    thetas = {
        "alpha": jax.vmap(additive_logistic)(thetas["alpha"]),
        "rho": jax.scipy.special.logit(thetas["rho"]),
        "u": thetas["u"],
    }
    thetas = jax.tree_map(
        lambda array: array + 0.01 * jax.random.normal(rng_key, array.shape), thetas
    )

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

    thetas = {
        "alpha": jax.vmap(reverse_additive_logistic)(thetas["alpha"]),
        "rho": jax.scipy.special.expit(thetas["rho"]),
        "u": thetas["u"],
    }
    thetas = jax.tree_util.tree_map(
        lambda leaf: jax.random.permutation(rng_key, leaf), thetas
    )
    thetas = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((outer_samples, inner_samples, *leaf.shape[1:])),
        thetas,
    )
    weights = weights.reshape((outer_samples, inner_samples))
    return ParticlesApprox(thetas, weights), n_iter


def additive_logistic(alpha):
    alpha_d = alpha[-1]
    return np.log(alpha[:-1]) - np.log(alpha_d)


def reverse_additive_logistic(y):
    exp_y = np.exp(y)
    alpha = np.hstack([exp_y, np.array([1.0])])
    return alpha / alpha.sum()


def _make_SMC_step(scale):
    def SMC_step(rng_key, position):
        alpha_i = lambda p_i, v_i: np.nan_to_num(p_i * (p_i * (1 - p_i) / (v_i) - 1))
        alpha_dirichlet = jax.vmap(alpha_i)(
            position["alpha"], 2.2 * np.diagonal(scale["alpha"]) / np.sqrt(3)
        )
        rho_dirichlet = jax.vmap(alpha_i)(position["rho"], np.diagonal(scale["rho"]))
        dx = {
            "u": generate_gaussian_noise(rng_key, position["u"]),
            "alpha": jax.random.dirichlet(rng_key, alpha_dirichlet) - position["alpha"],
            "rho": jax.random.dirichlet(rng_key, rho_dirichlet) - position["rho"],
        }
        # jax.debug.breakpoint()
        return dx

    return SMC_step


def SMC_step(rng_key, position):
    dx = {
        "u": generate_gaussian_noise(rng_key, position["u"]),
        "alpha": jax.random.dirichlet(rng_key, position["alpha"] / 0.1)
        - position["alpha"],
        "rho": jax.random.dirichlet(rng_key, position["rho"] / 0.1) - position["rho"],
    }
    return dx


def ces_proposal(rng_key):
    proposal = {
        "alpha": dist.Dirichlet(np.array([1, 1, 1])).sample(rng_key),
        # "rho":dist.Dirichlet(np.array([1, 1])),
        "rho": dist.Beta(np.array([1]), np.array([1])).sample(rng_key),
        "u": dist.Normal(np.array([1.0]), np.array([np.sqrt(3)])).sample(rng_key),
    }
    return proposal


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


def SMC_kernel(rng_key, log_lik, log_prob, state, resampling_fn):
    """
    Do a general SMC step on particles using log_lik to generate weights without MCMC step here
    :param Callable log_lik: Log_likelihood function to evaluate on particles
    :param Pytree particles: Set of particles
    """

    def markov_step(particles, rng_key, scale, potential_fn):
        N = jax.flatten_util.ravel_pytree(particles)[0].size
        # dic params keys are considered independents
        # For Sources
        (scale, mean) = scale
        # particles = jax.tree_map(lambda array: array + .01 * jax.random.normal(rng_key, array.shape), particles)
        # TODO essayer d'augmenter rho
        scale["rho"] *= 5.0
        cov = (
            2.2
            * jax.scipy.linalg.block_diag(*jax.tree_util.tree_leaves(scale))
            / np.sqrt(N)
            / 15.0
        )
        mean = np.concatenate(jax.tree_util.tree_leaves(mean))
        # jax.debug.print("cov: {} \n", cov)
        # jax.debug.print("mean: {} \n", mean)
        mrkv = blackjax.additive_step_random_walk.normal_random_walk(
            lambda x: potential_fn(x, rng_key), cov
        )

        # mrkv =  blackjax.mala(lambda x:potential_fn(x, rng_key), .0001)
        # mrkv =  blackjax.nuts(lambda x:potential_fn(x, rng_key), .001, cov)

        state = mrkv.init(particles)

        new_state, info = mrkv.step(rng_key + 1, state)
        # jax.debug.print("Markov Info: {}, {}", info.acceptance_rate, info.is_accepted)
        # jax.debug.print("Markov Info: {}", info.acceptance_rate)
        return new_state.position

    NM = jax.tree_util.tree_leaves(state.particles)[0].shape[0]

    # resample
    scan_key, resampling_key = jax.random.split(rng_key, 2)
    resampling_index = resampling_fn(resampling_key, state.weights, NM)
    n_particles = jax.tree_map(lambda x: x[resampling_index], state.particles)

    # move particles with mcmc step
    keys = jax.random.split(rng_key, NM)
    step = partial(markov_step, potential_fn=log_prob)
    scale = empirical_cov(state)
    mean = jax.tree_map(lambda array: array.mean(axis=0), state.particles)
    moved_particles = jax.vmap(step, in_axes=(0, 0, None))(
        n_particles, keys, (scale, mean)
    )

    # calculate weights and normalize
    weights = jax.vmap(log_lik, in_axes=(0, 0))(moved_particles, keys)
    n_wei, _ = _normalize(weights)

    return moved_particles, n_wei


def tempered_SMC_kernel(rng_key, state, log_lik, potential):
    target_ess = 0.98
    max_delta = 1 - state.lmbda
    root_solver = smc.solver.dichotomy

    inv_transform = lambda params: {"alpha": reverse_additive_logistic(params["alpha"]),
                    "rho": jax.scipy.special.expit(params["rho"]),
                    "u": params["u"]}

    def compute_delta(state: TemperedSMCState) -> float:
        lmbda = state.lmbda
        max_delta = 1 - lmbda
        delta = ess.ess_solver(
                lambda x: log_lik(jax.vmap(inv_transform)(x), rng_key),
            state.particles,
            target_ess,
            max_delta,
            root_solver,
        )
        delta = np.clip(delta, 0.0, max_delta)
        return delta
    delta = compute_delta(state)
    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        jax.debug.print("delta: {delta}, current tempering:{lmbda}", delta=delta, lmbda=state.lmbda)
    #logging.info(f"delta tempering:{delta}, current tempering:{state.lmbda}")
    lmbda = delta + state.lmbda

    def tempered_log_lik(particles, rng_key):
        particles = inv_transform(particles)
        return delta * log_lik(particles, rng_key)
    def tempered_log_prob(particles, rng_key):
        #  reparam CES
        log_det_jacob = lambda x: -(np.log((1 + np.exp(-x))**2) + x).squeeze()
        log_det_logitN = lambda x: np.log(x * (1 - x)).sum()
        log_dets = log_det_logitN(particles["alpha"]) +  log_det_jacob(particles["rho"]) #+ np.exp(particles["u"]).squeeze()

        i_particles = inv_transform(particles)

        logprior = potential(i_particles)
        tempered_loglikelihood = state.lmbda * log_lik(i_particles, rng_key)
        return logprior + tempered_loglikelihood + log_dets

    #n_particles, weights = SMC_kernel_wf(rng_key, tempered_log_lik, tempered_log_prob, state, resampling.systematic)
    n_particles, weights = SMC_kernel(rng_key, tempered_log_lik, tempered_log_prob, state, resampling.systematic)

    return TemperedSMCState(n_particles, weights, lmbda)


def SMC_kernel_wf(rng_key, log_lik, log_prob, state, resampling_fn):
    """
    Do a general SMC step on particles using log_lik to generate weights without MCMC step here
    :param Callable log_lik: Log_likelihood function to evaluate on particles
    :param Pytree particles: Set of particles
    """

    def markov_step(particles, rng_key, scale, potential_fn, waste_free_num):
        N = jax.flatten_util.ravel_pytree(particles)[0].size
        # particles = jax.tree_map(lambda array: array + .01 * jax.random.normal(rng_key, array.shape), particles)
        # dic params keys are considered independents
        # For Sources
        (scale, mean) = scale
        # scale["u"] *= 0.5
        cov = (
            2.2
            * jax.scipy.linalg.block_diag(*jax.tree_util.tree_leaves(scale))
            / np.sqrt(N)
            / 5.0
        )
        mean = np.concatenate(jax.tree_util.tree_leaves(mean))
        # jax.debug.print("cov: {} \n", cov)
        # jax.debug.print("mean: {} \n", mean)
        # mrkv =  blackjax.additive_step_random_walk.normal_random_walk(lambda x:potential_fn(x, rng_key), cov)
        mrkv = blackjax.mala(lambda x: potential_fn(x, rng_key), 0.01)

        num_steps = 100
        state = mrkv.init(particles)
        # new_state = jax.lax.fori_loop(0,
        #                            num_steps,
        #                            lambda i, state: mrkv.step(rng_key + i, state)[0],
        #                            state)

        keys = jax.random.split(rng_key, waste_free_num)
        new_state, carry = jax.lax.scan(
            lambda state, key: (mrkv.step(key, state)[0],) * 2, state, keys
        )
        new_state, info = mrkv.step(rng_key + 1, state)
        # jax.debug.print("Markov Info: {}, {}", info.acceptance_rate, info.is_accepted)
        # jax.debug.print("Markov Info: {}", info.acceptance_rate)
        return carry.position

    NM = jax.tree_util.tree_leaves(state.particles)[0].shape[0]

    # resample
    scan_key, resampling_key = jax.random.split(rng_key, 2)
    waste_free_num = 50
    wf_samples = NM // waste_free_num
    # wf_samples = NM // waste_free_num

    resampling_index = resampling_fn(resampling_key, state.weights, wf_samples)
    n_particles = jax.tree_map(lambda x: x[resampling_index], state.particles)
    # n_particles  = jax.tree_map(lambda array: array + .03 * jax.random.normal(key, array.shape), n_particles)

    n_particles = jax.tree_map(
        lambda array: jax.random.choice(resampling_key, array, shape=(wf_samples,)),
        state.particles,
    )
    # move particles with mcmc step
    keys = jax.random.split(rng_key, wf_samples)
    step = partial(markov_step, potential_fn=log_prob, waste_free_num=waste_free_num)
    scale = empirical_cov(state)
    mean = jax.tree_map(lambda array: array.mean(axis=0), state.particles)
    moved_particles = jax.vmap(step, in_axes=(0, 0, None))(
        n_particles, keys, (scale, mean)
    )

    moved_particles = jax.tree_util.tree_map(
        lambda leaf: jax.lax.collapse(leaf, 0, 2), moved_particles
    )

    # calculate weights and normalize
    keys = jax.random.split(scan_key, NM)
    weights = jax.vmap(log_lik, in_axes=(0, 0))(moved_particles, keys)
    n_wei, _ = _normalize(weights)

    return moved_particles, n_wei


# TODO: params here is useless as it is in hist
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
    log_lik = lambda theta, rng_key: exp_model.log_prob(theta, y_measured, xi_star)
    potential = exp_model.make_potential(hist, n_meas)
    smc_kernel = partial(
        tempered_SMC_kernel,
        log_lik=log_lik,
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

    cary = (0, initial_state, rng_key)

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state


# @partial(jax.jit, static_argnums=(4,))
def smc_no_temp(rng_key, params, hist, initial_state, exp_model, n_meas):
    inv_transform = lambda params: {
        "alpha": reverse_additive_logistic(params["alpha"]),
        "rho": jax.scipy.special.expit(params["rho"]),
        "u": params["u"],
    }

    last_data = jax.tree_map(lambda x: x[n_meas], hist)
    xi_star, y_measured = last_data["xi"], last_data["meas"]
    log_lik = lambda theta, rng_key: exp_model.log_prob(theta, y_measured, xi_star)
    potential = exp_model.make_potential(hist, n_meas)
    # modify log_lik and potential if reparam
    # TODO np.log to that
    log_det_logitN = lambda x: np.log(x * (1 - x)).sum()

    def r_log_lik(particles, rng_key):
        particles = inv_transform(particles)
        return log_lik(particles, rng_key)

    def r_log_prob(particles, rng_key):
        #  reparam CES
        log_det_jacob = lambda x: np.log(
            np.abs(jax.vmap(jax.grad(jax.scipy.special.expit))(x))
        ).sum()
        log_dets = (
            log_det_logitN(particles["alpha"])
            + log_det_jacob(particles["rho"])
            + np.exp(particles["u"]).squeeze()
        )
        particles = inv_transform(particles)

        logprior = potential(particles)
        return logprior + log_dets

    thetas, weights = SMC_kernel(
        rng_key, r_log_lik, r_log_prob, initial_state, resampling.systematic
    )
    return thetas, weights
