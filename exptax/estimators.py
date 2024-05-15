from functools import partial

import jax
from jax import Array, numpy as np
from jax._src.lib import pytree

from exptax.models.base import BaseExperiment
from exptax.base import ParticlesApprox

PRNGKey = jax.random.PRNGKeyArray
PyTreeDef = pytree.PyTreeDef

# plt.rcParams["figure.figsize"] = (25,13)


def reinforce_pce(
    design: Array,
    rng_key: PRNGKey,
    exp_model: BaseExperiment,
    particles: ParticlesApprox,
):
    """
    Reinforce version of PCE for discrete observations

    :param design: Pytree storing design
    :param exp_model: BaseExperiment class storing our probalistic model
    :param rng_key: random key
    :param weights: SMC weights of the particles
    :param thetas: Pytree containing samples on which we evaluate the PCE bound of EIG
    :param cv: control variate
    """
    thetas, weights = particles
    thetas_sgd = jax.tree_util.tree_map(
        lambda leaf: jax.lax.collapse(leaf, 0, 2), thetas
    )
    weights_sgd = jax.lax.collapse(weights, 0, 2)
    thetas = jax.tree_util.tree_map(
        lambda arr: jax.random.choice(rng_key, arr, (10, 20), p=weights_sgd), thetas_sgd
    )
    # thetas = contrastive_sampler(exp_model, rng_key, weights, thetas)
    N, L = jax.tree_util.tree_leaves(thetas)[0].shape[:2]
    N_samples = jax.tree_map(lambda leaf: leaf[:, 0], thetas)
    # thetas = jax.tree_util.tree_map(lambda leaf: jax.lax.collapse(leaf, 0, 2), thetas)
    # weights = jax.lax.collapse(weights, 0, 2)

    ## sample N and L particles
    # samples = jax.tree_util.tree_map(lambda leaf: jax.random.choice(rng_key, leaf, p=weights, shape=(N+L,)), thetas)
    # N_samples = jax.tree_util.tree_map(lambda leaf: leaf[:N], samples)
    # L_samples = jax.tree_util.tree_map(lambda leaf: leaf[N:], samples)

    # sample N y_n
    sample_keys = jax.random.split(rng_key, N)
    N_y = jax.vmap(exp_model.sample, (0, 0, None))(N_samples, sample_keys, design)

    # corresponding NxL log p(y_n| theta_l, design)
    log_liks_nm = jax.vmap(exp_model.log_prob, in_axes=(1, None, None), out_axes=1)(
        thetas, N_y, design
    )

    # (Nx1) log probs
    log_liks_n = log_liks_nm[:, 0]

    # Nx(L+1)
    all_liks = np.hstack([np.expand_dims(log_liks_n, -1), log_liks_nm])

    # Nx1 log( sum_l p(y_n| theta_l, design))
    contr_log_lik = jax.scipy.special.logsumexp(all_liks, axis=1) - np.log(L + 1)

    log_sum = log_liks_n - contr_log_lik

    ng_logSum = jax.lax.stop_gradient(log_sum)

    return np.sum(log_liks_n * ng_logSum + log_sum) / N


@partial(jax.jit, static_argnums=(0, 4))
def bounds_eig_fix_shape(
    exp_model: BaseExperiment,
    true_theta: Array,
    hist: PyTreeDef,
    rng_key: PRNGKey,
    n_meas,
    inner_samples=int(1e7),
):
    """
    Compute SPCE and SNMC bounds for EIG

    :param exp_model: BaseExperiment class storing our probalistic model
    :param true_theta: true parameter value
    :param hist: dictionary containing measurement data and design points
    :param rng_key: random key
    :param inner_samples: number of inner samples for contrastive estimation
    :return: tuple of SPCE and SNMC bounds
    """
    # contrastive samples
    thetas = jax.tree_map(
        lambda value, dis: dis.sample(rng_key, (inner_samples,)),
        exp_model.ground_truth,
        exp_model.params_distrib,
    )

    def cond_logprob(index, theta, y, xi):
        shpes = jax.eval_shape(exp_model.log_prob, theta, y, xi)
        return jax.lax.cond(
            index < n_meas,
            exp_model.log_prob,
            lambda *_: np.zeros_like(shpes),
            theta,
            y,
            xi,
        )

    indices = np.arange(hist["meas"].shape[0])
    # contrastive log likelihoods
    log_liks_c = (
        jax.vmap(cond_logprob, in_axes=(0, None, 0, 0))(
            indices, thetas, hist["meas"], hist["xi"]
        )
        .sum(axis=0)
        .squeeze()
    )
    # extra log likelihood
    log_lik_0 = (
        jax.vmap(cond_logprob, in_axes=(0, None, 0, 0))(
            indices, true_theta, hist["meas"], hist["xi"]
        )
        .sum(axis=0)
        .squeeze()
    )
    # (L + 1) log_liks
    all_liks = np.hstack([log_liks_c, log_lik_0])
    contr_log_lik = jax.scipy.special.logsumexp(all_liks) - np.log(inner_samples + 1)

    spce = log_lik_0 - contr_log_lik

    contr_log_lik = jax.scipy.special.logsumexp(log_liks_c) - np.log(inner_samples)
    snmc = log_lik_0 - contr_log_lik

    return spce, snmc


@partial(jax.jit, static_argnums=(0, 4))
def bounds_eig(
    exp_model: BaseExperiment,
    true_theta: Array,
    hist: PyTreeDef,
    rng_key: PRNGKey,
    inner_samples=int(1e7),
):
    """
    Compute SPCE and SNMC bounds for EIG

    :param exp_model: BaseExperiment class storing our probalistic model
    :param true_theta: true parameter value
    :param hist: dictionary containing measurement data and design points
    :param rng_key: random key
    :param inner_samples: number of inner samples for contrastive estimation
    :return: tuple of SPCE and SNMC bounds
    """
    # contrastive samples
    thetas = jax.tree_map(
        lambda value, dis: dis.sample(rng_key, (inner_samples,)),
        exp_model.ground_truth,
        exp_model.params_distrib,
    )

    # contrastive log likelihoods
    log_liks_c = (
        jax.vmap(exp_model.log_prob, in_axes=(None, 0, 0))(
            thetas, hist["meas"], hist["xi"]
        )
        .sum(axis=0)
        .squeeze()
    )
    # extra log likelihood
    log_lik_0 = (
        jax.vmap(exp_model.log_prob, in_axes=(None, 0, 0))(
            true_theta, hist["meas"], hist["xi"]
        )
        .sum(axis=0)
        .squeeze()
    )

    # (L + 1) log_liks
    all_liks = np.hstack([log_liks_c, log_lik_0])
    contr_log_lik = jax.scipy.special.logsumexp(all_liks) - np.log(inner_samples + 1)

    spce = log_lik_0 - contr_log_lik

    contr_log_lik = jax.scipy.special.logsumexp(log_liks_c) - np.log(inner_samples)
    snmc = log_lik_0 - contr_log_lik

    return spce, snmc


def pce_bound(
    design: Array,
    rng_key: PRNGKey,
    exp_model: BaseExperiment,
    particles: ParticlesApprox,
):
    """
    Sequential version of PCE

    :param design: Pytree storing design
    :param exp_model: BaseExperiment class storing our probalistic model
    :param rng_key: random key
    :param weights: SMC weights of the particles
    :param thetas: Pytree containing samples on which we evaluate the PCE bound of EIG
    """
    thetas, weights = particles
    # thetas = contrastive_sampler(exp_model, rng_key, weights, thetas)
    # sample N y_n
    N_samples = jax.tree_map(lambda leaf: leaf[:, 0], thetas)
    N, Lpp = jax.tree_util.tree_leaves(thetas)[0].shape[:2]
    sample_keys = jax.random.split(rng_key, N)
    sampler = partial(exp_model.sample, xi=design)
    N_y = jax.vmap(sampler, (0, 0))(N_samples, sample_keys)

    # corresponding NxL log p(y_n| theta_l, design)
    log_liks_nm = jax.vmap(exp_model.log_prob, in_axes=(1, None, None), out_axes=1)(
        thetas, N_y, design
    )

    # (Nx1) log probs
    log_liks_n = log_liks_nm[:, 0]

    contr_log_lik = jax.scipy.special.logsumexp(log_liks_nm, axis=1) - np.log(Lpp)

    return np.mean(log_liks_n - contr_log_lik)
    # return np.mean(np.sqrt(np.exp(log_liks_n - contr_log_lik)))


def econom(
    design: Array,
    exp_model: BaseExperiment,
    rng_key: PRNGKey,
    particles: ParticlesApprox,
    N=100,
    L=10,
):
    """
    PCE loss with less variance on samples as contrastive are re-used for every N
    :param design: Pytree storing design
    :param exp_model: BaseExperiment class storing our probalistic model
    :param rng_key: random key
    :param weights: SMC weights of the particles
    :param thetas: Pytree containing samples on which we evaluate the PCE bound of EIG
    """
    thetas, weights = particles
    thetas = jax.tree_util.tree_map(lambda leaf: jax.lax.collapse(leaf, 0, 2), thetas)
    weights = jax.lax.collapse(weights, 0, 2)
    # sample N and L particles
    samples = jax.tree_util.tree_map(
        lambda leaf: jax.random.choice(rng_key, leaf, p=weights, shape=(N + L,)), thetas
    )
    N_samples = jax.tree_util.tree_map(lambda leaf: leaf[:N], samples)
    L_samples = jax.tree_util.tree_map(lambda leaf: leaf[N:], samples)

    # sample N y_n
    sample_keys = jax.random.split(rng_key, N)
    N_y = jax.vmap(exp_model.sample, (0, 0, None))(N_samples, sample_keys, design)

    # corresponding Nx1 log p(y_n| theta_n, design)
    N_log_liks = exp_model.log_prob(N_samples, N_y, design)

    # corresponding NxL log p(y_n| theta_l, design)
    NL_log_liks = jax.vmap(exp_model.log_prob, in_axes=(0, None, None), out_axes=-1)(
        L_samples, N_y, design
    )

    # Nx(L+1)
    all_liks = np.hstack([np.expand_dims(N_log_liks, -1), NL_log_liks])

    # Nx1 log( sum_l p(y_n| theta_l, design))
    contr_log_lik = jax.scipy.special.logsumexp(all_liks, axis=1) - np.log(L + 1)

    return -np.mean(N_log_liks - contr_log_lik)


def contrastive_sampler(rng_key: PRNGKey, particles: ParticlesApprox):
    """
    Sample from product of distributions using SMC
    """
    thetas, weights = particles

    nrv = lambda column: np.exp(np.log(column) - np.log(column.sum()))
    column_normalized_w = jax.vmap(nrv, in_axes=1, out_axes=-1)(weights)
    N, L, *_ = jax.tree_util.tree_leaves(thetas)[0].shape

    sampler = lambda key, vec, w: jax.tree_util.tree_map(lambda leaf: jax.random.choice(key, leaf, p=w, shape=(N,)), vec)
    # def sampler(key, vec, w):
    #     return jax.tree_util.tree_map(
    #         lambda leaf: jax.random.choice(key, leaf, shape=(N,)), vec
    #     )

    sample_keys = jax.random.split(rng_key, N)
    Lsample_keys = jax.random.split(rng_key, L)

    thetas_NL = jax.vmap(sampler, in_axes=(0, 1, 1), out_axes=1)(
        Lsample_keys, thetas, column_normalized_w
    )
    return thetas_NL
