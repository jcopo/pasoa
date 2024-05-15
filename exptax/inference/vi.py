from typing import Callable, NamedTuple, Tuple
import logging

import jax
import jax.numpy as np
import optax
from optax import OptState, GradientTransformation
import jax.scipy as jsp

from exptax.inference.base import InferenceAlgorithm
from exptax.models.base import BaseExperiment
from exptax.base import ParticlesApprox


PyTreeDef = jax.tree_util.PyTreeDef
PRNGKey = jax.random.PRNGKeyArray


class ViState(NamedTuple):
    """
    Represents the state of variational inference.

    Attributes:
        parameters (PyTreeDef): The parameters of the model.
        opt_state (OptState): The optimization state.
    """

    parameters: PyTreeDef
    opt_state: OptState


class ProbabilisticModel(NamedTuple):
    """
    ProbabilisticModel: A probabilistic model with the specified
    sample distribution and log probability density function.

    Attributes:
        log_pdf (Callable): The log probability density function of the model.
        sample (Callable): A function that samples from the model's distribution.
    """

    log_pdf: Callable
    sample: Callable


def _sample(rng_key: PRNGKey, params: PyTreeDef, distributions: PyTreeDef):
    sample_keys = {b: rng_key + a for a, b in enumerate(params)}
    samples = jax.tree_util.tree_map(
        lambda key, dist, param: dist(key, *param), sample_keys, distributions, params
    )
    return samples


def _logdensity(positions: PyTreeDef, params: PyTreeDef, log_pdf: PyTreeDef):
    log_densities = jax.tree_util.tree_map(
        lambda pos, log_distr, param: log_distr(pos, *param),
        positions,
        log_pdf,
        params,
    )
    return jax.tree_util.tree_reduce(np.add, log_densities).sum(-1)


class VarationalFamily:
    """
    Represents a variational family for probabilistic modeling.

    Args:
        distribution (PyTreeDef): The distribution used for sampling.
        distrib_log_pdf (PyTreeDef): The distribution's log probability density function.

    Returns:
        ProbabilisticModel: A probabilistic model with the specified sample distribution and log probability density function.
    """

    sample = staticmethod(_sample)
    logdensity = staticmethod(_logdensity)

    def __new__(
        cls, distribution: PyTreeDef, distrib_log_pdf: PyTreeDef
    ) -> ProbabilisticModel:
        def sample(rng_key: PRNGKey, params: PyTreeDef):
            return cls.sample(rng_key, params, distribution)

        def log_pdf(x: PyTreeDef, params: PyTreeDef):
            return cls.logdensity(x, params, distrib_log_pdf)

        return ProbabilisticModel(log_pdf, sample)


def _kl_divergence(
    rng_key,
    params: PyTreeDef,
    log_joint: Callable,
    varational_family: ProbabilisticModel,
    num_samples: int = 10,
):
    # log_pdf, distributions = varational_family
    keys = jax.random.split(rng_key, num_samples)
    samples = jax.vmap(varational_family.sample, in_axes=(0, None))(keys, params)

    log_densities = jax.vmap(varational_family.log_pdf, in_axes=(0, None))(
        samples, params
    )
    log_likelihoods = jax.vmap(log_joint)(samples)

    return -np.mean(log_likelihoods - log_densities)


def _step(
    rng_key,
    state: ViState,
    optimizer: GradientTransformation,
    log_joint: Callable,
    varational_family: VarationalFamily,
    constrains: Callable = lambda x, y: y,
    num_samples: int = 100,
):
    params, opt_state = state
    grad_fn = jax.value_and_grad(_kl_divergence, argnums=1)
    kl_div, grad = grad_fn(rng_key, params, log_joint, varational_family, num_samples)
    updates, opt_state = optimizer.update(grad, opt_state)
    updates = constrains(params, updates)
    updates = jax.tree_util.tree_map(lambda x: np.nan_to_num(x, nan=0.0), updates)
    params = optax.apply_updates(params, updates)

    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        jax.debug.print("updates {}", updates)
        jax.debug.print("params {}", params)
        jax.debug.print("kl_div {}", kl_div)
        jax.debug.print("---------" * 5)
    return ViState(params, opt_state), kl_div


class VI:
    """
    Variational Inference (VI) class.

    Args:
        log_joint (Callable): The log joint probability function.
        varational_family (VarationalFamily): The variational family.
        optimizer (GradientTransformation): The optimizer for updating the parameters.
        constrains (Callable, optional): The constraints function. Defaults to lambda x, y: y.
        num_samples (int, optional): The number of samples. Defaults to 1000.
    """

    step = staticmethod(_step)

    def __new__(
        cls,
        log_joint: Callable,
        varational_family: VarationalFamily,
        optimizer: GradientTransformation,
        constrains: Callable = lambda x, y: y,
        num_samples: int = 1000,
    ):
        def step_fn(rng_key: PRNGKey, state: ViState) -> Tuple[ViState, float]:
            return cls.step(
                rng_key,
                state,
                optimizer,
                log_joint,
                varational_family,
                constrains,
                num_samples,
            )

        def sample_fn(rng_key: PRNGKey, state: ViState) -> PyTreeDef:
            parameters, _ = state
            return varational_family.sample(rng_key, parameters)

        def init_fn(rng_key: PRNGKey, params: Callable) -> ViState:
            params = params(rng_key)
            return ViState(params, optimizer.init(params))

        return InferenceAlgorithm(init_fn, step_fn, sample_fn)


def run_vi(
    particles: ParticlesApprox,
    rng_key: PRNGKey,
    current_hist: PyTreeDef,
    n_meas: int,
    vi_params: Callable,
    varational_family: VarationalFamily,
    exp_model: BaseExperiment,
    optimizer: GradientTransformation,
    constrains: Callable = lambda x, y: y,
    opt_steps: int = 5000,
) -> Tuple[ParticlesApprox, ViState]:
    """
    Runs the Variational Inference (VI) algorithm.
    This updates a particle approximation to fit into the global BOED procedure.

    Args:
        particles (ParticlesApprox): The initial particles approximation.
        rng_key (PRNGKey): The random number generator key.
        xi_star (PyTreeDef): Optimal design defining target distribution.
        current_hist (PyTreeDef): The current history of (xi_k, y_k) for k = 1, .., n_meas.
        n_meas (int): The number of measurements.
        vi_params (PyTreeDef): The parameters for the VI algorithm.
        varational_family (VarationalFamily): The variational family.
        exp_model (BaseExperiment): The experiment model.
        optimizer (GradientTransformation): The optimizer for VI.
        constrains (Callable, optional): The constraints function. Defaults to lambda x, y: y.
        opt_steps (int, optional): The number of optimization steps. Defaults to 5000.

    Returns:
        Tuple[ParticlesApprox, ViState]: The updated particles approximation and VI state.
    """
    outer_samples, inner_samples = particles.weights.shape

    last_data = jax.tree_map(lambda x: x[n_meas], current_hist)
    xi_star, y_measured = last_data["xi"], last_data["meas"]
    log_lik = lambda theta, rng_key: exp_model.log_prob(theta, y_measured, xi_star)
    potential = exp_model.make_potential(current_hist, n_meas)

    def joint(theta):
        return log_lik(theta, rng_key) + potential(theta)

    vi = VI(joint, varational_family, optimizer, constrains)

    vi_state = vi.init(rng_key, vi_params)
    rng_keys = jax.random.split(rng_key, opt_steps)

    def body_fn(state, rng_key):
        return vi.step(rng_key, state)

    end_state, hist_state = jax.lax.scan(body_fn, vi_state, rng_keys)

    num_samples = inner_samples * outer_samples
    sample_keys = jax.random.split(rng_key, num_samples)

    thetas = jax.vmap(vi.sample, in_axes=(0, None))(sample_keys, end_state)
    weights = np.ones((outer_samples, inner_samples)) / num_samples

    thetas = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((outer_samples, inner_samples, *leaf.shape[1:])),
        thetas,
    )
    return ParticlesApprox(thetas, weights), end_state.parameters


if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(0)

    distributions = {
        "theta": jax.random.multivariate_normal,
    }

    log_pdf = {
        "theta": jsp.stats.multivariate_normal.logpdf,
    }

    variational_family = VarationalFamily(distributions, log_pdf)
    num_sources = 2
    d = 1
    vi_params = {
        "theta": (np.zeros((num_sources, d)), np.eye(d, d)),
    }

    space = np.linspace(-3, 3, 100)
    log_on_space = jax.vmap(variational_family.log_pdf, in_axes=(0, None))(
        {"theta": space}, vi_params
    )
    samples_keys = jax.random.split(rng_key, 100)
    samples = jax.vmap(variational_family.sample, in_axes=(0, None))(
        samples_keys, vi_params
    )
    import matplotlib.pyplot as plt

    plt.plot(space, np.exp(log_on_space), label="log pdf")
    plt.scatter(
        samples["theta"], np.zeros_like(samples["theta"]), color="red", label="samples"
    )
    plt.legend()
    plt.show()

    # distributions = {
    #     "rho": lambda rng_key, a, b: jax.random.beta(rng_key, a, b),
    #     "u": lambda rng_key, mu, cov: jax.random.multivariate_normal(rng_key, mu, cov),
    #     "alpha": lambda rng_key, alpha: jax.random.dirichlet(rng_key, alpha),
    # }
    # log_pdf = {
    #     "rho": lambda x, a, b: jsp.stats.beta.logpdf(x, a, b),
    #     "u": lambda x, mu, cov: jsp.stats.multivariate_normal.logpdf(x, mu, cov),
    #     "alpha": lambda x, alpha: jsp.stats.dirichlet.logpdf(x, alpha),
    # }

    # varational_family = VarationalFamily(log_pdf, distributions)

    # vi_params = {
    #     "rho": (np.array([1.0]), np.array([1.0])),
    #     "u": (np.zeros(1), np.eye(1)),
    #     "alpha": (np.ones(3),),
    # }

    # samples = _sample(rng_key, vi_params, distributions)

    # print(samples)

    # log_probs = _logdensity(samples, vi_params, log_pdf)
    # print(log_probs)

    # # plot samples and _logdensities
    # import matplotlib.pyplot as plt
    # space = np.linspace(0, 1, 100)

    # optimizer = optax.adam(1e-3)
    # state = ViState(vi_params, optimizer.init(vi_params))

    # ces = CES()
    # xi = ces.xi(rng_key)
    # y = ces.measure(rng_key, xi)
    # log_likelihood = lambda x: ces.log_prob(x, y, xi)

    # step(rng_key, state, optimizer, log_likelihood, varational_family)

    # # init a dict by comprehension with enumerate on a dict
    # # d = {k: v for k, v in enumerate(["a", "b", "c"])}
    # from exptax.run_utils import create_meas_array, update_hist

    # n_meas = 0
    # hist = create_meas_array(rng_key, ces, 10)
    # hist = update_hist(hist, xi, y, n_meas)

    # def constrains(params, updates):
    #     mean, cov = updates["u"]
    #     _, p_cov = params["u"]
    #     cov = np.where(cov + p_cov < 0.0, -p_cov, cov)
    #     updates["u"] = (mean, cov)

    #     # aplpha, rho >= 0
    #     updates["alpha"] = jax.tree_map(
    #         lambda u, p: np.where(u + p < 0.0, -p, u), updates["alpha"], params["alpha"]
    #     )
    #     updates["rho"] = jax.tree_map(
    #         lambda u, p: np.where(u + p < 0.0, -p, u), updates["rho"], params["rho"]
    #     )

    #     # alpha, rho <= 1
    #     updates["alpha"] = jax.tree_map(
    #         lambda u, p: np.where(u + p > 1.0, 1.0 - p, u),
    #         updates["alpha"],
    #         params["alpha"],
    #     )
    #     updates["rho"] = jax.tree_map(
    #         lambda u, p: np.where(u + p > 1.0, 1.0 - p, u),
    #         updates["rho"],
    #         params["rho"],
    #     )

    #     # nan to num
    #     updates = jax.tree_map(lambda x: np.nan_to_num(x, nan=0.0), updates)

    #     return updates

    # # inference_method(
    # #            thetas, weights, n_meas, hist_exp_val, rng_key, xi_star
    # #        )
    # # MC_CES(thetas, weights, n_meas, current_hist, key, params, exp_model, no_temp)
    # print(
    #     run_vi(
    #         rng_key,
    #         xi,
    #         hist,
    #         n_meas,
    #         varational_family,
    #         ces,
    #         optimizer,
    #         constrains,
    #         830,
    #     )
    # )
    # print(ces.ground_truth)
