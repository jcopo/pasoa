import jax
import jax.experimental

from jax import numpy as np
from jaxtyping import PRNGKeyArray, PyTree

from exptax.base import ParticlesApprox
from exptax.estimators import bounds_eig_fix_shape
from exptax.models.base import BaseExperiment


def init_part_approx(
    rng_key: PRNGKeyArray, exp_model: BaseExperiment, outer_samples: int, inner_samples: int
) -> ParticlesApprox:
    """
    Initializes a particle approximation for the given experiment model.

    Args:
        rng_key (PRNGKey): The random number generator key.
        exp_model (BaseExperiment): The experiment model.
        outer_samples (int): The number of outer samples.
        inner_samples (int): The number of inner samples.

    Returns:
        ParticlesApprox: The initialized particle approximation made of samples and uniform weights.
    """
    thetas = jax.tree_map(
        lambda value, dis: dis.sample(rng_key, (outer_samples, inner_samples)),
        exp_model.ground_truth,
        exp_model.params_distrib,
    )
    weights = np.ones((outer_samples, inner_samples)) / (outer_samples * inner_samples)
    return ParticlesApprox(thetas, weights)


def create_meas_array(rng_key:PRNGKeyArray, exp_model:BaseExperiment, num_meas:int):
    """
    Instantiate the measurement array for the experiment model with the given number of measurements.
    """
    xi = exp_model.xi(rng_key)
    y = exp_model.measure(rng_key, xi)
    meas_list = jax.tree_map(lambda x: np.zeros((num_meas, *x.shape)), xi)
    value_list = jax.tree_map(lambda x: np.zeros((num_meas, *x.shape)), y)
    return {"xi": meas_list, "meas": value_list}


def update_hist(hist_exp_val:PyTree, xi_star:PyTree, y_measured:PyTree, n_meas:int):
    """
    Updates the history of the experiment values with the new measurement.
    """
    hist_exp_val["meas"] = hist_exp_val["meas"].at[n_meas].set(y_measured)
    hist_exp_val["xi"] = jax.tree_map(
        lambda x, y: x.at[n_meas].set(y), hist_exp_val["xi"], xi_star
    )
    return hist_exp_val


def sequential_design(
    rng_key:PRNGKeyArray,
    opt,
    inference_method,
    exp_model:BaseExperiment,
    num_meas:int,
    outer_samples:int,
    inner_samples:int,
    log_opt=False,
    show_plot=False,
):
    """
    Perform sequential design for an experiment.

    Args:
        rng_key (PRNGKey): The random number generator key.
        opt: The optimization method.
        inference_method: The inference method.
        exp_model (BaseExperiment): The experiment model.
        num_meas (int): The number of measurements.
        outer_samples (int): The number of outer samples.
        inner_samples (int): The number of inner samples.
        log_opt (bool, optional): Whether to log the optimization process. Defaults to False.
        show_plot (bool, optional): Whether to show the plot for interactive visualisation. Defaults to False.

    Returns:
        particles: The particles approx of the posterior after sequential design.
    """

    particles = init_part_approx(rng_key, exp_model, outer_samples, inner_samples)

    # create measurement - experiences array:
    hist_exp_val = create_meas_array(rng_key, exp_model, num_meas)
    opt_state = opt.init(rng_key, particles)
    rng_keys = jax.random.split(rng_key, num_meas)

    # Logging
    def log_scalar(scalar_dict, n_meas):
        for name, value in scalar_dict.items():
            opt.writer.add_scalar(name, float(value), n_meas)

    # Sequential design step
    def step(carry, tup):
        idx, key = tup
        key_run, key_meas, key_noise, key_inf = jax.random.split(key, 4)
        particles, opt_state, hist = carry
        xi_star, logs_opt = opt.run(key_run, opt_state, particles)

        y_measured = exp_model.measure(key_meas, xi_star)
        hist = update_hist(hist, xi_star, y_measured, idx)

        spce_val, snmc_val = bounds_eig_fix_shape(
            exp_model, exp_model.ground_truth, hist, key, idx + 1
        )

        particles, infoerence = inference_method(
            particles, key_inf, hist, idx
        )

        wass_value = exp_model.wasserstein_eval(particles)
        state = opt.init(key_noise, particles, opt_state)
        scalar_dict = {
            "SPCE": spce_val,
            "SNMC": snmc_val,
            "y_meas": y_measured,
            **wass_value,
        }
        if log_opt:
            jax.experimental.io_callback(
                opt.logger, None, xi_star, logs_opt, particles, idx, show_plot
            )

        jax.experimental.io_callback(log_scalar, None, scalar_dict, idx)
        jax.debug.print("\n exp number: {}, {} \n", idx, scalar_dict)

        return (particles, state, hist), None

    end_opt, _ = jax.lax.scan(
        step, (particles, opt_state, hist_exp_val), (np.arange(0, num_meas), rng_keys)
    )
    particles, opt_state, hist = end_opt

    return particles
