import argparse
import logging
import datetime
from functools import partial

import jax
import jax.numpy as np
import jax.scipy as jsp
import optax
import blackjax

from torch.utils.tensorboard import SummaryWriter

from exptax.models.model_ces import CES
from exptax.estimators import reinforce_pce
from exptax.optimizers.sgd import SGD
from exptax.optimizers.parallel_tempering import ParallelTempering
from exptax.run_utils import (
    sequential_design,
)
from exptax.inference.vi import VarationalFamily, run_vi
from exptax.inference.CES_smc import SMC_CES


def make_SGD(exp_model, writer, opt_steps, energy):
    # SGD
    exponential_decay_scheduler = optax.exponential_decay(
        init_value=1e-1,
        transition_steps=opt_steps,
        decay_rate=0.98,
        transition_begin=int(opt_steps * 0.25),
        staircase=False,
    )

    opt = SGD(
        exp_model,
        writer,
        opt_steps,
        {"learning_rate": exponential_decay_scheduler},
        optax.adam,
        energy,
    )

    return opt


def make_PT(exp_model, writer, opt_steps, energy):
    # PT
    temps = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.008])
    # temps = np.array([0.001, 0.01, 0.03, 0.05, .09])
    # temps = np.array([0.01, 0.3, 0.5, 70., 80., 100., 1000.])
    step_size = 1e1 * (temps) ** (1 / 4)

    opt = ParallelTempering(
        exp_model,
        writer,
        temps,
        # blackjax.additive_step_random_walk.normal_random_walk,
        # {"sigma": step_size},
        blackjax.mala,
        {"step_size": step_size},
        # hmc_parameters,
        opt_steps,
        energy,
    )

    return opt



def make_vi(exp_model, opt_steps):
    def constrains(params, updates):
        mean, cov = updates["u"]
        _, p_cov = params["u"]
        cov = np.where(cov + p_cov < 0.0, -p_cov, cov)
        updates["u"] = (mean, cov)

        # alpha, rho >= 0
        updates["alpha"] = jax.tree_map(
            lambda u, p: np.where(u + p < 0.0, -p, u), updates["alpha"], params["alpha"]
        )
        updates["rho"] = jax.tree_map(
            lambda u, p: np.where(u + p < 0.0, -p, u), updates["rho"], params["rho"]
        )

        # alpha, rho <= 1
        updates["alpha"] = jax.tree_map(
            lambda u, p: np.where(u + p > 1.0, 1.0 - p, u),
            updates["alpha"],
            params["alpha"],
        )
        updates["rho"] = jax.tree_map(
            lambda u, p: np.where(u + p > 1.0, 1.0 - p, u),
            updates["rho"],
            params["rho"],
        )

        # nan to num
        updates = jax.tree_map(lambda x: np.nan_to_num(x, nan=0.0), updates)

        return updates

    vi_params = lambda rng_key: {
        "rho": (np.array([1.0]), np.array([1.0])),
        "u": (np.zeros(1), np.eye(1)),
        "alpha": (np.ones(3),),
    }

    distributions = {
        "rho": jax.random.beta,
        "u": jax.random.multivariate_normal,
        "alpha": jax.random.dirichlet,
    }
    log_pdf = {
        "rho": jsp.stats.beta.logpdf,
        "u": jsp.stats.multivariate_normal.logpdf,
        "alpha": jsp.stats.dirichlet.logpdf,
    }
    varational_family = VarationalFamily(distributions, log_pdf)

    optimizer = optax.adam(1e-3)

    return jax.jit(
        partial(
            run_vi,
            vi_params=vi_params,
            varational_family=varational_family,
            exp_model=exp_model,
            optimizer=optimizer,
            constrains=constrains,
            opt_steps=opt_steps,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMC experiment design")
    parser.add_argument("--inner_samples", default=100, type=int)
    parser.add_argument("--outer_samples", default=100, type=int)
    parser.add_argument("--type_loss", default="PCE", type=str)
    parser.add_argument("--opt_type", default="SGD", type=str)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--iter_per_meas", default=1000, type=int)
    parser.add_argument("--num_meas", default=11, type=int)
    parser.add_argument("--plot_meas", action=argparse.BooleanOptionalAction)
    parser.add_argument("--no_temp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--plot_post", action=argparse.BooleanOptionalAction)
    parser.add_argument("--log_SGD", action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_hist", action=argparse.BooleanOptionalAction)
    parser.add_argument("--mini_batch", default=None, type=int)
    parser.add_argument("--rng_key", default=1, type=int)
    parser.add_argument(
        "--logging",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.logging.upper())
    logging.info("Logging now setup.")

    dir_name = "runs/" + args.prefix + "ces/" + args.name + "/"
    tensorboard_name = (
        dir_name
        + args.name
        + datetime.datetime.now().strftime("%S_%H:%M_%d_%m")
        + f"_{args.rng_key}_inner_{args.inner_samples}_outer_{args.outer_samples}"
    )
    print(dir_name)
    writer = SummaryWriter(tensorboard_name)
    writer.add_text("Params", str(args)[10:-1])

    rng_key = jax.random.PRNGKey(args.rng_key)
    exp_model = CES(rng_key)
    inference_method = jax.jit(partial(SMC_CES, exp_model=exp_model, no_temp=args.no_temp))
    # inference_method = make_vi(exp_model, 2500)
    loss = reinforce_pce
    # loss = pce_bound

    @jax.jit
    def energy(particles, positions, rng_key):
        return loss(positions, rng_key, exp_model, particles)

    opt = make_SGD(
        exp_model,
        writer,
        args.iter_per_meas,
        energy,
    )
    opt = make_PT(exp_model, writer, args.iter_per_meas, energy)

    logging.info(exp_model.ground_truth)

    sequential_design(
        rng_key,
        opt,
        inference_method,
        exp_model,
        args.num_meas,
        args.outer_samples,
        args.inner_samples,
        log_opt=args.plot_post,
        show_plot=True,
    )
