import argparse
import logging
import datetime
from functools import partial
from contextlib import nullcontext

import jax
import jax.numpy as np
import jax.scipy as jsp
import optax
import blackjax

from torch.utils.tensorboard import SummaryWriter

from exptax.models.model_sources import Sources
from exptax.estimators import pce_bound
from exptax.run_utils import (
    sequential_design,
)
from exptax.inference.SMC import SMC
from exptax.optimizers.sgd import SGD
from exptax.optimizers.parallel_tempering import ParallelTempering
from exptax.inference.vi import VarationalFamily, run_vi

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

print(jax.devices())



def make_SGD(exp_model, writer, opt_steps, energy):
    # SGD
    exponential_decay_scheduler = optax.exponential_decay(
        init_value=1e-2,
        transition_steps=opt_steps,
        decay_rate=0.95,
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



def make_vi(rng_key, exp_model, opt_steps):
    def constrains(params, updates):
        mean, cov = updates["theta"]
        _, p_cov = params["theta"]
        cov = np.where(cov + p_cov <= 0, -p_cov + 1e-6, cov)
        updates["theta"] = (mean, cov)

        return updates

    num_sources = exp_model.num_sources
    d = exp_model.d

    vi_params = lambda rng_key: {
        "theta": (4 * jax.random.normal(rng_key, (num_sources, d)), np.eye(d, d)),
    }

    # need to regularize cov to avoid PSD probs
    distributions = {
        "theta": lambda rng_key, mu, cov: jax.random.multivariate_normal(
            rng_key, mu, cov + 1e-10 * np.eye(d, d)
        ),
    }
    log_pdf = {"theta": jsp.stats.multivariate_normal.logpdf}
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
    parser.add_argument("--num_sources", default=2, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--iter_per_meas", default=1000, type=int)
    parser.add_argument("--num_meas", default=30, type=int)
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

    dir_name = "runs/" + args.prefix + "sources/" + args.name + "/"
    tensorboard_name = (
        dir_name
        + datetime.datetime.now().strftime("%H_%M_%S_%d_%m")
        + f"_{args.rng_key}_inner_{args.inner_samples}_outer_{args.outer_samples}"
    )
    print(dir_name)
    writer = SummaryWriter(tensorboard_name)
    writer.add_text("Params", str(args)[10:-1])

    rng_key = jax.random.PRNGKey(args.rng_key)
    inference_method = SMC

    exp_model = Sources(
        max_signal=1e-4,
        base_signal=0.1,
        num_sources=args.num_sources,
        source_var=10,
        d=2,
        rng_key=rng_key,
        noise_var=0.5,
    )
    loss = pce_bound

    @jax.jit
    def energy(particles, positions, rng_key):
        return loss(positions, rng_key, exp_model, particles)

    opt = make_SGD(exp_model, writer, args.iter_per_meas, energy)
    T = 4

    logging.info(exp_model.ground_truth)

    inference_method = jax.jit(partial(SMC, exp_model=exp_model, no_temp=args.no_temp))
    # inference_method = make_vi(rng_key, exp_model, 2500)

    with jax.profiler.trace(dir_name) if args.profile else nullcontext():
        particles = sequential_design(
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
