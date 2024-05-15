import jax
import jax.numpy as np
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController
from exptax.hankel import hankel1

from exptax.models.base import BaseExperiment


class PreyModel(BaseExperiment):
    def __init__(self, rng_key):
        # params_distrib ={"a":dist.Normal(-1.4, 1.35),
        #                "th":dist.Normal(-1.4, 1.35)}
        params_distrib = {
            "a": dist.Normal(np.array([-1.4]), np.array([1.35])).to_event(-1),
            "th": dist.Normal(np.array([-1.4]), np.array([1.35])).to_event(-1),
        }
        super().__init__(rng_key, params_distrib)

    def forward(self, xi, args):
        """
        Solves ODE dy =  a*y / ( 1 + a * th *y) with IC xi["y0"]
        :param xi: Design variable, here iInitial Condition of ODE
        :param args: Pytree of model parameters we want to infer
        """

        def vector_field(t, y, args):
            an2 = np.exp(args["a"]) * y * y
            return -an2 / (1 + np.exp(args["th"]) * an2)

        term = ODETerm(vector_field)
        # solver = Dopri5()
        # solver = Bosh3()
        solver = diffrax.Heun()
        saveat = SaveAt(ts=[24.0])
        # saveat = SaveAt(ts=[i/10 for i in range(31)])
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

        def f(a, th):
            return (
                diffeqsolve(
                    term,
                    solver,
                    t0=0,
                    t1=24,
                    dt0=1.0,
                    y0=xi["y0"],
                    saveat=saveat,
                    args={"a": a, "th": th},
                )
                .ys[0]
                .squeeze()
            )

        sol = np.vectorize(f)(args["a"], args["th"]).squeeze(-1)
        # sol = np.vectorize(f, signature='(i),(i)->()')(args["a"], args["th"])

        return sol

    def plot_particles(self, thetas, params, n_meas, writer, weights, title=None):
        """
        Utility function to plot particles and their IS weights

        :param thetas: Array of shape (N, M, n_sources, d) of particles
        :param param: Pytree of the design
        :param weights: Array of corresponding IS weights. Expected shape (N*M,)
        :param writer: SummaryWriter for logging to TensorBoard
        """
        s = 20
        title = "Particle approximation"
        plt.title(f"{title} {n_meas}")
        thetas = jax.tree_util.tree_map(
            lambda leaf: jax.lax.collapse(leaf, 0, 2).squeeze(), thetas
        )
        weights = jax.lax.collapse(weights, 0, 2)
        fig, ax1 = plt.subplots()
        ax1.scatter(np.ones_like(thetas["a"]), thetas["a"], c=weights, s=s)
        ax1.scatter(
            np.ones_like(self.ground_truth["a"]),
            self.ground_truth["a"],
            c="r",
            marker="x",
            label="ground truth",
        )
        ax1.set_xlim(0.5, 2.5)
        ax1.set_ylim(-3, 3)
        ax1.get_xaxis().set_visible(False)

        ax2 = ax1.twinx()
        # ax2.set_ylim(-2, 2)
        ax2.scatter(2 * np.ones_like(thetas["th"]), thetas["th"], c=weights, s=s)
        ax2.scatter(
            2 * np.ones_like(self.ground_truth["th"]),
            self.ground_truth["th"],
            c="r",
            marker="x",
            label="ground truth",
        )

        plt.title(f"{title} {n_meas}")
        writer.add_figure(title, fig, n_meas)
        writer.close()
        # fig.colorbar()
        # plt.show()

    def sample(self, args, rng_key, xi):
        y_t = jax.vmap(self.forward, in_axes=(0, None), out_axes=-1)(xi, args)
        p_t = (xi["y0"] - y_t) / xi["y0"]
        y = dist.Binomial(xi["y0"].astype(int), p_t).sample(rng_key)
        return y

    def log_prob(self, theta, y, xi):
        y_t = jax.vmap(self.forward, in_axes=(0, None), out_axes=-1)(xi, theta)
        p_t = (xi["y0"] - y_t) / xi["y0"]
        return (
            dist.Binomial(xi["y0"].astype(int), p_t).log_prob(y.astype(int)).squeeze(-1)
        )

    def xi(self, rng_key):
        return {"y0": np.array([233.0])}


# solution of Helm equation with radiating Sommerfeld conditions
def phi_x(x, z, k):
    return 0.25 * 1j * hankel1(0, k * np.linalg.norm(x - z, ord=1).astype(np.complex64))
