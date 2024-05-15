import jax
import jax.numpy as np
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from exptax.hankel import hankel1

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from exptax.base import ParticlesApprox

from exptax.models.base import BaseExperiment


# solution of Helm equation with radiating Sommerfeld conditions
def phi_x(x, z, k):
    return 0.25 * 1j * hankel1(0, k * np.linalg.norm(x - z, ord=1).astype(np.complex64))


class PremiumSources(BaseExperiment):
    def __init__(self, rng_key, num_sources=2.0, source_var=1.0, noise_var=1.0):
        self.num_sources = num_sources
        self.noise_var = noise_var

        params_distrib = {
            "pos": dist.MultivariateNormal(
                np.zeros((num_sources, 2)), source_var * np.eye(2, 2)
            ),
            "eta": dist.MultivariateNormal(
                np.zeros((num_sources, 2)), source_var * np.eye(2, 2)
            ),
            # "lambda": dist.MultivariateNormal(np.zeros((num_sources, 1)),
            #                            source_var*np.eye(1, 1)),
            "lambda": dist.Normal(
                np.zeros((num_sources, 1)), source_var * np.eye(1, 1)
            ).to_event(1),
        }
        super().__init__(rng_key, params_distrib)

    def forward(self, xi, theta):
        """
        xi: shape (d) coordinates where to compute intensity
        theta: shape (p, d) position of the sources

        output shape with broadcasting theta.shape[:-2]
        """
        xi = jax.tree_map(lambda arr: arr.astype(np.complex64), xi)
        theta = jax.tree_map(lambda arr: arr.astype(np.complex64), theta)

        # closed form of radiating fields
        # 4.1 of https://arxiv.org/pdf/1801.05584.pdf
        def u(x, k, z, lmbda, eta):
            value, grad = jax.value_and_grad(phi_x, holomorphic=True)(x, z, k)
            return np.abs(-np.multiply(value, lmbda) + np.dot(eta, grad))

        return np.vectorize(u, signature="(i),(),(i),(k),(i)->(k)")(
            xi["pos"], xi["k"], theta["pos"], theta["lambda"], theta["eta"]
        ).sum((-1, -2))

    def log_prob(self, thetas, y, xi):
        # scalar output
        f_values = jax.vmap(self.forward, in_axes=(0, None), out_axes=-1)(xi, thetas)
        return (
            dist.Normal(0, self.noise_var)
            .log_prob(np.log(y) - np.log(f_values))
            .sum(-1)
        )

    def sample(self, args, rng_key, xi):
        # output shape is args.shape + xi.shape[0]
        f_values = jax.vmap(self.forward, in_axes=(0, None), out_axes=-1)(xi, args)
        return np.exp(
            np.log(f_values)
            + dist.Normal(0, self.noise_var).sample(
                rng_key, sample_shape=f_values.shape
            )
        )

    def wasserstein_eval(self, particles:ParticlesApprox):
        thetas, weights = particles
        ground_truth = self.ground_truth.copy()

        thetas = jax.tree_util.tree_map(lambda leaf: jax.lax.collapse(leaf, 0, 2), thetas)
        thetas = jax.tree_util.tree_map(lambda leaf: jax.lax.collapse(leaf, 1, 3), thetas)
        ground_truth = jax.tree_util.tree_map(
            lambda leaf: jax.lax.collapse(leaf, 0, 2), ground_truth
        )
        weights = jax.lax.collapse(weights, 0, 2)

        def wass(y, x):
            y = np.expand_dims(y, 0)
            geom = pointcloud.PointCloud(x, y)

            # Define a linear problem with that cost structure.
            ot_prob = linear_problem.LinearProblem(geom, a=weights)
            # Create a Sinkhorn solver
            solver = sinkhorn.Sinkhorn()
            # Solve OT problem
            ot = solver(ot_prob)
            ot_cost = np.sum(ot.matrix * ot.geom.cost_matrix)

            # pondered = diff_squared * weights
            return ot_cost

        return jax.tree_util.tree_map(wass, ground_truth, thetas)

    def xi(self, rng_key):
        return {
            "pos": jax.random.uniform(rng_key, (1, 2)),
            "k": np.exp(jax.random.uniform(rng_key, (1,))),
        }

    def plot_particles(self, thetas, params, n_meas, writer, weights, title=None):
        """
        Utility function to plot particles and their IS weights

        :param thetas: Array of shape (N, M, n_sources, d) of particles
        :param param: Pytree of the design
        :param weights: Array of corresponding IS weights. Expected shape (N*M,)
        :param writer: SummaryWriter for logging to TensorBoard
        """
        flat_thetas = thetas["pos"].reshape((-1, *thetas["pos"].shape[2:]))
        lim = 6
        fig = plt.figure()
        for source in range(self.num_sources):
            plt.scatter(
                flat_thetas[:, source, 0],
                y=flat_thetas[:, source, 1],
                s=1,
                zorder=-1,
                c=weights,
            )
            plt.scatter(
                x=self.ground_truth["pos"][source, 0],
                y=self.ground_truth["pos"][source, 1],
                marker="x",
                zorder=1,
                c="r",
            )

        plt.scatter(
            x=params["pos"][0, 0], y=params["pos"][0, 1], c="g", marker="x", zorder=1
        )
        clb = plt.colorbar()
        clb.ax.set_ylabel("Importance Sampling weights", rotation=270, labelpad=14.5)
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        if not title:
            title = "Particle approximation"
        plt.title(f"{title} {n_meas}")

        writer.add_figure(title, fig, n_meas)
        writer.close()
