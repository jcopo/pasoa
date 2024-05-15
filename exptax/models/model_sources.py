import jax
import jax.numpy as np
import numpyro.distributions as dist

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from exptax.base import ParticlesApprox
from exptax.models.base import BaseExperiment


class Sources(BaseExperiment):
    def __init__(
        self,
        rng_key,
        max_signal=1e-4,
        base_signal=1e-1,
        num_sources=1,
        d=2,
        source_var=1,
        noise_var=0.5,
    ):
        self.max_signal = max_signal
        self.base_signal = base_signal
        self.num_sources = num_sources
        self.source_var = source_var
        self.noise_var = noise_var
        self.d = d

        self.dist_sources = dist.MultivariateNormal(
            np.zeros((num_sources, d)), self.source_var * np.eye(d, d)
        ).to_event(1)
        self.base_xi_shape = {"pos": (1, self.d)}
        super().__init__(rng_key, {"theta": self.dist_sources})

    def forward(self, xi, theta):
        """
        xi: shape (d) coordinates where to compute intensity
        theta: shape (p, d) position of the sources

        output shape with broadcasting theta.shape[:-2]
        """
        inv_value = self.max_signal + np.power(theta["theta"] - xi["pos"], 2).sum(-1)
        arr = self.base_signal + np.power(inv_value, -1).sum(-1)
        return arr

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
        num_sources = thetas["theta"].shape[2]
        x = jax.lax.collapse(thetas["theta"], 0, 3)
        y = self.ground_truth["theta"]
        weights = np.dstack([weights] * num_sources)
        geom = pointcloud.PointCloud(x, y)

        # Define a linear problem with that cost structure.
        ot_prob = linear_problem.LinearProblem(geom)
        # Create a Sinkhorn solver
        solver = sinkhorn.Sinkhorn()
        # Solve OT problem
        ot = solver(ot_prob)
        ot_cost = np.sum(ot.matrix * ot.geom.cost_matrix)

        # pondered = diff_squared * weights
        return {"theta": ot_cost}


    def plot_opt(self, fig, ax, positions, energies):
        positions = positions["pos"]
        alpha = np.linspace(0.1, 1, positions.shape[0])
        if self.d == 1:
            positions = jax.lax.collapse(positions, 1)
            ax.scatter(positions[:, 0], energies, c=alpha, marker="x", zorder=1)
            ax.scatter(positions[:, 1], energies, c=alpha, marker="x", zorder=1)

        elif self.d == 2:
            ax.scatter(
                positions[:, 0, 0],
                positions[:, 0, 1],
                c=energies,
                marker="x",
                zorder=0,
                alpha=0.4,
            )

    def plot_energy(self, fig, axs, particles, energy):
        rng_key = jax.random.PRNGKey(0)
        if self.d == 1:
            xmax = 20
            space = np.linspace(-xmax, xmax, 4000)
            space = np.expand_dims(space, axis=-1)
            space = {"pos": space}
            plotter_loss = lambda positions, key: energy(particles, positions, key)
            keys = jax.random.split(rng_key, space["pos"].size)
            energie_domain = jax.vmap(plotter_loss)(space, keys)

            for ax in axs:
                ax.plot(space["pos"], energie_domain)

    def plot_inference(
        self, fig, ax, particles, params, n_meas, writer=None, title=None
    ):
        """
        Utility function to plot particles and their IS weights

        :param thetas: Array of shape (N, M, n_sources, d) of particles
        :param param: Pytree of the design
        :param weights: Array of corresponding IS weights. Expected shape (N*M,)
        :param writer: SummaryWriter for logging to TensorBoard
        """
        thetas, weights = particles
        flat_thetas = thetas["theta"].reshape((-1, *thetas["theta"].shape[2:]))
        lim = 6.0
        if self.d == 1:
            for source in range(self.num_sources):
                zeros = np.zeros_like(flat_thetas[:, source])
                ax.scatter(flat_thetas[:, source], y=zeros, s=1, zorder=-1, c=weights)
                ax.scatter( x=self.ground_truth["theta"][source], y=0, marker="x", zorder=1, c="r", linewidths=1,)

                ax.scatter(
                    x=params["pos"][0], y=0, c="m", marker="+", zorder=1, linewidths=1
                )
                # ax.xlabel(r"$x$")

        elif self.d == 2:
            for source in range(self.num_sources):
                ax.scatter( flat_thetas[:, source, 0], y=flat_thetas[:, source, 1], s=1, zorder=-1, c=weights,)
                ax.scatter( x=self.ground_truth["theta"][source, 0], y=self.ground_truth["theta"][source, 1], marker="x", zorder=1, c="r", linewidths=1,)

            x = (params["pos"][:, 0],)
            y = (params["pos"][:, 1],)
            ax.scatter( x=x, y=y, c="orange", marker="+", zorder=1, linewidths=1, s=500,)

            # plt.show()
            # clb = ax.colorbar()
            # clb.ax.set_ylabel("Importance Sampling weights", rotation=270, labelpad=14.5)
            # ax.xlim([-lim, lim])
            # ax.ylim([-lim, lim])
            # ax.xlabel(r"$x$")
            # ax.ylabel(r"$y$")

        if not title:
            title = "Particle approximation"
        # plt.title(f"{title} {n_meas}")
        # plt.axis('off')
        # plt.savefig(f'runs/sources/TSMC/fig_{n_meas}.png', bbox_inches='tight',pad_inches=0, dpi=400)
        # writer.add_figure(title, fig, n_meas, close=False)

    def xi(self, rng_key):
        return {"pos": jax.random.normal(rng_key, (1, self.d))}

    def xi_part(self, particles, rng_key):
        thetas, weights = particles
        collapsed = jax.lax.collapse(thetas["theta"], 0, 3)
        rstart = jax.random.normal(rng_key) + jax.random.choice(
            rng_key, collapsed, shape=(1,)
        )
        return {"pos": rstart}
