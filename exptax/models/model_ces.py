import jax
import jax.numpy as np
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from jax.scipy.special import logit, expit
import diffrax

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from exptax.base import ParticlesApprox
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController
from exptax.hankel import hankel1
from exptax.models.base import BaseExperiment

EPS = 2**-22


class CES(BaseExperiment):
    def __init__(self, rng_key, obs_sd=0.005):
        self.obs_sd = obs_sd
        self.epsilon = EPS
        params_distrib = {
            "alpha": dist.Dirichlet(np.array([1, 1, 1])),
            # "rho":dist.Dirichlet(np.array([1, 1])),
            "rho": dist.Beta(np.array([1]), np.array([1])),
            "u": dist.Normal(np.array([1.0]), np.array([np.sqrt(3)])),
        }

        super().__init__(rng_key, params_distrib)
        self.base_xi_shape = {"x": (1, 3), "xp": (1, 3)}

    def forward(self, xi, theta):
        """
        xi: shape (d) coordinates where to compute intensity
        theta: shape (p, d) position of the sources
        """
        rho = theta["rho"][..., [0]]

        def U(x):
            return np.power(
                np.sum(x**rho * theta["alpha"], -1, keepdims=True), np.exp(-np.log(rho))
            )

        # map u over x, x'
        U_xi = jax.tree_util.tree_map(U, xi)
        mean = np.exp(theta["u"]) * (U_xi["x"] - U_xi["xp"])
        sd = (
            np.exp(theta["u"])
            * self.obs_sd
            * (
                # mean = theta["u"] * (U_xi["x"] - U_xi["xp"])
                # sd = theta["u"] * self.obs_sd * (
                1 + np.linalg.norm(xi["x"] - xi["xp"], axis=-1, ord=2, keepdims=True)
            )
        )

        return np.nan_to_num(mean.squeeze(-1)), np.nan_to_num(sd.squeeze(-1))

    def log_prob(self, theta, y, xi):
        # normal N(f(xi))
        mean, sd = jax.vmap(self.forward, in_axes=(0, None), out_axes=-1)(xi, theta)

        # log_prob
        log_probs = jax.scipy.stats.norm.logpdf(logit(y), mean, sd) - np.log(
            y * (1 - y)
        )
        tol = 2 * np.finfo(y.dtype).tiny

        # compute p0, p1
        def norm_dst(x):
            return (x - mean) / sd

        p0 = jax.scipy.special.ndtr(norm_dst(logit(self.epsilon)))
        p1 = 1 - jax.scipy.special.ndtr(norm_dst(logit(1 - self.epsilon)))

        pond = 1 - p0 - p1
        # set index where not to change
        c0 = p0 >= tol
        c1 = p1 >= tol
        c2 = pond >= tol

        # where p_i < Tol compute approx
        def approx(x):
            return jax.scipy.stats.norm.logpdf(x) - 0.5 * np.log(tol + x**2)

        # this form needed to make where diff with jax
        log_p0 = np.where(
            c0, np.log(np.where(c0, p0, 1.0)), approx(norm_dst(logit(self.epsilon)))
        )
        log_p1 = np.where(
            c1, np.log(np.where(c1, p1, 1.0)), approx(norm_dst(logit(1 - self.epsilon)))
        )

        # on border set corresponding log_prob
        equal_down = y == self.epsilon
        log_probs = np.where(equal_down, np.where(equal_down, log_p0, 1.0), log_probs)
        equal_up = y == (1 - self.epsilon)
        log_probs = np.where(equal_up, np.where(equal_up, log_p1, 1.0), log_probs)

        # log 1 - p0 - p1 or approx if <tol
        lp0p1 = np.where(
            c2,
            np.log(np.where(c2, 1 - p0 - p1, 1.0)),
            0.5
            * np.log(
                (norm_dst(logit(self.epsilon)) - norm_dst(logit(1 - self.epsilon))) ** 2
            )
            + jax.scipy.stats.norm.logpdf(norm_dst(logit(self.epsilon))),
        )
        # jax.debug.breakpoint()
        # outside set log_prb = -inf
        cond = (y < self.epsilon) | (y > 1 - self.epsilon)
        log_probs = np.where(cond, float("-inf"), log_probs)  # + lp0p1

        return log_probs.squeeze(-1)

    def sample(self, args, rng_key, xi):
        # output shape is args.shape + xi.shape
        mean, sd = jax.vmap(self.forward, in_axes=(0, None), out_axes=-1)(xi, args)
        dst = dist.Normal(np.zeros_like(mean), np.ones_like(sd)).to_event(1)
        eta = mean + sd * dst.sample(rng_key)
        y = expit(eta)

        # censore parts outside of epsilon, 1 - epsilon
        y = np.where(y < self.epsilon, self.epsilon, y)
        y = np.where(y > 1 - self.epsilon, 1 - self.epsilon, y)
        return y

    def wasserstein_eval(self, particles:ParticlesApprox):
        thetas, weights = particles
        ground_truth = self.ground_truth.copy()

        thetas = jax.tree_util.tree_map(lambda leaf: jax.lax.collapse(leaf, 0, 2), thetas)
        weights = jax.lax.collapse(weights, 0, 2)
        thetas["rho"] = thetas["rho"][:, [0]]
        ground_truth["rho"] = ground_truth["rho"][np.array([0])]
        x = np.hstack(jax.tree_util.tree_leaves(thetas))
        y = np.hstack(jax.tree_util.tree_leaves(ground_truth))
        y = np.expand_dims(y, 0)

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


    def plot_inference(
        self, fig, ax, particles, params, n_meas, weights=None, title=None
    ):
        """
        Utility function to plot particles and their IS weights

        :param thetas: Array of shape (N, M, n_sources, d) of particles
        :param param: Pytree of the design
        :param weights: Array of corresponding IS weights. Expected shape (N*M,)
        :param writer: SummaryWriter for logging to TensorBoard
        """
        particles = jax.tree_util.tree_map(
            lambda leaf: jax.lax.collapse(leaf, 0, 2).squeeze(), particles
        )
        thetas, weights = particles
        ax = fig.add_subplot(211, sharex=ax, frameon=False)

        # Creating the plot
        # fig, ax = plt.subplots()

        # Customizing the plot
        ax.set_xlabel("Variables")
        ax.set_ylabel("rho Values")

        # Adding a reference point to the first y-axis
        ax.plot(
            1.5, self.ground_truth["rho"][0], "ro", label="Reference Point"
        )  # Plotting a red circle at (0, reference_value)
        ax.legend(
            ["$\\rho$ Values"],
            loc="lower left",
            bbox_to_anchor=(0.05, 0.05),
            frameon=True,
            framealpha=0.7,
            fontsize="small",
        )

        # Plotting the first box plot
        positions = [1.5]  # X-coordinate for the first box plot
        ax.violinplot([item for item in thetas["rho"].tolist()], positions)

        # Creating a twin y-axis
        ax2 = ax.twinx()
        # Adding a reference point to the first y-axis
        ax2.plot(
            2.5, self.ground_truth["u"], "gx", label="Reference Point"
        )  # Plotting a red circle at (0, reference_value)
        ax2.legend(
            ["u Values"],
            loc="lower right",
            bbox_to_anchor=(0.98, 0.05),
            frameon=True,
            framealpha=0.7,
            fontsize="small",
        )
        # Customizing the twin y-axis
        ax2.set_ylabel("u Values")

        # Plotting the second box plot
        positions = [2.5]  # X-coordinate for the second box plot
        ax2.violinplot(thetas["u"].tolist(), positions=positions)

        if not title:
            title = "Particle approximation"
        plt.title(f"{title} {n_meas}")

        # self.writer.add_figure(title, fig, n_meas)
        # fig2, ax2 = plt.subplots()
        ax3 = fig.add_subplot(212, sharex=ax, frameon=False)
        ax3

        # Customizing the plot
        ax3.set_xlabel("Coordinate")
        ax3.set_ylabel("$\\alpha$ Values")

        # Adding a reference point to the first y-axis
        ax3.plot(
            1, self.ground_truth["alpha"][0], "ro", label="Reference Point"
        )  # Plotting a red circle at (0, reference_value)
        ax3.plot(
            2, self.ground_truth["alpha"][1], "ro", label="Reference Point"
        )  # Plotting a red circle at (0, reference_value)
        ax3.plot(
            3, self.ground_truth["alpha"][2], "ro", label="Reference Point"
        )  # Plotting a red circle at (0, reference_value)
        ax3.legend(
            ["$\\alpha$ Values"],
            loc="lower left",
            bbox_to_anchor=(0.05, 0.05),
            fontsize="small",
        )
        positions = [[1], [2], [3]]  # X-coordinate for the first box plot

        ax3.violinplot([item[0] for item in thetas["alpha"].tolist()], positions=[1])
        ax3.violinplot([item[1] for item in thetas["alpha"].tolist()], positions=[2])
        ax3.violinplot([item[2] for item in thetas["alpha"].tolist()], positions=[3])
        plt.title(f"{title} {n_meas}")
        # self.writer.add_figure(title + " alpha", fig, n_meas)
        # self.writer.close()
        # plt.show()
        # plt.close()
        # Displaying the plot

    def xi(self, rng_key):
        u = jax.random.uniform(rng_key, minval=0, maxval=10)
        return {
            "x": dist.TruncatedNormal(u, 5.0, low=0.1, high=100).sample(
                rng_key, sample_shape=(1, 3)
            ),
            "xp": dist.TruncatedNormal(u, 5.0, low=0.1, high=100).sample(
                rng_key + 1, sample_shape=(1, 3)
            ),
        }

    def xi_part(self, particles, rng_key):
        return self.xi(rng_key)
        # return {"x": jax.random.uniform(rng_key, (1, 3), maxval=100),
        #        "xp": jax.random.uniform(rng_key+1, (1, 3), maxval=100)}
