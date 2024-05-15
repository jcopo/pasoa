import jax
import pytest
from exptax.models import exp_models as models
import jax.numpy as np
import chex


@pytest.mark.parametrize("theta_shape", [(10,), (10, 10), (4, 9)])
@pytest.mark.parametrize(
    "model",
    [
        models.Sources(d=2, num_sources=2),
        models.Sources(d=2, num_sources=1),
        models.CES(),
        models.PreyModel(),
    ],
)
def test_shape_forward(theta_shape, model):
    rng_key = jax.random.PRNGKey(2356)
    thetas = jax.tree_map(
        lambda value, dis: dis.sample(rng_key, theta_shape),
        model.ground_truth,
        model.params_distrib,
    )
    xi = model.xi(rng_key)
    forwad_output = model.forward(xi, thetas)
    chex.assert_shape(forwad_output, theta_shape)


@pytest.mark.parametrize("theta_shape", [(10,), (10, 10), (4, 9)])
@pytest.mark.parametrize(
    "model",
    [
        models.Sources(d=2, num_sources=2),
        models.Sources(d=2, num_sources=1),
        models.PremiumSources(num_sources=3),
        models.PremiumSources(num_sources=1),
        models.PremiumSources(num_sources=2),
        models.CES(),
        models.PreyModel(),
    ],
)
def test_shape_sample(theta_shape, model):
    rng_key = jax.random.PRNGKey(2356)
    thetas = jax.tree_map(
        lambda value, dis: dis.sample(rng_key, theta_shape),
        model.ground_truth,
        model.params_distrib,
    )
    xi = model.xi(rng_key)
    samples = model.sample(thetas, rng_key, xi)
    chex.assert_shape(samples, (*theta_shape, 1))


@pytest.mark.parametrize("theta_shape", [(10,), (10, 10), (4, 9)])
@pytest.mark.parametrize(
    "model",
    [
        models.Sources(d=2, num_sources=2),
        models.Sources(d=2, num_sources=1),
        models.PremiumSources(num_sources=2),
        models.PremiumSources(num_sources=1),
        models.CES(),
        models.PreyModel(),
    ],
)
def test_shape_logprob(theta_shape, model):
    rng_key = jax.random.PRNGKey(2356)
    thetas = jax.tree_map(
        lambda value, dis: dis.sample(rng_key, theta_shape),
        model.ground_truth,
        model.params_distrib,
    )
    xi = model.xi(rng_key)
    log_prob = model.log_prob(thetas, model.measure(rng_key, xi), xi)
    chex.assert_shape(log_prob, theta_shape)


@pytest.mark.parametrize("theta_shape", [(10,), (10, 10), (4, 9)])
@pytest.mark.parametrize(
    "model", [models.Sources(d=2, num_sources=2), models.Sources(d=2, num_sources=1)]
)
def test_forward_sources(theta_shape, model):
    """
    test whether forwad close to a source is correct
    """
    rng_key = jax.random.PRNGKey(2356)
    xi = {"pos": model.ground_truth["theta"]}
    forwad_output = model.forward(xi, model.ground_truth)
    max_intensity = model.base_signal + sum(
        1 / model.max_signal for _ in range(model.num_sources)
    )

    chex.assert_equal(forwad_output, max_intensity)


@pytest.mark.parametrize("theta_shape", [(10,), (10, 10), (4, 9)])
@pytest.mark.parametrize(
    "model", [models.Sources(d=2, num_sources=2), models.Sources(d=2, num_sources=1)]
)
def test_intensity_sources(theta_shape, model):
    """
    test whether measurementclose to a source is correct (noise not in 0.99 qtile of N(0, 1)
    """
    rng_key = jax.random.PRNGKey(2356)
    xi = {"pos": model.ground_truth["theta"]}
    forwad_output = model.forward(xi, model.ground_truth)
    max_intensity = model.base_signal + sum(
        1 / model.max_signal for _ in range(model.num_sources)
    )

    exp_output = model.measure(rng_key, xi)
    chex.assert_scalar_negative(
        (np.abs(np.log(exp_output)[0] - np.log(max_intensity)) - 2.326348).item()
    )
