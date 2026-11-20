"""验证多期限原始指标、评分门槛与稳定排名。"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from factor_engine.expression import make_expr, terminal
from factor_engine.gp import Candidate
from factor_engine.metrics import MetricStats
from factor_engine.metric_types import METRIC_COUNT, Metric
from factor_engine.fitness import (
    multitarget_fitness,
    rank_multitarget_candidates,
)
from factor_engine.fitness import fitness_and_order


def make_stats(rank_ic, horizons=None):
    rank_ic = np.asarray(rank_ic, dtype=np.float64)
    roots, horizon_count = rank_ic.shape
    raw = np.zeros((roots, horizon_count, METRIC_COUNT, 2), dtype=np.float64)
    raw[..., 1] = 1
    raw[..., Metric.RANK_IC, 0] = np.nan_to_num(rank_ic)
    raw[..., Metric.RANK_IC, 1] = np.isfinite(rank_ic)
    if horizons is None:
        horizons = tuple(range(1, horizon_count + 1))
    return MetricStats(horizons, raw)


def population():
    first = terminal("x")
    second = terminal("y")
    third = make_expr("add", first, second)
    return [Candidate(expr, expr.key) for expr in (first, second, third)]


def test_stats_owns_immutable_raw_sum_count_and_zero_count_means_nan():
    raw = np.zeros((2, 2, METRIC_COUNT, 2), dtype=np.float64)
    raw[0, 0, Metric.RANK_IC] = [0.75, 3]
    result = MetricStats([5, 30], raw)
    raw[0, 0, Metric.RANK_IC] = [-1, 1]
    assert result.horizons == (5, 30)
    assert result.stats.dtype == np.float64
    assert result.stats[0, 0, Metric.RANK_IC, 0] == 0.75
    assert result.means[0, 0, Metric.RANK_IC] == 0.25
    assert np.isnan(result.means[1]).all()
    with pytest.raises(ValueError):
        result.stats[0, 0, 0, 0] = 3
    with pytest.raises(ValueError):
        result.stats.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "horizons", (3,))
    means = result.means
    means[0, 0, Metric.RANK_IC] = 99
    assert result.means[0, 0, Metric.RANK_IC] == 0.25


@pytest.mark.parametrize("shape", [(1, 2, 2), (1, 2, 3, 2), (1, 2, 2, 3), (0, 2, 2, 2)])
def test_stats_rejects_misaligned_or_empty_raw_shape(shape):
    with pytest.raises(ValueError):
        MetricStats((5, 30), np.zeros(shape))


@pytest.mark.parametrize("horizons", [(), (0,), (-5,), (True,), (1.5,), (5, 5)])
def test_stats_rejects_invalid_horizon_identity(horizons):
    with pytest.raises(ValueError):
        MetricStats(horizons, np.zeros((1, len(horizons), METRIC_COUNT, 2)))


@pytest.mark.parametrize(
    "pair", [(np.nan, 1), (np.inf, 1), (0, np.inf), (0, -1), (0, 0.5), (0, 2**53 + 2), (1, 0)]
)
def test_stats_rejects_invalid_sum_count(pair):
    raw = np.zeros((1, 1, METRIC_COUNT, 2))
    raw[0, 0, Metric.RANK_IC] = pair
    with pytest.raises(ValueError):
        MetricStats((30,), raw)


def test_single_horizon_matches_existing_rank_ic_complexity_scoring():
    stats = make_stats([[0.25], [-0.5], [np.nan], [0.25]])
    complexities = [1, 3, 1, 1]
    _, expected, _ = fitness_and_order(stats.stats[:, 0, Metric.RANK_IC], complexities, 0.125)
    np.testing.assert_array_equal(
        multitarget_fitness(stats, complexities, complexity_penalty=0.125), expected
    )


def test_weights_normalize_once_and_default_is_equal_horizon_weight():
    stats = make_stats([[0.2, 0.8], [0.6, -0.2]])
    np.testing.assert_allclose(multitarget_fitness(stats, [1, 2]), [0.5, 0.2])
    np.testing.assert_allclose(
        multitarget_fitness(stats, [1, 2], weights=[1, 3], complexity_penalty=0.1),
        [0.55, -0.2],
    )
    np.testing.assert_allclose(
        multitarget_fitness(stats, [1, 2], weights=[1e308, 1e308]), [0.5, 0.2]
    )


def test_zero_weight_horizon_does_not_contaminate_rank_ic():
    stats = make_stats([[0.5, np.nan], [0.4, -0.9]])
    np.testing.assert_array_equal(multitarget_fitness(stats, [1, 1], weights=[1, 0]), [0.5, 0.4])
    np.testing.assert_array_equal(multitarget_fitness(stats, [1, 1]), [-np.inf, -0.25])


@pytest.mark.parametrize("weights", [[], [1], [[1, 1]], [0, 0], [-1, 2], [np.nan, 1], [np.inf, 1]])
def test_invalid_weights_never_fall_back_to_equal_weight(weights):
    with pytest.raises(ValueError):
        multitarget_fitness(make_stats([[0.5, 0.3]]), [1], weights=weights)


@pytest.mark.parametrize("complexities", [[], [1, 2], [[1]], [-1], [np.nan], [np.inf]])
def test_invalid_complexities_are_rejected(complexities):
    with pytest.raises(ValueError):
        multitarget_fitness(make_stats([[0.5]]), complexities)


@pytest.mark.parametrize(
    "option,value",
    [
        ("complexity_penalty", -1),
        ("complexity_penalty", np.inf),
    ],
)
def test_invalid_scalar_constraints_are_rejected(option, value):
    with pytest.raises(ValueError):
        multitarget_fitness(make_stats([[0.5]]), [1], **{option: value})


def test_custom_receives_raw_stats_and_complexities_then_enforces_finiteness():
    stats = make_stats(
        [[np.nan, 0.4], [0.1, 0.2], [0.5, 0.5], [0.6, 0.5]],
    )
    seen = []

    def custom(raw, complexities):
        seen.append(raw)
        assert not complexities.flags.writeable
        np.testing.assert_array_equal(complexities, [1, 2, 3, 4])
        return np.array([4.0, 3.0, np.nan, np.inf])

    scores = multitarget_fitness(stats, [1, 2, 3, 4], custom=custom, complexity_penalty=1)
    assert seen == [stats]
    np.testing.assert_array_equal(scores, [4, 3, -np.inf, -np.inf])


@pytest.mark.parametrize("output", [1, [1, 2], [[1]], np.zeros((1, 1, 1))])
def test_custom_output_shape_is_not_broadcast(output):
    with pytest.raises(ValueError):
        multitarget_fitness(make_stats([[0.5]]), [1], custom=lambda *_: output)


def test_custom_exceptions_propagate():
    def custom(*_):
        raise RuntimeError("scoring failure")

    with pytest.raises(RuntimeError, match="scoring failure"):
        multitarget_fitness(make_stats([[0.5]]), [1], custom=custom)


def test_rank_preserves_population_raw_stats_and_stable_ties_with_per_horizon_topk():
    candidates = population()
    stats = make_stats([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]], horizons=(5, 30))
    result = rank_multitarget_candidates(candidates, stats, top_k=2)
    assert result.population == tuple(candidates)
    assert result.stats is stats
    assert result.order == (0, 1, 2)
    assert result.top_indices == (0, 1)
    assert result.horizon_top_indices == {5: (0, 2), 30: (1, 2)}
    np.testing.assert_array_equal(result.fitness, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        result.fitness.setflags(write=True)


def test_horizon_topk_is_independent_of_aggregate_weight():
    stats = make_stats([[0.7, 0.9], [0.4, 0.8], [0.5, 0.6]], horizons=(5, 30))
    result = rank_multitarget_candidates(population(), stats, weights=[1, 0], top_k=2)
    assert result.order == (0, 2, 1)
    assert result.horizon_top_indices == {5: (0, 2), 30: (0, 1)}


def test_rank_custom_changes_aggregate_order_and_preserves_stable_invalid_order():
    stats = make_stats([[0.7], [0.4], [0.5]])
    result = rank_multitarget_candidates(
        population(), stats, custom=lambda *_: [np.nan, 3, np.inf], top_k=None
    )
    assert result.order == (1, 0, 2)
    assert result.top_indices == result.order
    assert result.horizon_top_indices == {1: (0, 2, 1)}


@pytest.mark.parametrize("top_k", [-1, True, 1.5])
def test_rank_validates_top_k(top_k):
    with pytest.raises(ValueError):
        rank_multitarget_candidates(population(), make_stats([[0.5], [0.4], [0.3]]), top_k=top_k)


def test_rank_rejects_population_alignment_and_clamps_large_topk():
    stats = make_stats([[0.5], [0.4], [0.3]])
    with pytest.raises(ValueError):
        rank_multitarget_candidates(population()[:2], stats)
    assert rank_multitarget_candidates(population(), stats, top_k=99).top_indices == (0, 1, 2)
    result = rank_multitarget_candidates(population(), stats, top_k=0)
    assert result.top_indices == ()
    assert result.horizon_top_indices == {1: ()}


def test_complex_sum_count_is_not_silently_projected_to_real():
    raw = np.zeros((1, 1, METRIC_COUNT, 2), dtype=np.complex128)
    raw[0, 0, Metric.RANK_IC] = [0.2 + 0.1j, 1]
    with pytest.raises(ValueError):
        MetricStats((30,), raw)


@pytest.mark.parametrize("option", ["weights", "complexities", "custom"])
def test_complex_scoring_input_is_not_silently_projected_to_real(option):
    options: dict = {"complexities": [1]}
    if option == "custom":
        options[option] = lambda *_: np.array([0.5 + 0.1j])
    else:
        options[option] = np.array([1 + 0.1j])
    with pytest.raises(ValueError):
        multitarget_fitness(make_stats([[0.5]]), **options)


@pytest.mark.parametrize("option", ["complexity_penalty"])
@pytest.mark.parametrize("value", ["0.5", [0.5], 0.5 + 0.1j])
def test_nonreal_scalar_constraints_raise_value_error(option, value):
    with pytest.raises(ValueError):
        multitarget_fitness(make_stats([[0.5]]), [1], **{option: value})


@pytest.mark.parametrize(
    "metric,pair",
    [
        (Metric.RANK_IC, (1.1, 1)),
        (Metric.PEARSON, (-1.1, 1)),
    ],
)
def test_stats_rejects_impossible_metric_totals(metric, pair):
    raw = np.zeros((1, 1, METRIC_COUNT, 2))
    raw[0, 0, metric] = pair
    with pytest.raises(ValueError):
        MetricStats((30,), raw)


def test_stats_accepts_correlation_reduction_roundoff_without_clipping_raw_totals():
    raw = np.zeros((1, 1, METRIC_COUNT, 2))
    raw[0, 0, Metric.RANK_IC] = [1 + 1e-12, 1]
    stats = MetricStats((30,), raw)
    assert stats.stats[0, 0, Metric.RANK_IC, 0] == 1 + 1e-12


def test_positive_weight_that_underflows_after_normalization_is_still_constrained():
    stats = make_stats([[0.5, np.nan]])
    np.testing.assert_array_equal(
        multitarget_fitness(stats, [1], weights=[1e308, 1e-308]), [-np.inf]
    )
