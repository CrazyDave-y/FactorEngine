from pathlib import Path

import numpy as np
import pytest


def api():
    from factor_engine.metric_types import METRIC_COUNT, Metric
    from factor_engine.target_bank import TargetBatch

    from tests.reference.metrics import reference_conditional_rank, reference_multitarget

    return Metric, METRIC_COUNT, TargetBatch, reference_multitarget, reference_conditional_rank


def pairwise_ranks(values):
    # 两两比较直接定义平均秩，避免与被测排序实现共用推导。
    return np.array(
        [np.count_nonzero(values < x) + 0.5 * (np.count_nonzero(values == x) - 1) for x in values],
        dtype=np.float64,
    )


def independent_pearson(x, y):
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    a = x.astype(np.longdouble)
    b = y.astype(np.longdouble)
    a -= a.mean()
    b -= b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def test_metric_names_and_numeric_contract():
    from factor_engine.metric_types import NativeLimit

    Metric, count, *_ = api()
    assert NativeLimit.ABI_VERSION == 4
    assert count == 2
    assert [(m.name, int(m)) for m in Metric] == [
        ("PEARSON", 0),
        ("RANK_IC", 1),
    ]


@pytest.mark.parametrize("time", [1, 2, 3, 31, 32, 33, 127, 257])
@pytest.mark.parametrize("distribution", ["normal", "heavy", "ties", "constant", "zeros"])
def test_multitarget_matches_independent_pairwise_oracle(time, distribution):
    Metric, count, TargetBatch, reference, _ = api()
    rng = np.random.default_rng(102 + time)
    factors = rng.normal(size=(3, 2, time)).astype(np.float32)
    target = rng.normal(size=(3, 2, time)).astype(np.float32)
    if distribution == "heavy":
        factors = (rng.standard_t(1, factors.shape) * 1e12).astype(np.float32)
        target = (rng.standard_t(1, target.shape) * 1e12).astype(np.float32)
    elif distribution == "ties":
        factors = np.round(factors)
        target = np.round(target)
    elif distribution in ("constant", "zeros"):
        factors[0] = 1 if distribution == "constant" else 0
        target[0] = -1 if distribution == "constant" else -0.0
    if time > 2:
        factors[0, 0, 1] = np.inf
        factors[1, 1, 0] = np.nan
        target[0, 1, 2] = -np.inf
        target[1, 0, 0] = np.nan
    eligible = rng.random(target.shape) > 0.2
    actual = reference(factors, TargetBatch((2, 5, 30), target, eligible))
    assert actual.shape == (3, 2, 3, count) and actual.dtype == np.float64
    for r in range(3):
        for lane in range(2):
            for horizon in range(3):
                good = eligible[horizon, lane] & np.isfinite(factors[r, lane])
                good &= np.isfinite(target[horizon, lane])
                x, y = factors[r, lane, good], target[horizon, lane, good]
                expected = [
                    independent_pearson(x, y),
                    independent_pearson(pairwise_ranks(x), pairwise_ranks(y)),
                ]
                np.testing.assert_allclose(
                    actual[r, lane, horizon], expected, rtol=2e-13, atol=2e-13, equal_nan=True
                )


def test_joint_masks_with_equal_counts_rebuild_ranks_and_zero_ties():
    Metric, _, TargetBatch, reference, conditional = api()
    factors = np.array([[[0.0, -0.0, 1, 2, 3, 4]]], dtype=np.float32)
    target = np.array([[[3, 1, 1, 0, -2, 4]], [[3, 1, 1, 0, -2, 4]]], dtype=np.float32)
    masks = np.array([[[1, 1, 1, 1, 0, 0]], [[0, 1, 1, 0, 1, 1]]], dtype=bool)
    order = np.argsort(factors[0, 0], kind="stable")
    a = conditional(factors[0, 0], order, masks[0, 0])
    b = conditional(factors[0, 0], order, masks[1, 0])
    np.testing.assert_array_equal(a, [1, 1, 4, 6, -1, -1])
    np.testing.assert_array_equal(b, [-1, 0, 2, -1, 4, 6])
    actual = reference(factors, TargetBatch((2, 5), target, masks))
    assert actual[0, 0, 0, Metric.RANK_IC] != actual[0, 0, 1, Metric.RANK_IC]


@pytest.mark.parametrize("time", [0, 1, 2, 3, 32, 33, 257])
def test_conditional_rank_matches_pairwise_definition(time):
    *_, conditional = api()
    rng = np.random.default_rng(time)
    values = np.round(rng.normal(size=time)).astype(np.float32)
    if time >= 3:
        values[-3:] = [np.nan, -np.inf, np.inf]
    mask = rng.random(time) > 0.3
    order = np.argsort(values, kind="stable")
    actual = conditional(values, order, mask)
    good = mask & np.isfinite(values)
    expected = np.full(time, -1, dtype=np.int64)
    expected[good] = (pairwise_ranks(values[good]) * 2).astype(np.int64)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64


def test_conditional_rank_validates_full_sorted_permutation_and_bool_mask():
    *_, conditional = api()
    values = np.array([2, 1, 1, np.nan], dtype=np.float32)
    mask = np.ones(4, dtype=bool)
    for order in ([0, 1, 2, 3], [1, 1, 0, 3], [1, 2, 0], [1, 2, 0, 4], [1.0, 2.0, 0.0, 3.0]):
        with pytest.raises(ValueError, match="order"):
            conditional(values, np.asarray(order), mask)
    with pytest.raises(ValueError, match="mask"):
        conditional(values, np.array([1, 2, 0, 3]), mask.astype(int))


def test_extreme_fp32_values_do_not_overflow_or_drop_finite_metrics():
    Metric, _, TargetBatch, reference, _ = api()
    max32 = np.finfo(np.float32).max
    tiny32 = np.nextafter(np.float32(0), np.float32(1))
    factors = np.array(
        [
            [[-max32, -max32 / 2, 0, max32 / 2, max32]],
            [[-tiny32, 0, tiny32, 2 * tiny32, 3 * tiny32]],
        ],
        dtype=np.float32,
    )
    targets = TargetBatch((1,), -factors[:1], np.ones((1, 1, 5), dtype=bool))
    actual = reference(factors, targets)
    np.testing.assert_allclose(actual[..., Metric.PEARSON], -1, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual[..., Metric.RANK_IC], -1, rtol=1e-14, atol=1e-14)


def test_multitarget_rejects_misaligned_root_shape():
    _, _, TargetBatch, reference, _ = api()
    target = TargetBatch((1,), np.ones((1, 2, 3)), np.ones((1, 2, 3), dtype=bool))
    for factors in (np.ones((2, 3)), np.ones((1, 2, 4)), np.ones((0, 2, 3))):
        with pytest.raises(ValueError, match="factors"):
            reference(factors, target)


def test_fp64_constant_factors_keep_nan_correlation_despite_centering_roundoff():
    Metric, _, TargetBatch, reference, _ = api()
    factors = np.full((1, 1, 7), 0.1, dtype=np.float64)
    targets = TargetBatch((1,), np.arange(7).reshape(1, 1, 7), np.ones((1, 1, 7), dtype=bool))
    result = reference(factors, targets)[0, 0, 0]
    assert np.isnan(result[Metric.PEARSON])
    assert np.isnan(result[Metric.RANK_IC])


@pytest.mark.parametrize("horizons", [(30,), (5, 30, 300), (2, 5, 8, 15, 30, 60, 120, 300, 900)])
def test_h1_h3_h9_external_targets_share_one_factor_lane_axis(horizons):
    Metric, _, TargetBatch, reference, _ = api()
    rng = np.random.default_rng(93)
    factors = rng.normal(size=(4, 3, 67)).astype(np.float32)
    values = rng.normal(size=(len(horizons), 3, 67)).astype(np.float32)
    eligible = rng.random(values.shape) > 0.25
    actual = reference(factors, TargetBatch(horizons, values, eligible))
    for h in range(len(horizons)):
        isolated = reference(
            factors, TargetBatch((horizons[h],), values[h : h + 1], eligible[h : h + 1])
        )
        np.testing.assert_array_equal(actual[:, :, h], isolated[:, :, 0])
    assert np.isfinite(actual[..., Metric.PEARSON]).all()


def test_empty_joint_mask_and_two_observations_are_nan():
    Metric, _, TargetBatch, reference, _ = api()
    factors = np.array([[[1, 1, np.nan, np.inf]]], dtype=np.float32)
    values = np.array([[[2, 3, 4, 5]], [[2, 3, 4, 5]]], dtype=np.float32)
    eligible = np.array([[[False] * 4], [[True] * 4]], dtype=bool)
    result = reference(factors, TargetBatch((1, 2), values, eligible))[0, 0]
    assert np.isnan(result[:, Metric.PEARSON]).all()
    assert np.isnan(result[:, Metric.RANK_IC]).all()


@pytest.mark.parametrize(
    ("factor", "target"),
    [
        (np.arange(1, 4, dtype=np.float64) * 1e200, np.arange(1, 4)),
        (np.arange(1, 4, dtype=np.float64) * 1e-200, np.arange(1, 4)),
        (np.array([-3, -1, 0, 2, 5], dtype=np.float64) * 1e200, [-3, -1, 0, 2, 5]),
        (np.array([-3, -1, 0, 2, 5], dtype=np.float64) * 1e-200, [-3, -1, 0, 2, 5]),
        (np.array([-1, -0.5, 0, 0.5, 1]) * np.finfo(np.float64).max, [-2, -1, 0, 1, 2]),
        (np.arange(1, 4) * np.nextafter(np.float64(0), np.float64(1)), [1, 2, 3]),
        (np.ldexp(1 + np.arange(-2, 3) * np.finfo(np.float64).eps, 700), [-2, -1, 0, 1, 2]),
        (np.ldexp(1 + np.arange(-2, 3) * np.finfo(np.float64).eps, -500), [-2, -1, 0, 1, 2]),
    ],
    ids=[
        "large",
        "small",
        "large-mixed",
        "small-mixed",
        "finite-limit",
        "subnormal",
        "large-offset",
        "small-offset",
    ],
)
def test_finite_fp64_extremes_keep_defined_pearson(factor, target):
    Metric, _, TargetBatch, reference, _ = api()
    values = np.asarray(target, dtype=np.float32)[None, None]
    targets = TargetBatch((1,), values, np.ones(values.shape, dtype=bool))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        result = reference(factor[None, None], targets)[0, 0, 0]
    assert np.isfinite(factor).all()
    np.testing.assert_allclose(result[Metric.PEARSON], 1, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(result[Metric.RANK_IC], 1, rtol=3e-15, atol=3e-15)


def test_normal_fp32_large_offset_preserves_original_two_pass_rounding():
    Metric, _, TargetBatch, reference, _ = api()
    rng = np.random.default_rng(731)
    base = np.float32(1e30)
    factor = base + np.spacing(base) * rng.integers(-8, 9, 31).astype(np.float32)
    target = rng.normal(size=31).astype(np.float32)
    a = factor.astype(np.float64)
    b = target.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    expected = float(np.clip((a @ b) / np.sqrt(a @ a) / np.sqrt(b @ b), -1, 1))
    values = target[None, None]
    targets = TargetBatch((1,), values, np.ones(values.shape, dtype=bool))
    result = reference(factor[None, None], targets)[0, 0, 0, Metric.PEARSON]
    assert result == expected
    np.testing.assert_allclose(result, independent_pearson(factor, target), rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    ("factor", "target"),
    [
        (np.array([2**63, 2**63 + 1, 2**63 + 2], dtype=np.uint64), [1, 2, 3]),
        (np.array([2**62, 2**62 + 1, 2**62 + 2], dtype=np.int64), [1, 2, 3]),
        (np.array([-(2**63), -(2**63) + 1, -(2**63) + 2], dtype=np.int64), [1, 2, 3]),
        (np.array([2**64 - 1, 2**64 - 2, 2**64 - 3], dtype=np.uint64), [3, 2, 1]),
        (np.array([-(2**63), 0, 2**63 - 1], dtype=np.int64), [-1, 0, 1]),
        (np.array([0, 2**63, 2**64 - 1], dtype=np.uint64), [0, 1, 2]),
    ],
    ids=[
        "uint64-offset",
        "int64-offset",
        "int64-min-offset",
        "uint64-descending",
        "int64-full-range",
        "uint64-full-range",
    ],
)
def test_integer_factors_remove_offset_before_fp64_conversion(factor, target):
    Metric, _, TargetBatch, reference, _ = api()
    values = np.asarray(target, dtype=np.float32)[None, None]
    targets = TargetBatch((1,), values, np.ones(values.shape, dtype=bool))
    result = reference(factor[None, None], targets)[0, 0, 0]
    np.testing.assert_allclose(result[Metric.PEARSON], 1, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(result[Metric.RANK_IC], 1, rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize("scale", [1e200, -1e200, 1e-200, -1e-200])
def test_extreme_scale_nonperfect_correlations_match_extended_precision(scale):
    Metric, _, TargetBatch, reference, _ = api()
    factor = np.array([-3, -1, 0, 2, 5], dtype=np.float64) * scale
    target = np.array([2, -1, 4, 0, -3], dtype=np.float32)
    values = target[None, None]
    targets = TargetBatch((1,), values, np.ones(values.shape, dtype=bool))
    result = reference(factor[None, None], targets)[0, 0, 0]
    np.testing.assert_allclose(
        result[Metric.PEARSON], independent_pearson(factor, target), rtol=3e-15, atol=3e-15
    )


def direct_and_analytic(groups):
    n = sum(groups)
    direct = 0
    start = 0
    correction = 0
    for m in groups:
        doubled_average = 2 * start + m - 1
        direct += m * (doubled_average - (n - 1)) ** 2
        correction += m**3 - m
        start += m
    numerator = n * (n * n - 1) - correction
    assert numerator % 3 == 0
    assert 0 <= numerator < 2**63
    assert direct == numerator // 3


def compositions(n):
    if n == 0:
        yield []
        return
    for cuts in range(1 << (n - 1)):
        group = []
        m = 1
        for pos in range(n - 1):
            if cuts & (1 << pos):
                group.append(m)
                m = 1
            else:
                m += 1
        yield group + [m]


@pytest.mark.parametrize("n", range(11))
def test_all_small_tie_partitions(n):
    for groups in compositions(n):
        direct_and_analytic(groups)


@pytest.mark.parametrize("n", [31, 32, 33, 511, 512, 513, 8192, 8193, 23400, 23552, 24576])
def test_large_integer_boundaries(n):
    direct_and_analytic([n])
    direct_and_analytic([1] * n)
    rng = np.random.default_rng(n)
    for _ in range(20):
        cuts = np.unique(np.r_[0, rng.integers(0, n + 1, 30), n])
        direct_and_analytic(np.diff(cuts).tolist())


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_stats_converts_strided_input_to_owned_immutable_fp64(dtype):
    from factor_engine.metric_types import METRIC_COUNT, Metric
    from factor_engine.metrics import MetricStats

    raw = np.zeros((1, 2, METRIC_COUNT, 4), dtype=dtype)[..., ::2]
    raw[0, 0, Metric.RANK_IC] = [1, 2]
    stats = MetricStats((5, 30), raw)
    raw.fill(0)
    assert stats.stats.dtype == np.float64
    assert stats.stats.flags.c_contiguous
    assert stats.stats[0, 0, Metric.RANK_IC].tolist() == [1, 2]
    assert stats.means[0, 0, Metric.RANK_IC] == 0.5
    assert np.isnan(stats.means[0, 1]).all()
    with pytest.raises(ValueError):
        stats.stats.setflags(write=True)
