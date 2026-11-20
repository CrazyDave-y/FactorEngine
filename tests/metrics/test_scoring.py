"""用手算例子核对有序累计、原始结果和稳定 Fitness 排名。"""

import numpy as np
import pytest


def test_complete_result_preserves_ordered_sum_invalid_roots_and_stable_ties():
    # 这里同时包含不同批次、无效分数和同分排序。
    from factor_engine.metrics import LaneAccumulator

    accumulator = LaneAccumulator(4, 3)
    accumulator.commit(
        0, np.array([[0.5, -0.5], [0.25, 0.25], [np.nan, np.nan], [0.0, -0.0]], dtype=np.float64)
    )
    accumulator.commit(2, np.array([[0.75], [0.25], [np.nan], [np.nan]], dtype=np.float64))
    from factor_engine.fitness import fitness_and_order

    result = accumulator.finish()
    raw, fitness, order = fitness_and_order(result.stats, [1, 1, 3, 0], 0.125)
    np.testing.assert_array_equal(result.stats, [[0.75, 3], [0.75, 3], [0, 0], [0, 2]])
    np.testing.assert_array_equal(result.raw_ic, [0.25, 0.25, np.nan, 0.0])
    np.testing.assert_array_equal(fitness, [0.125, 0.125, -np.inf, 0.0])
    assert tuple(order) == (0, 1, 3, 2)
    assert result.stats.dtype == np.float64
    assert not result.stats.flags.writeable


def test_accumulator_adds_each_lane_in_global_order_without_subtotals():
    from factor_engine.metrics import LaneAccumulator

    accumulator = LaneAccumulator(1, 4)
    # ((1 + 2**-53) - 1) + 2**-53 得到 2**-53；先合并批总和则会得到零。
    accumulator.commit(0, np.array([[1.0, 2.0**-53]], dtype=np.float64))
    accumulator.commit(2, np.array([[-1.0, 2.0**-53]], dtype=np.float64))
    assert accumulator.finish().stats[0, 0] == 2.0**-53


@pytest.mark.parametrize("first,width", [(1, 1), (0, 4)])
def test_accumulator_rejects_missing_out_of_order_and_oversized_lane_windows(first, width):
    from factor_engine.metrics import LaneAccumulator

    with pytest.raises(ValueError, match="lane"):
        LaneAccumulator(4, 3).commit(first, np.zeros((4, width), dtype=np.float64))


def test_failed_accumulator_cannot_publish_stale_or_partial_score():
    from factor_engine.metrics import LaneAccumulator

    accumulator = LaneAccumulator(4, 3)
    accumulator.commit(0, np.ones((4, 1), dtype=np.float64))
    accumulator.invalidate()
    with pytest.raises(RuntimeError, match="invalid"):
        accumulator.finish()
    with pytest.raises(RuntimeError, match="invalid"):
        accumulator.commit(1, np.ones((4, 2), dtype=np.float64))


def test_unfinished_or_nonfinite_stats_cannot_be_labeled_complete():
    from factor_engine.metrics import LaneAccumulator, ScoreStats

    with pytest.raises(ValueError, match="complete"):
        LaneAccumulator(4, 3).finish()
    with pytest.raises(ValueError, match="invalid"):
        ScoreStats(np.array([[np.nan, 3], [0, 0], [0, 0], [0, 0]]))


def test_finalizer_rejects_fractional_or_excess_lane_counts():
    from factor_engine.metrics import LaneAccumulator

    for count in (0.5, 4, -1):
        accumulator = LaneAccumulator(1, 3)
        accumulator.commit(0, np.zeros((1, 3), dtype=np.float64))
        accumulator.stats[0, 1] = count
        with pytest.raises(ValueError, match="count"):
            accumulator.finish()


def test_scores_compute_mean_but_never_fitness():
    from factor_engine.metrics import ScoreStats

    values = np.array([[1.5, 2], [0.0, 0], [-2.0, 2]], np.float64)
    result = ScoreStats(values, {"lanes": 2})
    np.testing.assert_array_equal(result.raw_ic, [0.75, np.nan, -1.0])
    values[0, 0] = 0
    assert result.stats[0, 0] == 1.5
    assert not result.stats.flags.writeable
    assert not hasattr(result, "fitness")


@pytest.mark.parametrize(
    "values",
    [
        [[np.inf, 1]],
        [[2.0, 1]],
        [[1.0, 0]],
        [[0.0, -1]],
        [[0.0, 0.5]],
        [[0.0]],
    ],
)
def test_scores_reject_invalid_totals(values):
    from factor_engine.metrics import ScoreStats

    with pytest.raises(ValueError):
        ScoreStats(np.array(values, np.float64), {"lanes": 4})


def test_accumulator_keeps_identical_order_across_partitions():
    from factor_engine.metrics import LaneAccumulator

    rng = np.random.default_rng(813)
    values = rng.uniform(-1, 1, (7, 19))
    values[2, 5:8] = np.nan
    expected = np.zeros((7, 2), np.float64)
    for root in range(7):
        for value in values[root]:
            if np.isfinite(value):
                expected[root, 0] += value
                expected[root, 1] += 1
    for width in (1, 2, 5, 19):
        accumulator = LaneAccumulator(7, 19)
        for first in range(0, 19, width):
            accumulator.commit(first, values[:, first : first + width])
        actual = accumulator.finish().stats
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_diagnostic_summary_is_bounded_and_preserves_totals():
    from factor_engine.metrics import BatchTelemetry

    for _ in range(2):
        log = BatchTelemetry()
        for i in range(500):
            log.add({"upload_h2d_bytes": 100, "kernel_launches": 3, "owned_device_bytes": 1000 + i})
        assert log.batches == 500
        assert log.summary["upload_h2d_bytes"] == 50000
        assert log.summary["kernel_launches"] == 1500
        assert log.summary["owned_device_bytes"] == 1499
        assert not hasattr(log, "records")
