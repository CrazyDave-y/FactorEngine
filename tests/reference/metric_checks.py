"""独立校验生产评分结果，不执行计时或参数搜索。"""

PEARSON_RTOL = 2e-12
PEARSON_ATOL = 2e-12
# FP32 中心矩对照 FP64 参考的联合 gate；近零项由绝对误差项约束。
FP32_PEARSON_RTOL = 2e-5
FP32_PEARSON_ATOL = 3e-7


def _pearson_errors(actual, expected):
    import numpy as np

    valid = np.isfinite(expected)
    diff = np.abs(actual[valid] - expected[valid])
    denominator = np.abs(expected[valid])
    nonzero = denominator != 0
    return {
        "max_abs": float(diff.max()) if diff.size else None,
        "max_relative_nonzero_reference": (
            float((diff[nonzero] / denominator[nonzero]).max()) if nonzero.any() else None
        ),
        "zero_reference_count": int((~nonzero).sum()),
        "zero_reference_nonzero_error_count": int(((~nonzero) & (diff != 0)).sum()),
        "nan_mismatch_count": int(np.count_nonzero(np.isnan(actual) != np.isnan(expected))),
        "finite_count": int(valid.sum()),
    }


def assert_same(actual, expected):
    import numpy as np

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    integer = np.uint32 if actual.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(actual[finite].view(integer), expected[finite].view(integer))
    np.testing.assert_array_equal(actual[~finite], expected[~finite])


def assert_metric_parity(actual, expected, *, oracle=False, fused=False, exact=False):
    """GPU 策略秩指标逐 bit 相同；独立 NumPy 求和允许 FP64 舍入误差。"""
    import numpy as np
    from factor_engine.metric_types import METRIC_COUNT, Metric

    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.dtype != np.float64
        or actual.shape != expected.shape
        or actual.shape[-1] != METRIC_COUNT
    ):
        raise AssertionError("metric detail shape/dtype differs")
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    if np.isinf(actual).any() or np.isinf(expected).any():
        raise AssertionError("metric details contain infinity")
    pearson_actual = actual[..., Metric.PEARSON]
    pearson_expected = expected[..., Metric.PEARSON]
    errors = None
    if fused:
        errors = _pearson_errors(pearson_actual, pearson_expected)
        np.testing.assert_allclose(
            pearson_actual,
            pearson_expected,
            rtol=FP32_PEARSON_RTOL,
            atol=FP32_PEARSON_ATOL,
            equal_nan=True,
            err_msg=f"FP32 Pearson gate failed; raw errors: {errors}",
        )
    elif exact:
        assert_same(pearson_actual, pearson_expected)
    else:
        np.testing.assert_allclose(
            pearson_actual,
            pearson_expected,
            rtol=PEARSON_RTOL,
            atol=PEARSON_ATOL,
            equal_nan=True,
        )
    for metric in Metric:
        if metric == Metric.PEARSON:
            continue
        a, b = actual[..., metric], expected[..., metric]
        if oracle and metric == Metric.RANK_IC:
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12, equal_nan=True)
        else:
            finite = np.isfinite(b)
            np.testing.assert_array_equal(
                np.ascontiguousarray(a[finite]).view(np.uint64),
                np.ascontiguousarray(b[finite]).view(np.uint64),
            )

    return errors


def stats_from_details(horizons, details):
    """与公开合同相同，按原 lane 次序串行汇总，NaN 不进入 Sum/Count。"""
    import numpy as np
    from factor_engine.metric_types import METRIC_COUNT
    from factor_engine.metrics import MetricStats

    details = np.asarray(details, dtype=np.float64)
    if details.ndim != 4 or details.shape[2:] != (len(horizons), METRIC_COUNT):
        raise ValueError("aligned [root,lane,horizon,metric] details required")
    stats = np.zeros((details.shape[0], len(horizons), METRIC_COUNT, 2), dtype=np.float64)
    for lane in range(details.shape[1]):
        values = details[:, lane]
        good = np.isfinite(values)
        stats[..., 0] += np.where(good, values, 0)
        stats[..., 1] += good
    return MetricStats(horizons, stats)


def assert_stats_parity(actual, expected, *, fused=False, exact=False):
    import numpy as np
    from factor_engine.metric_types import Metric

    if actual.horizons != expected.horizons:
        raise AssertionError("aggregate horizons differ")
    np.testing.assert_array_equal(actual.stats[..., 1], expected.stats[..., 1])
    if fused:
        # 逐 lane 的 FP32 绝对误差限只能直接用于均值；Sum 仍完整保留原始误差。
        errors = {
            "sums": _pearson_errors(
                actual.stats[..., Metric.PEARSON, 0], expected.stats[..., Metric.PEARSON, 0]
            ),
            "means": assert_metric_parity(actual.means, expected.means, fused=True),
        }
        assert_same(actual.stats[..., Metric.RANK_IC, 0], expected.stats[..., Metric.RANK_IC, 0])
        return errors
    return assert_metric_parity(
        actual.stats[..., 0], expected.stats[..., 0], fused=fused, exact=exact
    )
