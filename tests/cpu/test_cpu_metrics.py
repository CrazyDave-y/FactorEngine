"""原生 CPU Pearson/RankIC的数值、掩码及流式累计合同。"""

import os
from pathlib import Path

import numpy as np
import pytest

from factor_engine.cpu_backend import CPUBackend
from factor_engine.data import Panel
from factor_engine.expression import terminal, neg, rolling_mean
from factor_engine.ir import compile_expressions
from tests.reference.metrics import reference_multitarget
from factor_engine.metric_types import Metric, METRIC_COUNT
from factor_engine.native_plan_backend import NativePlanBackend
from factor_engine.session_source import ArraySource
from factor_engine.target_bank import MultiTargetSource, TargetBatch


@pytest.fixture
def library():
    path = os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB")
    if not path or not Path(path).is_file():
        pytest.skip("explicit native CPU build required")
    return Path(path)


def numerical_case(time):
    rng = np.random.default_rng(7201)
    x = rng.standard_normal((3, 8, time), dtype=np.float32)
    y = rng.standard_normal((2, 8, time), dtype=np.float32)
    x[1] = np.rint(x[1])
    x[2] = -x[0]
    x[:, 1] = np.float32(7)  # 因子常数。
    y[:, 2] = np.float32(-4)  # 目标常数。
    x[:, 3] = np.float32(1e20) + np.arange(time, dtype=np.float32) * np.float32(2e13)
    y[:, 3] = np.arange(time, dtype=np.float32)
    x[:, 4] = rng.choice([-1, 1], (3, time)) * np.finfo(np.float32).max
    y[:, 4] = rng.choice([-1, 1], (2, time)) * np.finfo(np.float32).max
    x[1, 4] = rng.integers(0, 2**32, time, dtype=np.uint32).view(np.float32)
    y[1, 4] = rng.integers(0, 2**32, time, dtype=np.uint32).view(np.float32)
    smallest = np.nextafter(np.float32(0), np.float32(1))
    x[:, 5] = rng.integers(-6, 7, (3, time)).astype(np.float32) * smallest
    y[:, 5] = rng.integers(-6, 7, (2, time)).astype(np.float32) * smallest
    x[:, 6, ::3] = np.nan
    y[:, 6, 1::5] = np.inf
    x[:, 7, 0::2], x[:, 7, 1::2] = 0.0, -0.0
    mask = rng.random(y.shape) > 0.5
    mask[:, 0] = True
    mask[:, 6] = False
    mask[:, 6, :2] = True  # 联合有效数不足三点。
    return x, TargetBatch((1, 2), y, mask)


@pytest.mark.parametrize("time", [3, 97, 3600, 23400])
def test_native_pearson_and_all_metrics_against_independent_oracle(library, time, record_property):
    x, targets = numerical_case(time)
    expected = reference_multitarget(x, targets)
    native = NativePlanBackend(library)
    try:
        actual = native.pearson_lanes(x, targets.values[0], targets.eligible[0])
        oracle = expected[:, :, 0, Metric.PEARSON]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(oracle))
        np.testing.assert_allclose(actual, oracle, rtol=2e-12, atol=2e-12, equal_nan=True)
        valid = np.isfinite(oracle)
        error = np.abs(actual[valid] - oracle[valid])
        record_property("pearson_max_abs", float(error.max(initial=0)))
        nonzero = np.abs(oracle[valid]) > 0
        relative = error[nonzero] / np.abs(oracle[valid][nonzero])
        record_property("pearson_max_relative_nonzero", float(relative.max(initial=0)))
        panel = Panel({f"x{k}": x[k][None] for k in range(len(x))})
        plan = compile_expressions([terminal(name) for name in panel.fields])
        actual_all = native.score_metric_lanes(plan, panel, targets)
        np.testing.assert_array_equal(np.isnan(actual_all), np.isnan(expected))
        for metric in Metric:
            if metric in (Metric.PEARSON, Metric.RANK_IC):
                np.testing.assert_allclose(
                    actual_all[..., metric],
                    expected[..., metric],
                    rtol=2e-12,
                    atol=2e-12,
                    equal_nan=True,
                )
            else:
                np.testing.assert_array_equal(actual_all[..., metric], expected[..., metric])
    finally:
        native.close()


def test_cpu_multitarget_dag_and_batch_width_preserve_order(library):
    x, targets = numerical_case(97)
    panel = Panel({f"x{k}": x[k][None] for k in range(len(x))})
    source = MultiTargetSource(
        ArraySource(panel, targets.values[0][None], source_identity="cpu-metric-contract"), targets
    )
    expr = terminal("x0")
    plan = compile_expressions([expr, neg(expr), rolling_mean(expr, 5), terminal("x1")])
    workers = NativePlanBackend(library).cpu_layout()["max_threads"]
    results, details = [], []
    for width in (1, 3, 8):
        backend = CPUBackend(
            library, workers=workers, batch_lanes=width, memory_budget_bytes=1024**3
        )
        try:
            with pytest.raises(ValueError, match="prepare_multitarget"):
                backend.score_multitarget(plan, source)
            backend.prepare_multitarget(plan, source)
            result = backend.score_multitarget(plan, source, detail=True)
            assert backend.last_metric_details is not None
            assert backend.last_metric_details.shape == (4, 8, 2, METRIC_COUNT)
            details.append(backend.last_metric_details)
            results.append(result.stats)
            again = backend.score_multitarget(plan, source)
            assert backend.last_metric_details is None
            np.testing.assert_array_equal(result.stats.view(np.uint64), again.stats.view(np.uint64))
            # 独立地导出 DAG，检验原生融合入口没有拿错 root 或复用过期 slot。
            outputs = np.stack(backend.backend.evaluate(plan, panel))[:, 0]
            expected = reference_multitarget(outputs, targets)
            np.testing.assert_allclose(
                details[-1], expected, rtol=2e-12, atol=2e-12, equal_nan=True
            )
        finally:
            backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            backend.score_multitarget(plan, source)
    for result in results[1:]:
        np.testing.assert_array_equal(result.view(np.uint64), results[0].view(np.uint64))
    for value in details[1:]:
        np.testing.assert_array_equal(value.view(np.uint64), details[0].view(np.uint64))


def test_native_pearson_null_arguments_return_error(library):
    native = NativePlanBackend(library)
    try:
        assert native._pearson_fn(None, 1, None, None, 1, 3, None) != 0
        assert (
            native._metric_fn(None, 0, None, 0, 1, 3, 0, None, 0, None, 1, None, None, 1, None) != 0
        )
    finally:
        native.close()
