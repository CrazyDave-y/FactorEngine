"""滚动极值保持 CPU 的最新同值、无效窗口和 lane 边界语义。"""

from contextlib import closing

import numpy as np
import pytest

from factor_engine.cuda_native import input_payload_bytes
from factor_engine.metric_types import METRIC_COUNT
from factor_engine.data import Panel
from factor_engine.expression import rolling_max, rolling_mean, rolling_min, terminal
from factor_engine.ir import compile_expressions
from factor_engine.native_plan_backend import NativePlanBackend
from tests.cuda.test_cuda_numerics import export_values


def _assert_extrema_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))


def _signed_zero_case():
    panel = Panel(
        {
            "a": np.array([1.0, 0.0, -0.0, 2.0, -0.0, 0.0, 2.0], np.float32).reshape(1, 1, 7),
            "b": np.array([-1.0, -0.0, 0.0, -2.0, 0.0, -0.0, -2.0], np.float32).reshape(1, 1, 7),
        }
    )
    plan = compile_expressions([rolling_min(terminal("a"), 2), rolling_max(terminal("b"), 2)])
    expected = np.array(
        [[np.nan, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0], [np.nan, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0]],
        np.float32,
    ).reshape(2, 1, 1, 7)
    return panel, plan, expected


def test_native_extrema_contract_chooses_latest_signed_zero():
    panel, plan, expected = _signed_zero_case()
    with closing(NativePlanBackend()) as cpu:
        _assert_extrema_bits(cpu.evaluate(plan, panel), expected)


def test_cuda_extrema_chooses_latest_signed_zero(gpu_backend):
    panel, plan, expected = _signed_zero_case()
    _assert_extrema_bits(export_values(gpu_backend, plan, panel), expected)


def test_small_extrema_needs_no_prefix_scratch(gpu_backend):
    rng = np.random.default_rng(24)
    panel = Panel({"x": rng.normal(size=(1, 3, 257)).astype(np.float32)})
    target = rng.normal(size=panel.shape).astype(np.float32)
    plan = compile_expressions([rolling_min(terminal("x"), 5), rolling_max(terminal("x"), 5)])
    gpu_backend.score_rank_ic(plan, panel, target)
    cells, roots = 3 * 257, len(plan.root_ids)
    expected = (
        input_payload_bytes(len(panel.fields), 3, 257)
        + plan.slot_count * cells * 4
        + roots * 8
        + roots * 3 * METRIC_COUNT * 8
    )
    assert gpu_backend.last_run_stats["counter_summary"]["owned_device_bytes"] == expected


@pytest.mark.parametrize("mean_first", [False, True])
def test_temporal_operators_reuse_two_prefix_buffers(gpu_backend, mean_first):
    """极值与均值交替执行，临时区覆盖不得污染下一算子的有效性信息。"""
    rng = np.random.default_rng(25)
    x = rng.integers(-7, 8, (1, 3, 257)).astype(np.float32)
    x[0, 0, 31] = np.nan
    x[0, 1] = np.where(np.arange(257) % 2, -0.0, 0.0).astype(np.float32)
    panel = Panel({"x": x})
    target = rng.normal(size=panel.shape).astype(np.float32)
    first_op, second_op = (rolling_mean, rolling_min) if mean_first else (rolling_min, rolling_mean)
    first = first_op(terminal("x"), 60)
    plan = compile_expressions([first, second_op(first, 40)])
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(plan, panel, target)
    np.testing.assert_array_equal(gpu_backend.score_rank_ic(plan, panel, target), expected)
    cells, roots = 3 * 257, len(plan.root_ids)
    expected_bytes = (
        input_payload_bytes(1, 3, 257)
        + plan.slot_count * cells * 4
        + roots * 8
        + roots * 3 * METRIC_COUNT * 8
        + cells * (np.dtype(np.float64).itemsize + np.dtype(np.uint64).itemsize)
    )
    assert gpu_backend.last_run_stats["counter_summary"]["owned_device_bytes"] == expected_bytes
    # 导出会额外分配，必须在预算检查之后进行。
    with closing(NativePlanBackend()) as cpu:
        expected_values = cpu.evaluate(plan, panel)
    _assert_extrema_bits(export_values(gpu_backend, plan, panel), expected_values)


@pytest.mark.parametrize("lanes", [3, 38])
@pytest.mark.parametrize("window", [1, 2, 5, 32, 60, 120, 300, 900, 1300])
def test_cuda_extrema_preserves_invalid_windows_epoch_edges_and_tail(gpu_backend, lanes, window):
    rng = np.random.default_rng(0x13528A6)
    x = rng.integers(-4, 5, (1, lanes, 1027)).astype(np.float32)
    x[0, 0] = 7.0
    x[0, 1] = np.where(np.arange(x.shape[-1]) % 2, -0.0, 0.0).astype(np.float32)
    x[0, 2, 31] = np.nan
    x[0, 2, 256] = np.inf
    x[0, 2, 511] = -np.inf
    valid = np.ones(x.shape, dtype=bool)
    valid[0, -1, 777] = False
    panel = Panel({"x": x}, valid)
    plan = compile_expressions(
        [rolling_min(terminal("x"), window), rolling_max(terminal("x"), window)]
    )
    with closing(NativePlanBackend()) as cpu:
        _assert_extrema_bits(export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel))


@pytest.mark.parametrize("window", [32, 257, 900])
def test_cuda_extrema_large_window_recovers_after_invalid_value_leaves(gpu_backend, window):
    time = 2 * window + 17
    x = np.tile(np.arange(time, dtype=np.float32), (1, 3, 1))
    x[0, 0, window - 1] = np.nan
    x[0, 1, window] = np.inf
    valid = np.ones(x.shape, dtype=bool)
    valid[0, 2, window + 1] = False
    panel = Panel({"x": x}, valid)
    plan = compile_expressions(
        [rolling_min(terminal("x"), window), rolling_max(terminal("x"), window)]
    )
    with closing(NativePlanBackend()) as cpu:
        actual = export_values(gpu_backend, plan, panel)
        _assert_extrema_bits(actual, cpu.evaluate(plan, panel))
    assert np.isfinite(actual[..., -1]).all()
