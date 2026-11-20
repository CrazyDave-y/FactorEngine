"""核对 Generic CUDA 的算子值、并列秩和逐 lane 评分。"""

from contextlib import closing

import numpy as np
import pytest
from factor_engine.cuda_buffers import CUDABuffers
from factor_engine.data import Panel
from factor_engine.expression import (
    constant,
    corr,
    cov,
    delay,
    ema,
    make_expr,
    neg,
    rolling_std,
    signed_log1p,
    sub,
    terminal,
)
from factor_engine.ir import compile_expressions
from factor_engine.native_plan_backend import NativePlanBackend
from factor_engine.session_source import ArraySource

from tests.reference.cases import build_case
from tests.reference.fixtures import SyntheticLaneSource
from tests.reference.metric_checks import assert_same


@pytest.mark.parametrize("time", [257, 3841, 23400])
def test_delay_output_does_not_reapply_original_input_mask(gpu_backend, time):
    rng = np.random.default_rng(2207)
    x = rng.normal(size=(1, 2, time)).astype(np.float32)
    target = rng.normal(size=x.shape).astype(np.float32)
    valid = np.ones(x.shape, bool)
    valid[..., 5::13] = False
    panel = Panel({"x": x}, valid)
    plan = compile_expressions([delay(terminal("x"), 2)])
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(plan, panel, target)
    np.testing.assert_array_equal(gpu_backend.score_rank_ic(plan, panel, target), expected)


def export_values(backend, plan, panel):
    """通过有大小限制的结果导出接口取回 root，供逐项比较。"""
    source = ArraySource(panel, np.zeros(panel.shape, np.float32), source_identity="numeric-export")
    width = min(32, source.lane_count)
    output = np.empty((len(plan.root_ids), source.lane_count, source.time), np.float32)
    with closing(
        CUDABuffers(
            backend.library,
            field_names=source.field_names,
            capacity_lanes=width,
            time_length=source.time,
            plan=plan,
            budget_bytes=backend.memory_budget_bytes,
        )
    ) as owner:
        for batch in source.iter_batches(0, width):
            count = batch.panel.shape[1]
            owner._upload(0, batch.panel, batch.target, final=True)
            values = owner.evaluate(0, count, plan).reshape(len(plan.root_ids), count, source.time)
            output[:, batch.first_lane : batch.first_lane + count] = values
    return output.reshape((len(plan.root_ids), *panel.shape))


@pytest.mark.parametrize("case", ["normal", "edge", "short", "long_window"])
def test_every_current_operator_matches_native_cpu_values_and_scores(gpu_backend, case):
    case = build_case(case)
    plan = compile_expressions([entry.expression for entry in case.expressions])
    with closing(NativePlanBackend()) as cpu:
        expected = np.asarray(cpu.evaluate(plan, case.panel))
        actual = export_values(gpu_backend, plan, case.panel)
        np.testing.assert_array_equal(actual, expected)
        target = case.panel.fields["y"]
        np.testing.assert_array_equal(
            gpu_backend.score_rank_ic(plan, case.panel, target),
            cpu.score_rank_ic(plan, case.panel, target),
        )


@pytest.mark.parametrize("lanes", [3, 38])
@pytest.mark.parametrize("window", [2, 32, 120, 200])
def test_centered_pair_preserves_constants_invalids_and_tail(gpu_backend, lanes, window):
    rng = np.random.default_rng(1831)
    a = rng.integers(-3, 4, (1, lanes, 137)).astype(np.float32)
    b = rng.integers(-2, 3, a.shape).astype(np.float32)
    a[:, ::7] = 1
    b[:, ::11] = 2
    a[0, -1, 71] = np.nan
    b[0, -2, 97] = np.inf
    panel = Panel({"a": a, "b": b})
    plan = compile_expressions(
        [corr(terminal("a"), terminal("b"), window), cov(terminal("a"), terminal("b"), window)]
    )
    with closing(NativePlanBackend()) as cpu:
        np.testing.assert_array_equal(
            export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel)
        )


@pytest.mark.parametrize("span", [1, 5, 60])
def test_ema_and_centered_std_keep_native_order_bits(gpu_backend, span):
    rng = np.random.default_rng(317)
    x = np.float32(1e8) + 8 * rng.integers(-3, 4, (1, 38, 512)).astype(np.float32)
    x[0, 1] = np.float32(1e8)
    x[0, 2, 79] = np.nan
    x[0, 3, ::83] = np.inf
    panel = Panel({"x": x})
    plan = compile_expressions([ema(terminal("x"), span), rolling_std(terminal("x"), 120)])
    with closing(NativePlanBackend()) as cpu:
        np.testing.assert_array_equal(
            export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel)
        )


@pytest.mark.parametrize("length", [1, 255, 256, 257, 513, 23400])
def test_ema_tile_edges_keep_signed_zero_and_invalid_state_bits(gpu_backend, length):
    rng = np.random.default_rng(811)
    x = rng.normal(size=(1, 3, length)).astype(np.float32)
    x[0, 0] = np.float32(-0.0)
    x[0, 1, 0] = -np.finfo(np.float32).max
    for t in (0, 254, 255, 256, 511):
        if t < length:
            x[0, 2, t] = np.nan if t % 2 else np.inf
    panel = Panel({"x": x})
    plan = compile_expressions([ema(terminal("x"), 60)])
    if length < 3:
        # 评分 Buffer 的现有合同要求至少 3 点，不能为单步算子测试放宽它。
        with pytest.raises(ValueError, match="at least three"):
            export_values(gpu_backend, plan, panel)
        return
    with closing(NativePlanBackend()) as cpu:
        expected = np.asarray(cpu.evaluate(plan, panel))
        actual = export_values(gpu_backend, plan, panel)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(
            actual[finite].view(np.uint32), expected[finite].view(np.uint32)
        )


def test_signed_log_keeps_the_rank_tie_counterexample(gpu_backend):
    panel, target = SyntheticLaneSource(1, 400, 390, 0x135288F).tile(0, 400)
    v = terminal("volume_change")
    plan = compile_expressions([sub(signed_log1p(v), v), sub(neg(v), v)])
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(plan, panel, target)
        actual = gpu_backend.score_rank_ic(plan, panel, target)
        np.testing.assert_array_equal(actual, expected)
        assert actual[0] == actual[1]


def test_scalar_subtrees_match_cpu_including_zero_denominator(gpu_backend):
    panel = Panel({"x": np.arange(64, dtype=np.float32).reshape(1, 2, 32)})
    one, zero = constant(1), constant(0)
    invalid = make_expr("safe_div", one, zero)
    scalar_chain = make_expr("signed_log1p", make_expr("add", one, one))
    plan = compile_expressions(
        [make_expr("add", terminal("x"), invalid), make_expr("add", terminal("x"), scalar_chain)],
        canonicalize_exprs=False,
    )
    with closing(NativePlanBackend()) as cpu:
        np.testing.assert_array_equal(
            export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel)
        )
        np.testing.assert_array_equal(
            gpu_backend.score_rank_ic(plan, panel, panel.fields["x"]),
            cpu.score_rank_ic(plan, panel, panel.fields["x"]),
        )


def test_scalar_constant_folding_matches_native_execution(gpu_backend):
    panel = Panel({"x": np.arange(64, dtype=np.float32).reshape(1, 2, 32)})
    big, one = constant(16777216), constant(1)
    scalars = [
        make_expr("sub", make_expr("add", big, one), big),
        make_expr("safe_div", one, constant(1.00000001e-12)),
        make_expr("signed_log1p", constant(0.25)),
        make_expr("sign", constant(-2)),
    ]
    expressions = [make_expr("add", terminal("x"), value) for value in scalars]
    raw = compile_expressions(expressions, canonicalize_exprs=False)
    folded = compile_expressions(expressions, canonicalize_exprs=True)
    with closing(NativePlanBackend()) as cpu:
        expected = np.asarray(cpu.evaluate(raw, panel))
        np.testing.assert_array_equal(cpu.evaluate(folded, panel), expected)
        np.testing.assert_array_equal(export_values(gpu_backend, raw, panel), expected)
        np.testing.assert_array_equal(export_values(gpu_backend, folded, panel), expected)


def test_device_budget_rejection_does_not_publish_a_previous_score(gpu_backend):
    panel = Panel({"x": np.arange(32 * 128, dtype=np.float32).reshape(1, 32, 128)})
    plan = compile_expressions([terminal("x")])
    gpu_backend.score_rank_ic(plan, panel, panel.fields["x"])
    gpu_backend.memory_budget_bytes = 1
    with pytest.raises(MemoryError, match="budget"):
        gpu_backend.score_rank_ic(plan, panel, panel.fields["x"])
    assert gpu_backend.last_run_stats is None


def test_scoring_returns_lane_ic_without_exporting_unused_root_stats(gpu_backend):
    rng = np.random.default_rng(816)
    x = rng.integers(-3, 4, (1, 38, 257)).astype(np.float32)
    target = rng.integers(-5, 6, x.shape).astype(np.float32)
    x[0, 1, 10] = np.nan
    target[0, 2, 13] = np.inf
    valid = np.ones(x.shape, dtype=bool)
    valid[0, 3, 17] = False
    panel = Panel({"x": x}, valid)
    plan = compile_expressions([terminal("x"), neg(terminal("x"))])
    with closing(NativePlanBackend()) as cpu:
        np.testing.assert_array_equal(
            gpu_backend.score_rank_ic(plan, panel, target), cpu.score_rank_ic(plan, panel, target)
        )
    counters = gpu_backend.last_run_stats["counter_summary"]
    assert gpu_backend.last_run_stats["batches"] == 2
    assert counters["root_export_bytes"] == 0
    assert counters["lane_ic_d2h_bytes"] == 2 * 38 * 8


def test_rank_input_fusion_keeps_ties_and_missing_values(gpu_backend):
    rng = np.random.default_rng(913)
    x = rng.integers(-3, 4, (1, 3, 257)).astype(np.float32)
    target = rng.integers(-5, 6, x.shape).astype(np.float32)
    x[0, 0, 7] = np.nan
    x[0, 1, 11] = np.inf
    target[0, 0, 13] = np.nan
    valid = np.ones(x.shape, dtype=bool)
    valid[0, 2, 17] = False
    panel = Panel({"x": x}, valid)
    plan = compile_expressions([terminal("x")])
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(plan, panel, target)
        actual = gpu_backend.score_rank_ic(plan, panel, target)
    np.testing.assert_array_equal(actual, expected)
    counters = gpu_backend.last_run_stats["counter_summary"]
    # 单 terminal 没有 DAG kernel；两次排序、ties、归约均在同一 kernel。
    assert counters["kernel_launches"] == 1


@pytest.mark.parametrize("length", [3, 31, 32, 33, 127, 128, 129, 511, 512, 1025, 23401])
def test_temporal_boundaries_preserve_native_bits(gpu_backend, length):
    rng = np.random.default_rng(731)
    shape = (1, 3, length)
    x = rng.integers(-16, 17, shape).astype(np.float32) / 8
    y = rng.integers(-12, 13, shape).astype(np.float32) / 4
    x[0, 0] = np.float32(-0.0)
    y[0, 0] = np.float32(0.0)
    x[0, 1] = np.nan
    first = min(256, length - 1)
    x[0, 1, first] = np.finfo(np.float32).max
    for index in (4, 5, 31, 32, 59, 60, 127, 128, 255, 256, 299, 300, 511, 512, 899, 900):
        if index < length:
            x[0, 2, index] = np.nan if index % 2 else np.inf
            y[0, 2, index] = -np.inf if index % 3 else 0
    panel = Panel({"x": x, "y": y})
    a, b = terminal("x"), terminal("y")
    expressions = [ema(a, span) for span in (300, 900)]
    for window in (2, 5, 60, 300, 900):
        expressions.extend((cov(a, b, window), corr(a, b, window)))
    plan = compile_expressions(expressions)
    with closing(NativePlanBackend()) as cpu:
        expected = np.asarray(cpu.evaluate(plan, panel))
    actual = export_values(gpu_backend, plan, panel)
    assert_same(actual, expected)
