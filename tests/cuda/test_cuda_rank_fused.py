"""单 block RankIC 的规模边界、并列值和联合 mask。"""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from factor_engine.cuda_native import input_payload_bytes
from factor_engine.metric_types import METRIC_COUNT, NativeLimit
from factor_engine.data import Panel
from factor_engine.expression import terminal
from factor_engine.ir import compile_expressions
from factor_engine.native_plan_backend import NativePlanBackend


@pytest.mark.parametrize(
    "time", [3, 31, 257, 1024, 1025, 4096, 4097, 16384, 16385, 23400, 24576, 24577]
)
def test_rank_sort_sizes_match_cpu(gpu_backend, time):
    rng = np.random.default_rng(401)
    x = rng.integers(-7, 8, (1, 4, time)).astype(np.float32)
    y = rng.integers(-5, 6, x.shape).astype(np.float32)
    x[0, 0, ::31] = np.nan
    y[0, 0, ::37] = np.inf
    x[0, 1] = 3
    x[0, 2] = np.nan
    x[0, 2, :2] = (1, 2)
    x[0, 3, ::7] = -0.0
    y[0, 3, ::11] = 0.0
    panel = Panel({"x": x})
    plan = compile_expressions([terminal("x")])
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(plan, panel, y)
        actual = gpu_backend.score_rank_ic(plan, panel, y)
    np.testing.assert_array_equal(actual, expected)
    launches = gpu_backend.last_run_stats["counter_summary"]["kernel_launches"]
    if time <= 4096:
        assert launches == 1
    elif time > 24576:
        assert launches == 6
    else:
        # 大 block 的共享内存配额取决于 GPU；两条规模策略都要保持数值一致。
        assert launches in (1, 6)


def test_rank_capacity_reuse_with_fewer_roots(gpu_backend):
    from factor_engine.cuda_buffers import CUDABuffers
    from factor_engine.session_source import ArraySource

    rng = np.random.default_rng(17)
    shape = (1, 4, 257)
    panel = Panel({name: rng.normal(size=shape).astype(np.float32) for name in ("x", "y")})
    target = rng.normal(size=shape).astype(np.float32)
    source = ArraySource(panel, target, source_identity="rank-capacity-reuse")
    panel, target = source.panel, source.target
    large = compile_expressions([terminal("x"), terminal("y")])
    small = compile_expressions([terminal("y")])
    with closing(
        CUDABuffers(
            gpu_backend.library,
            field_names=tuple(panel.fields),
            capacity_lanes=4,
            time_length=257,
            plan=large,
        )
    ) as owner:
        owner._upload(0, panel, target, final=True)
        handle = owner._prepare(4, large)
        owner.native.score_lanes(handle, 0, 4, large)
        result, counters = owner.native.score_lanes(handle, 0, 4, small)
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(small, panel, target)
    np.testing.assert_allclose(result.mean(axis=1), expected, rtol=0, atol=1e-16)
    assert counters["lane_ic_d2h_bytes"] == 4 * 8


def test_fused_rank_scratch_is_only_digitpair_and_metric_output(gpu_backend):
    rng = np.random.default_rng(910)
    shape = (1, 3, 23400)
    fields = {name: rng.normal(size=shape).astype(np.float32) for name in ("x", "y")}
    target = rng.normal(size=shape).astype(np.float32)
    panel = Panel(fields)
    plan = compile_expressions([terminal(name) for name in fields])
    with closing(NativePlanBackend()) as cpu:
        expected = cpu.score_rank_ic(plan, panel, target)
    actual = gpu_backend.score_rank_ic(plan, panel, target)
    np.testing.assert_array_equal(actual, expected)
    counters = gpu_backend.last_run_stats["counter_summary"]
    # 融合路径保留双 IC与 D23，不申请完整 Key/RankX 数组。
    if counters["kernel_launches"] != 1:
        pytest.skip("this device uses segmented sorting at time=23400")
    input_bytes = input_payload_bytes(len(fields), shape[1], shape[2])
    segments = len(fields) * shape[1]
    expected_workspace = segments * (
        METRIC_COUNT * np.dtype(np.float64).itemsize
        + NativeLimit.CUDA_LONG_CAPACITY * np.dtype(np.uint16).itemsize
    )
    assert counters["rank_workspace_bytes"] == expected_workspace
    assert (
        counters["owned_device_bytes"]
        == input_bytes + expected_workspace + len(fields) * np.dtype(np.uintp).itemsize
    )


@pytest.mark.parametrize("time", [1025, 24577])
def test_score_export_score_preserves_roots_and_masks(gpu_backend, time):
    from factor_engine.cuda_buffers import CUDABuffers
    from factor_engine.session_source import ArraySource

    rng = np.random.default_rng(911)
    shape = (1, 3, time)
    fields = {name: rng.integers(-9, 10, shape).astype(np.float32) for name in ("x", "y")}
    fields["x"][..., ::13] = np.nan
    fields["y"][..., ::17] = np.nan
    target = rng.integers(-4, 5, shape).astype(np.float32)
    target[..., ::19] = np.nan
    source = ArraySource(Panel(fields), target, source_identity="rank-export-lifetime")
    plan = compile_expressions([terminal(name) for name in fields])
    with closing(
        CUDABuffers(
            gpu_backend.library,
            field_names=source.field_names,
            capacity_lanes=3,
            time_length=time,
            plan=plan,
        )
    ) as owner:
        owner._upload(0, source.panel, source.target, final=True)
        handle = owner._prepare(3, plan)
        planned_score = owner.native.requirement(handle, plan)
        planned_export = owner.native.requirement(handle, plan, for_export=True)
        extra = 2 * 3 * time * 4 if time <= 24576 else 0
        assert planned_export == planned_score + extra
        before = owner._score_window(0, 3, plan).copy()
        score_bytes = owner.native.requirement(handle, plan)
        export_bytes = owner.native.requirement(handle, plan, for_export=True)
        assert export_bytes == score_bytes + extra
        exported = owner.evaluate(0, 3, plan)
        assert exported.shape == (2, 1, 3, time)
        for row, name in zip(exported, fields, strict=True):
            np.testing.assert_array_equal(row, source.panel.fields[name])
        after = owner._score_window(0, 3, plan)
        np.testing.assert_array_equal(after, before)


def test_export_checks_budget_after_scoring_prepared_workspace(gpu_backend):
    from factor_engine.cuda_buffers import CUDABuffers
    from factor_engine.session_source import ArraySource

    rng = np.random.default_rng(912)
    shape = (1, 2, 257)
    panel = Panel({name: rng.normal(size=shape).astype(np.float32) for name in ("x", "y")})
    target = rng.normal(size=shape).astype(np.float32)
    source = ArraySource(panel, target, source_identity="rank-export-budget")
    plan = compile_expressions([terminal("x"), terminal("y")])
    payload = input_payload_bytes(len(panel.fields), shape[1], shape[2])
    # 两个 root 指针与双 IC输出恰好够评分，但不够导出两条完整因子。
    budget = (
        payload
        + len(plan.root_ids) * np.dtype(np.uintp).itemsize
        + len(plan.root_ids) * shape[1] * METRIC_COUNT * np.dtype(np.float64).itemsize
    )
    with closing(
        CUDABuffers(
            gpu_backend.library,
            field_names=source.field_names,
            capacity_lanes=2,
            time_length=257,
            plan=plan,
            budget_bytes=budget,
        )
    ) as owner:
        owner._upload(0, source.panel, source.target, final=True)
        before = owner._score_window(0, 2, plan).copy()
        with pytest.raises(MemoryError, match="factor output workspace"):
            owner.evaluate(0, 2, plan)
        np.testing.assert_array_equal(owner._score_window(0, 2, plan), before)


@pytest.mark.parametrize("score_first", [False, True])
def test_tail_subset_export_checks_retained_capacity_budget(gpu_backend, score_first):
    from factor_engine.cuda_buffers import CUDABuffers
    from factor_engine.session_source import ArraySource

    rng = np.random.default_rng(913)
    shape = (1, 3, 257)
    panel = Panel(
        {name: rng.normal(size=shape).astype(np.float32) for name in ("a", "b", "c", "d")}
    )
    target = rng.normal(size=shape).astype(np.float32)
    source = ArraySource(panel, target, source_identity="rank-subset-export-budget")
    plan = compile_expressions([terminal(name) for name in panel.fields])
    subset = replace(plan, root_ids=plan.root_ids[:2])
    budget = 21 * 3 * 257 + 4160
    with closing(
        CUDABuffers(
            gpu_backend.library,
            field_names=source.field_names,
            capacity_lanes=3,
            time_length=257,
            plan=plan,
            budget_bytes=budget,
        )
    ) as owner:
        owner._upload(0, source.panel, source.target, final=True)
        if score_first:
            owner._score_window(0, 3, plan)
        # 尾批现在保留三条 Lane 的实际容量，导出预算应更早拒绝；
        # 不能为成功导出而假装已经释放了第三条 Lane 的空间。
        before = owner._score_window(0, 2, plan).copy()
        with pytest.raises(MemoryError, match="factor output workspace exceeds memory budget"):
            owner.evaluate(0, 2, subset)
        np.testing.assert_array_equal(owner._score_window(0, 2, plan), before)
