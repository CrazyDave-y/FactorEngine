"""用独立参考核对多期限设备输出和现有 H30。"""

import os

import numpy as np
import pytest
from factor_engine.cuda_backend import CUDABackend
from factor_engine.data import Panel
from factor_engine.expression import terminal
from factor_engine.ir import compile_expressions
from factor_engine.metric_types import METRIC_COUNT, Metric
from factor_engine.runtime import ROOT
from factor_engine.session_source import ArraySource
from factor_engine.target_bank import MultiTargetSource, TargetBatch

from tests.reference.metrics import reference_multitarget
from tests.reference.metric_checks import FP32_PEARSON_ATOL, FP32_PEARSON_RTOL

CUDA_LIBRARY = (
    ROOT
    / os.environ.get("FACTORENGINE_TEST_CUDA_BUILD_DIR", "build-study")
    / "libfactor_engine_cuda.so"
)


def case(time=97, lanes=5, horizons=(5, 30, 300)):
    rng = np.random.default_rng(42)
    roots = rng.integers(-4, 5, (3, lanes, time)).astype(np.float32)
    roots[0, 0, ::7] = np.nan
    roots[1, 1] = 3
    roots[2, -1] = np.inf
    roots[0, 2, :2] = -0.0
    targets = rng.integers(-3, 4, (len(horizons), lanes, time)).astype(np.float32)
    eligible = rng.random(targets.shape) > 0.05
    targets[0, 0, ::11] = np.nan
    for h, horizon in enumerate(horizons):
        eligible[h, :, -min(time, horizon) :] = False
    names = tuple(f"x{i}" for i in range(len(roots)))
    panel = Panel({name: roots[i][None] for i, name in enumerate(names)})
    source = ArraySource(panel, targets[0][None], source_identity="metric-test")
    multi = MultiTargetSource(source, TargetBatch(horizons, targets, eligible))
    plan = compile_expressions([terminal(name) for name in names])
    return roots, multi, plan


def test_reused_workspace_uploads_new_target_values():
    """形状兼容只复用空间；换标签后仍须上传，不能复用旧指标。"""
    require_gpu()
    from tests.reference.metric_checks import assert_metric_parity, assert_stats_parity

    _, source, plan = case(horizons=(5, 30))
    changed = MultiTargetSource(
        source.source,
        TargetBatch(
            source.targets.horizons,
            -source.targets.values,
            source.targets.eligible,
            source.targets.mature,
        ),
    )
    library = CUDA_LIBRARY
    backend = CUDABackend(library, batch_lanes=3, memory_budget_bytes=1024**3)
    fresh = CUDABackend(library, batch_lanes=3, memory_budget_bytes=1024**3)
    try:
        backend.prepare_multitarget(plan, source)
        owners = dict(backend._metric_buffers)
        original = backend.score_multitarget(plan, source)
        actual = backend.score_multitarget(plan, changed, detail=True)
        assert backend._metric_buffers == owners

        fresh.prepare_multitarget(plan, changed)
        expected = fresh.score_multitarget(plan, changed, detail=True)
        assert_metric_parity(backend.last_metric_details, fresh.last_metric_details)
        assert_stats_parity(actual, expected)
        before = original.means[..., Metric.RANK_IC]
        after = actual.means[..., Metric.RANK_IC]
        valid = np.isfinite(before) & np.isfinite(after)
        assert np.any(before[valid] != after[valid])
    finally:
        backend.close()
        fresh.close()


def require_gpu():
    if os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("real GPU tests enabled explicitly")


@pytest.mark.parametrize("lanes", [3, 5, 8])
def test_search_input_session_rebind_growth_tail_and_generic_refresh(lanes):
    require_gpu()
    library = CUDA_LIBRARY
    _, source, _ = case(time=97, lanes=lanes, horizons=(5, 30))
    backend = CUDABackend(library, batch_lanes=3, memory_budget_bytes=1024**3)
    fresh = CUDABackend(library, batch_lanes=3, memory_budget_bytes=1024**3)
    try:
        with backend.multitarget_input_session(source) as info:
            for index, roots in enumerate((3, 7, 1, 3)):
                plan = compile_expressions([terminal(f"x{i % 3}") for i in range(roots)])
                backend.prepare_multitarget(plan, source)
                if index == 0:
                    assert all(
                        owner.upload_bytes == 0 for owner in backend._metric_buffers.values()
                    )
                actual = backend.score_multitarget(plan, source)
                fresh.prepare_multitarget(plan, source)
                expected = fresh.score_multitarget(plan, source)
                np.testing.assert_array_equal(actual.stats, expected.stats)
            batches = (lanes + 2) // 3
            assert info["input_uploads"] == batches
            assert info["target_uploads"] == batches * 2
            assert backend.multitarget_resources()["owned_device_bytes"] <= 1024**3
        # 同形状的新标签在普通 API 和下一次搜索中都必须刷新。
        changed = MultiTargetSource(
            source.source,
            TargetBatch(source.targets.horizons, -source.targets.values, source.targets.eligible),
        )
        backend.prepare_multitarget(plan, changed)
        actual = backend.score_multitarget(plan, changed)
        fresh.prepare_multitarget(plan, changed)
        expected = fresh.score_multitarget(plan, changed)
        np.testing.assert_array_equal(actual.stats, expected.stats)
        with backend.multitarget_input_session(changed) as again:
            backend.prepare_multitarget(plan, changed)
            actual = backend.score_multitarget(plan, changed)
            np.testing.assert_array_equal(actual.stats, expected.stats)
            assert again["input_uploads"] == batches
    finally:
        backend.close()
        fresh.close()


@pytest.mark.parametrize("time", [3, 97, 257, 1025, 3600, 3840, 4097, 23400])
def test_multitarget_device_oracle(time, record_property):
    require_gpu()
    factors, source, plan = case(time)
    expected = reference_multitarget(factors, source.targets)
    backend = CUDABackend(CUDA_LIBRARY, batch_lanes=3, memory_budget_bytes=1024**3)
    try:
        backend.prepare_multitarget(plan, source)
        stats = backend.score_multitarget(plan, source, detail=True)
        actual = backend.last_metric_details
        assert actual is not None
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        for metric in Metric:
            if metric == Metric.PEARSON:
                error = np.abs(actual[..., metric] - expected[..., metric])
                record_property("pearson_max_abs", float(error[np.isfinite(error)].max(initial=0)))
                finite = np.isfinite(expected[..., metric])
                assert np.isfinite(actual[..., metric][finite]).all()
                assert np.all(np.abs(actual[..., metric][finite]) <= 1 + 1e-10)
                np.testing.assert_allclose(
                    actual[..., metric],
                    expected[..., metric],
                    rtol=FP32_PEARSON_RTOL,
                    atol=FP32_PEARSON_ATOL,
                    equal_nan=True,
                )
            elif metric == Metric.RANK_IC:
                # NumPy参考分开开方，旧R2对平方和乘积开方；独立公式可差1–2 ULP。
                np.testing.assert_allclose(
                    actual[..., metric],
                    expected[..., metric],
                    rtol=2e-15,
                    atol=2e-15,
                    equal_nan=True,
                )
            else:
                np.testing.assert_array_equal(actual[..., metric], expected[..., metric])
        assert actual.shape == (3, 5, 3, METRIC_COUNT)
        assert stats.stats.shape == (3, 3, METRIC_COUNT, 2)
        again = backend.score_multitarget(plan, source)
        np.testing.assert_array_equal(stats.stats, again.stats)
        assert backend.last_metric_details is None
    finally:
        backend.close()


@pytest.mark.parametrize("time", [3600, 3841, 23400])
def test_fp32_pearson_high_bias_and_sparse_pairs(time):
    """FP32 Pearson gate 对照独立 FP64 oracle；不要求两者同精度。"""
    require_gpu()
    rng = np.random.default_rng(20261120)
    factors = rng.standard_normal((1, 8, time), dtype=np.float32)
    values = np.float32(0.65) * factors + np.float32(0.3) * rng.standard_normal(
        factors.shape, dtype=np.float32
    )
    offsets = rng.integers(-3, 4, (2, time)).astype(np.float32)
    noise = rng.integers(-2, 3, (2, time)).astype(np.float32)
    for lane, origin in enumerate((1.0, 1e20)):
        base = np.float32(origin)
        step = np.spacing(base)
        factors[0, lane] = base + step * offsets[lane]
        values[0, lane] = base + step * (2 * offsets[lane] + noise[lane])
    eligible = np.ones(values.shape, dtype=bool)
    eligible[:, :3, ::7] = False
    values[0, 0, ::29] = np.nan
    factors[0, 1, ::31] = np.inf
    factors[0, 3] = values[0, 3] = np.nan
    points = [0, time // 2, time - 1]
    factors[0, 3, points] = values[0, 3, points] = [-1, 0, 1]
    factors[0, 4] = 1
    values[0, 5] = 1
    eligible[:, 6] = False
    eligible[:, 7] = False
    eligible[:, 7, :2] = True
    targets = TargetBatch((1,), values, eligible)
    base = ArraySource(
        Panel({"x": factors[0][None]}), values[0][None], source_identity=f"pearson-bias-{time}"
    )
    source = MultiTargetSource(base, targets)
    plan = compile_expressions([terminal("x")])
    expected = reference_multitarget(factors, targets)
    assert np.isfinite(expected[0, :4]).all()
    assert np.isnan(expected[0, 4:]).all()
    backend = CUDABackend(CUDA_LIBRARY, batch_lanes=3, memory_budget_bytes=1024**3)
    try:
        backend.prepare_multitarget(plan, source)
        stats = backend.score_multitarget(plan, source, detail=True)
        actual = backend.last_metric_details
        assert actual is not None
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        assert np.isfinite(actual[np.isfinite(expected)]).all()
        assert np.all(np.abs(actual[np.isfinite(actual)]) <= 1 + 1e-10)
        # Pearson 输出来自 FP32 中心矩；这是真正的精度 gate，不能只记录误差。
        np.testing.assert_allclose(
            actual[..., Metric.PEARSON],
            expected[..., Metric.PEARSON],
            rtol=FP32_PEARSON_RTOL,
            atol=FP32_PEARSON_ATOL,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual[..., Metric.RANK_IC],
            expected[..., Metric.RANK_IC],
            rtol=2e-15,
            atol=2e-15,
            equal_nan=True,
        )
        np.testing.assert_array_equal(stats.stats[..., 1], np.full((1, 1, METRIC_COUNT), 4))
    finally:
        backend.close()


def test_h30_matches_single_target_rankic():
    require_gpu()
    factors, source, plan = case(97, 5, (30,))
    # 单目标调用使用 target 的 NaN 表达多期限 eligibility。
    old_target = np.where(source.targets.eligible[0], source.targets.values[0], np.nan)
    assert isinstance(source.source, ArraySource)
    single = ArraySource(source.source.panel, old_target[None], source_identity="old-h30")
    backend = CUDABackend(CUDA_LIBRARY, batch_lanes=3, memory_budget_bytes=1024**3)
    try:
        expected = backend.score(plan, single)
        backend.prepare_multitarget(plan, source)
        result = backend.score_multitarget(plan, source, detail=True)
        np.testing.assert_array_equal(result.stats[:, 0, Metric.RANK_IC], expected.stats)
    finally:
        backend.close()


def test_multitarget_budget_and_closed():
    require_gpu()
    _, source, plan = case()
    backend = CUDABackend(CUDA_LIBRARY, memory_budget_bytes=1024**2)
    try:
        backend.prepare_multitarget(plan, source)
        assert backend.multitarget_resources()["owned_device_bytes"] > 0
        owner = next(iter(backend._metric_buffers.values()))
        owner.budget = owner.payload_bytes
        with pytest.raises(MemoryError):
            backend.prepare_multitarget(plan, source)
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.score_multitarget(plan, source)


def test_scratch_growth_counts_old_and_new_allocation():
    require_gpu()
    from factor_engine.cuda_buffers import CUDABuffers

    _, source, plan = case(time=4097, horizons=(1, 2, 3))
    backend = CUDABackend(CUDA_LIBRARY, batch_lanes=3, memory_budget_bytes=1024**3)
    try:
        small = compile_expressions([terminal("x0")])
        backend.prepare_multitarget(small, source)
        owner = backend._metric_buffers[3]
        resources = backend.multitarget_resources()["batches"][3]
        old_scratch = resources["requested_scratch_bytes"]
        new_scratch = old_scratch * len(plan.root_ids)
        storage = owner.native.requirement(owner.handle, plan)
        # 只够新 scratch，容不下 prepare 期间仍保留的旧 scratch 和存储。
        owner.budget = owner.payload_bytes + storage + new_scratch
        owner.rebind_plan(plan)
        with pytest.raises(MemoryError, match="workspace"):
            CUDABuffers.prepare_metrics(owner, plan, source.targets.horizons)
    finally:
        backend.close()
