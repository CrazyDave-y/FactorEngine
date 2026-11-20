"""直接验证生产 CPU 后端的完整 lane、确定性累计及失败恢复。"""

import os
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from factor_engine.cpu_backend import CPUBackend
from factor_engine.data import Panel
from factor_engine.expression import ema, rolling_mean, terminal
from factor_engine.ir import compile_expressions
from factor_engine.session_source import ArraySource

from tests.reference.metrics import mean_lane_rank_ic
from tests.reference.numpy_backend import NumpyPlanBackend

pytestmark = pytest.mark.skipif(
    not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="current native CPU build required"
)


def backend(width=2, **options):
    return CPUBackend(os.environ["FACTORENGINE_NATIVE_PLAN_LIB"], batch_lanes=width, **options)


def source():
    # 各 lane 的 Spearman 为 .1、.2、.1、.3、NaN；逐 lane FP64 累计为 .7。
    x = np.array(
        [[[0, 3, 4, 2, 1], [0, 3, 4, 1, 2], [0, 3, 4, 2, 1], [0, 2, 4, 3, 1], [2, 2, 2, 2, 2]]],
        np.float32,
    )
    target = np.broadcast_to(np.arange(5, dtype=np.float32), x.shape)
    return ArraySource(Panel({"x": x}), target, source_identity="cpu-streaming-contract")


@pytest.mark.parametrize("width", [1, 2, 4, 32])
def test_batch_width_and_repeated_runs_preserve_fp64_sum_order(width):
    plan = compile_expressions([terminal("x")])
    with closing(backend(width)) as cpu:
        first = cpu.score(plan, source())
        second = cpu.score(plan, source())
        np.testing.assert_array_equal(first.stats, [[0.7, 4.0]])
        np.testing.assert_array_equal(first.stats.view(np.uint64), second.stats.view(np.uint64))
        assert second.run_stats["lanes"] == 5
        assert second.run_stats["batches"] == (5 + width - 1) // width


def test_temporal_values_do_not_cross_session_or_lane_boundaries():
    rng = np.random.default_rng(201)
    x = rng.normal(size=(3, 5, 73)).astype(np.float32)
    y = rng.normal(size=x.shape).astype(np.float32)
    x[0, :3] = 4
    y[1, 1] = np.nan
    x[2, 4, :20] = np.inf
    panel = Panel({"x": x})
    plan = compile_expressions(
        [terminal("x"), ema(terminal("x"), 5), rolling_mean(terminal("x"), 9)]
    )
    expected = [mean_lane_rank_ic(value, y) for value in NumpyPlanBackend().evaluate(plan, panel)]
    with closing(backend(4)) as cpu:
        actual = cpu.score_rank_ic(plan, panel, y)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert cpu.last_run_stats is not None
        assert cpu.last_run_stats["batches"] == 4


class BatchSource:
    """保留生产只读输入，故障仅发生在批次边界。"""

    def __init__(self, transform):
        self.base = source()
        self.transform = transform
        self.closed = False
        self.field_names = self.base.field_names
        self.time = self.base.time
        self.lane_count = self.base.lane_count
        self.bytes_per_lane = self.base.bytes_per_lane

    def iter_batches(self, first, width):
        try:
            for batch in self.base.iter_batches(first, width):
                selected = self.transform(batch)
                if selected is not None:
                    yield selected
        finally:
            self.closed = True


@pytest.mark.parametrize("failure", ["reorder", "incomplete", "read_error", "bad_target"])
def test_bad_source_cannot_publish_partial_results_and_backend_can_retry(failure):
    def transform(batch):
        if failure == "reorder":
            return replace(batch, first_lane=batch.first_lane + 1)
        if failure == "incomplete" and batch.first_lane > 0:
            return None
        if failure == "read_error" and batch.first_lane > 0:
            raise OSError("source read failed")
        if failure == "bad_target":
            return replace(batch, target=batch.target[..., :-1])
        return batch

    bad = BatchSource(transform)
    plan = compile_expressions([rolling_mean(terminal("x"), 2)])
    with closing(backend()) as cpu:
        cpu.score(plan, source())
        with pytest.raises((ValueError, OSError)):
            cpu.score(plan, bad)
        assert bad.closed and cpu.last_run_stats is None
        assert cpu.backend.workspace_capacity_floats == 0
        recovered = cpu.score(plan, source())
        assert recovered.run_stats["lanes"] == 5


def test_budget_is_checked_before_reading_source():
    reads = []
    bad = BatchSource(lambda batch: reads.append(batch) or batch)
    with closing(backend(memory_budget_bytes=1)) as cpu:
        with pytest.raises(MemoryError, match="budget"):
            cpu.score(compile_expressions([terminal("x")]), bad)
    assert not reads


def test_failed_iterator_close_clears_run_stats_and_workspace():
    class CloseFailure(BatchSource):
        def iter_batches(self, first, width):
            yield from super().iter_batches(first, width)
            raise OSError("source close failed")

    with closing(backend()) as cpu:
        with pytest.raises(OSError, match="close failed"):
            cpu.score(
                compile_expressions([rolling_mean(terminal("x"), 2)]), CloseFailure(lambda b: b)
            )
        assert cpu.last_run_stats is None
        assert cpu.backend.workspace_capacity_floats == 0
