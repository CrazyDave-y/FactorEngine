"""真实 cudaMalloc OOM 后缩小批次，仍完整保留逐 lane 数值。"""

from factor_engine.cuda_native import CUDANative


def test_real_cuda_oom_then_smaller_chunk_keeps_all_lane_stats(gpu_backend, monkeypatch):
    from contextlib import closing

    import numpy as np
    from factor_engine.cuda_backend import CUDABackend
    from factor_engine.cuda_native import input_payload_bytes
    from factor_engine.data import Panel
    from factor_engine.expression import terminal
    from factor_engine.ir import compile_expressions
    from factor_engine.metrics import LaneAccumulator
    from factor_engine.native_plan_backend import NativePlanBackend
    from factor_engine.session_source import ArraySource

    rng = np.random.default_rng(2011)
    fields = {name: rng.integers(-7, 8, (1, 65, 97)).astype(np.float32) for name in ("x", "y")}
    target = rng.integers(-4, 5, (1, 65, 97)).astype(np.float32)
    source = ArraySource(Panel(fields), target, source_identity="actual-oom-retry")
    plan = compile_expressions([terminal("x"), terminal("y")])
    with closing(NativePlanBackend()) as cpu:
        values = cpu.score_lanes(plan, source.panel, source.target)
    accumulator = LaneAccumulator(2, 65)
    accumulator.commit(0, values)
    expected = accumulator.finish()
    allocate = CUDANative.allocate
    attempts = []

    def fail_first(self, names, lanes, time_length, budget):
        attempts.append(lanes)
        if len(attempts) == 1:
            # 受控真实 cudaMalloc OOM：请求超过物理总量，不占住整卡或建立 Host 巨数组。
            total = self.device_info()["total_bytes"]
            per_lane = input_payload_bytes(len(names), 1, time_length)
            oversized = total // per_lane + 1
            return allocate(self, names, oversized, time_length, oversized * per_lane)
        return allocate(self, names, lanes, time_length, budget)

    monkeypatch.setattr(CUDANative, "allocate", fail_first)
    with closing(
        CUDABackend(
            gpu_backend.library,
            batch_lanes=None,
            memory_budget_bytes=128 * 1024**2,
        )
    ) as backend:
        result = backend.score(plan, source)
        np.testing.assert_array_equal(result.stats, expected.stats)
        assert attempts == [65, 32]
        assert result.run_stats["batch_capacity"] == 32
        assert len(result.run_stats["allocation_retries"]) == 1
        assert backend._single_buffer is not None
    assert backend._single_buffer is None
