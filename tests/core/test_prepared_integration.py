"""跨完整批和尾批复用计划，同时检查结果直接随调用返回。"""

from contextlib import closing
import numpy as np

from factor_engine.cpu_backend import CPUBackend
from factor_engine.data import Panel
from factor_engine.engine import FactorEngine, SearchConfig
from factor_engine.expression import terminal, rolling_mean
from factor_engine.ir import compile_expressions
from factor_engine.runtime import ROOT
from factor_engine.session_source import ArraySource


def source():
    rng = np.random.default_rng(921)
    x = rng.normal(size=(1, 7, 257)).astype(np.float32)
    x[..., ::31] = np.nan
    y = rng.normal(size=x.shape).astype(np.float32)
    return ArraySource(Panel({"x": x}), y, source_identity="prepared-tail-regression")


def test_cpu_encodes_once_across_batches_and_invalidates_new_plan():
    data = source()
    plan = compile_expressions([terminal("x"), rolling_mean(terminal("x"), 5)])
    library = ROOT / "build-study/libfactor_engine_plan.so"
    with closing(CPUBackend(library, batch_lanes=3)) as cpu:
        first = cpu.score(plan, data)
        assert first.run_stats["batches"] == 3
        assert cpu.backend.plans.preparations == 1
        second = cpu.score(plan, data)
        assert cpu.backend.plans.preparations == 1
        np.testing.assert_array_equal(first.stats.view(np.uint64), second.stats.view(np.uint64))
        changed = compile_expressions([rolling_mean(terminal("x"), 11)])
        assert cpu.score(changed, data).stats.shape == (1, 2)
        assert cpu.backend.plans.preparations == 2


def test_gpu_prepared_plan_covers_tail_and_matches_cpu_totals(gpu_backend):
    from factor_engine.cuda_buffers import CUDABuffers

    data = source()
    plan = compile_expressions([terminal("x"), rolling_mean(terminal("x"), 5)])
    with CUDABuffers(
        gpu_backend.library,
        field_names=data.field_names,
        capacity_lanes=3,
        time_length=data.time,
        plan=plan,
    ) as buffers:
        actual = buffers.score_stats(data, plan)
        assert buffers.native.plans.preparations == 1
        assert actual.run_stats["batches"] == 3
        assert "counters" not in actual.run_stats
    with closing(CPUBackend(ROOT / "build-study/libfactor_engine_plan.so", batch_lanes=3)) as cpu:
        expected = cpu.score(plan, data)
    np.testing.assert_array_equal(actual.stats.view(np.uint64), expected.stats.view(np.uint64))


def test_search_diagnostics_are_opt_in():
    data = source()
    config = SearchConfig(population_size=8, generations=2, elite_size=2, max_depth=2, seed=42)
    with closing(CPUBackend(ROOT / "build-study/libfactor_engine_plan.so", batch_lanes=3)) as cpu:
        engine = FactorEngine(plan_backend=cpu)
        first = engine.search_indexed(data, ["x"], config)
        assert engine.last_search_stats is not None
        assert engine.last_search_stats["generations"] == []
        assert engine.last_search_stats["execution"] == []
        second = engine.search_indexed(data, ["x"], config, capture_generations=True)
        assert first == second
        assert engine.last_search_stats is not None
        assert len(engine.last_search_stats["generations"]) == 3


def test_buffer_close_preserves_other_cuda_users_allocation(gpu_backend):
    import ctypes as C

    runtime = C.CDLL("libcudart.so.12")
    runtime.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
    runtime.cudaMalloc.restype = C.c_int
    runtime.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    runtime.cudaMemcpy.restype = C.c_int
    runtime.cudaFree.argtypes, runtime.cudaFree.restype = [C.c_void_p], C.c_int
    pointer = C.c_void_p()
    expected = np.array([1.0, -0.0, -3.0, 7.0], dtype=np.float32)
    assert runtime.cudaMalloc(C.byref(pointer), expected.nbytes) == 0
    try:
        assert runtime.cudaMemcpy(pointer, expected.ctypes.data, expected.nbytes, 1) == 0
        data = source()
        plan = compile_expressions([terminal("x")])
        gpu_backend.score(plan, data)
        actual = np.empty_like(expected)
        assert runtime.cudaMemcpy(actual.ctypes.data, pointer, actual.nbytes, 2) == 0
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    finally:
        assert runtime.cudaFree(pointer) == 0


def test_cpu_lane_export_matches_existing_totals():
    from factor_engine.expression import terminal, make_expr
    from factor_engine.ir import compile_expressions
    from factor_engine.native_plan_backend import NativePlanBackend
    from factor_engine.metrics import LaneAccumulator
    from factor_engine.synthetic import make_synthetic_panel

    panel, target = make_synthetic_panel(sessions=1, securities=3, time=64, seed=9)
    plan = compile_expressions(
        [
            terminal("return_1"),
            make_expr("roll_mean", terminal("return_1"), param=5),
        ]
    )
    with closing(NativePlanBackend()) as native:
        lanes = native.score_lanes(plan, panel, target)
        totals = native.score_rank_ic_stats(plan, panel, target)
        accumulator = LaneAccumulator(2, 3)
        accumulator.commit(0, lanes)
        np.testing.assert_array_equal(
            accumulator.finish().stats.view(np.uint64), totals.view(np.uint64)
        )
