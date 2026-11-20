"""验证融合调度边界，以及真实 CUDA 的逐阶段数值和 launch 数。"""

import ctypes as C
import os
import shutil
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from factor_engine.cuda_buffers import CUDABuffers
from factor_engine.data import Panel
from factor_engine.expression import abs_, add, constant, make_expr, neg, terminal
from factor_engine.ir import compile_expressions
from factor_engine.native_abi import NativeNode, encode_nodes
from factor_engine.native_plan_backend import NativePlanBackend
from factor_engine.runtime import ROOT
from factor_engine.session_source import ArraySource

from tests.cuda.test_cuda_numerics import export_values


@pytest.fixture(scope="module")
def fusion_schedule(tmp_path_factory):
    """直接执行生产主机调度器；无需 CUDA、GPU admission 或动态库重建。"""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host fusion scheduling tests require a C++ compiler")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("dag-fusion-host")
    source = directory / "schedule.cpp"
    library = directory / "schedule.so"
    source.write_text(
        '#include "cuda/pointwise.cuh"\n'
        'extern "C" void schedule(const factor_engine::PlanNodePOD* nodes, int count,\n'
        "                         const int* roots, int n_roots, unsigned char* out) {\n"
        "    const auto spans = factor_engine::cuda::build_pointwise_fusion_schedule(nodes, count, roots, n_roots);\n"
        "    for (int i = 0; i < count; ++i) out[i] = spans[i];\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root),
            "-I",
            str(root / "cpp/include"),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    owner = C.CDLL(str(library))
    function = owner.schedule
    function.argtypes = [
        C.POINTER(NativeNode),
        C.c_int,
        C.POINTER(C.c_int),
        C.c_int,
        C.POINTER(C.c_ubyte),
    ]
    function.restype = None

    def schedule(plan):
        fields = tuple(dict.fromkeys(n.param for n in plan.nodes if n.op == "terminal"))
        nodes = encode_nodes(plan, fields)
        roots = (C.c_int * len(plan.root_ids))(*plan.root_ids)
        output = (C.c_ubyte * len(plan.nodes))()
        function(nodes, len(nodes), roots, len(roots), output)
        return list(output)

    return schedule


def _raw(expressions):
    return compile_expressions(expressions, canonicalize_exprs=False)


def _chain(a, b):
    return make_expr("safe_div", make_expr("mul", make_expr("sub", a, b), b), b)


def _groups(plan, schedule):
    return [(i, count) for i, count in enumerate(schedule(plan)) if count]


def test_host_fuses_sub_mul_safe_div_and_preserves_existing_slots(fusion_schedule):
    plan = _raw([_chain(terminal("a"), terminal("b"))])
    assert [n.op for n in plan.nodes] == ["terminal", "terminal", "sub", "mul", "safe_div"]
    assert _groups(plan, fusion_schedule) == [(2, 3)]
    # 最后一个输出复用 sub 的 slot；寄存器链必须避免重读该槽位。
    assert plan.nodes[2].slot == plan.nodes[4].slot


def test_host_preserves_intermediate_roots_and_fanout(fusion_schedule):
    a, b = terminal("a"), terminal("b")
    difference = make_expr("sub", a, b)
    scaled = make_expr("mul", difference, b)
    output = make_expr("safe_div", scaled, b)
    rooted = _raw([difference, output])
    assert _groups(rooted, fusion_schedule) == [(3, 2)]
    branched = _raw([output, make_expr("add", difference, b)])
    assert _groups(branched, fusion_schedule) == [(3, 2)]
    repeated = _raw([make_expr("mul", difference, difference)])
    assert _groups(repeated, fusion_schedule) == []


def test_host_does_not_cross_scalar_or_temporal_barriers(fusion_schedule):
    a = terminal("a")
    scalar = make_expr("add", constant(1), constant(2))
    mixed = _raw([make_expr("mul", make_expr("neg", a), scalar)])
    assert _groups(mixed, fusion_schedule) == []
    temporal = _raw([make_expr("neg", make_expr("delay", make_expr("neg", a), param=1))])
    assert _groups(temporal, fusion_schedule) == []
    # 先已折叠的 scalar 可以成为融合组边界输入。
    available = _raw([scalar, make_expr("abs", make_expr("add", a, scalar))])
    assert [count for _, count in _groups(available, fusion_schedule)] == [2]


def test_host_bounds_long_chain_without_abandoning_tail(fusion_schedule):
    expression = terminal("a")
    for index in range(19):
        expression = make_expr("neg" if index % 2 else "abs", expression)
    plan = _raw([expression])
    assert _groups(plan, fusion_schedule) == [(1, 8), (9, 8), (17, 3)]


def _assert_same_values(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(actual, expected)
    # assert_array_equal 把 +/-0 视为相等；有限结果额外验证 float32 位模式。
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual.view(np.uint32)[finite], expected.view(np.uint32)[finite])


def _numeric_panel():
    rng = np.random.default_rng(1729)
    a = rng.integers(-4, 5, (1, 3, 257)).astype(np.float32)
    b = rng.integers(1, 5, a.shape).astype(np.float32)
    a[0, 0, :9] = [
        np.nan,
        np.inf,
        -np.inf,
        0.0,
        -0.0,
        np.finfo(np.float32).max,
        -np.finfo(np.float32).max,
        16777216,
        -16777216,
    ]
    threshold = np.float32(1e-12)
    b[0, 1, :6] = [
        0.0,
        -0.0,
        threshold,
        -threshold,
        np.nextafter(threshold, np.float32(np.inf)),
        np.nextafter(-threshold, np.float32(-np.inf)),
    ]
    a[0, 1, :6] = 1.0
    b[0, 2, :3] = [np.nan, np.inf, -np.inf]
    target = rng.integers(-3, 4, a.shape).astype(np.float32)
    target[0, 1, 11] = np.nan
    valid = np.ones(a.shape, dtype=bool)
    valid[0, 2, 13] = False
    return Panel({"a": a, "b": b}, valid), target


@pytest.mark.parametrize(
    "op",
    [
        "neg",
        "abs",
        "sign",
        "signed_sqrt",
        "signed_log1p",
        "add",
        "sub",
        "mul",
        "safe_div",
        "min",
        "max",
    ],
)
def test_fused_stages_keep_cpu_float32_values_and_tied_scores(gpu_backend, op):
    panel, target = _numeric_panel()
    a, b = terminal("a"), terminal("b")
    unary = {"neg", "abs", "sign", "signed_sqrt", "signed_log1p"}
    stage = make_expr(op, a) if op in unary else make_expr(op, a, b)
    plan = _raw([make_expr("neg", stage)])
    with closing(NativePlanBackend()) as cpu:
        _assert_same_values(export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel))
        np.testing.assert_array_equal(
            gpu_backend.score_rank_ic(plan, panel, target), cpu.score_rank_ic(plan, panel, target)
        )
    assert gpu_backend.last_run_stats["counter_summary"]["kernel_launches"] == 2


def test_fused_overflow_is_cleaned_before_a_later_min(gpu_backend):
    panel, _ = _numeric_panel()
    # 若融合只清理最终输出，min(inf, b) 会产生有限结果并污染有效样本交集。
    a, b = terminal("a"), terminal("b")
    plan = _raw([make_expr("min", make_expr("mul", a, b), b)])
    with closing(NativePlanBackend()) as cpu:
        _assert_same_values(export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel))


def test_fused_chain_reduces_launches_and_keeps_root_and_fanout_values(gpu_backend):
    panel, target = _numeric_panel()
    a, b = terminal("a"), terminal("b")
    difference = make_expr("sub", a, b)
    output = _chain(a, b)
    plans = [
        (_raw([output]), 2),
        (_raw([difference, output]), 3),
        (_raw([output, make_expr("add", difference, b)]), 4),
    ]
    with closing(NativePlanBackend()) as cpu:
        for plan, launches in plans:
            _assert_same_values(export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel))
            np.testing.assert_array_equal(
                gpu_backend.score_rank_ic(plan, panel, target),
                cpu.score_rank_ic(plan, panel, target),
            )
            assert gpu_backend.last_run_stats["counter_summary"]["kernel_launches"] == launches


def test_fused_multiply_add_does_not_contract_across_stage_boundary(gpu_backend):
    # (1 + 2^-23) * (1 - 2^-23) 在 float32 中舍入为 1；融合 FMA 会保留 -2^-46。
    a = np.full((1, 3, 257), np.float32(1 + 2**-23), dtype=np.float32)
    b = np.full_like(a, np.float32(1 - 2**-23))
    c = np.full_like(a, -1.0)
    panel = Panel({"a": a, "b": b, "c": c})
    # 提前 emit c，保证 mul/add 连续且 c 是边界输入。
    plan = _raw(
        [
            terminal("c"),
            make_expr("add", make_expr("mul", terminal("a"), terminal("b")), terminal("c")),
        ]
    )
    with closing(NativePlanBackend()) as cpu:
        actual = export_values(gpu_backend, plan, panel)
        _assert_same_values(actual, cpu.evaluate(plan, panel))
        assert np.all(actual[1] == 0.0)


def test_fused_scalar_boundary_and_long_chain_keep_rounding(gpu_backend):
    panel, _ = _numeric_panel()
    scalar = make_expr("safe_div", constant(1), constant(3))
    expression = make_expr("add", terminal("a"), scalar)
    for index in range(18):
        expression = make_expr("neg" if index % 2 else "abs", expression)
    # 第一个 series root 使 scalar 先被折叠；第二条长链只引用已有的 scalar。
    plan = _raw([make_expr("add", terminal("b"), scalar), expression])
    with closing(NativePlanBackend()) as cpu:
        _assert_same_values(export_values(gpu_backend, plan, panel), cpu.evaluate(plan, panel))


def test_fused_dag_export_matches_cpu():
    if os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("explicit GPU test required")
    x = np.arange(2 * 97, dtype=np.float32).reshape(1, 2, 97) - 90
    panel = Panel({"close": x})
    plan = compile_expressions([abs_(neg(add(terminal("close"), terminal("close"))))])
    source = ArraySource(panel, x, source_identity="gpu-dag-test")
    panel, target = source.panel, source.target
    cpu = NativePlanBackend()
    expected = cpu.evaluate(plan, panel)
    with CUDABuffers(
        ROOT / "build-study/libfactor_engine_cuda.so",
        field_names=("close",),
        capacity_lanes=2,
        time_length=97,
        plan=plan,
    ) as owner:
        owner._upload(0, panel, target, final=True)
        output = owner.evaluate(0, 2, plan)
        np.testing.assert_array_equal(output, expected)
    cpu.close()


@pytest.mark.parametrize("root_count", [1, 2])
def test_dag_export_honors_exact_budget(root_count):
    if os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("explicit GPU test required")
    x = np.arange(194, dtype=np.float32).reshape(1, 2, 97)
    source = ArraySource(Panel({"close": x}), x, source_identity="dag-budget")
    expressions = [neg(terminal("close")), abs_(terminal("close"))][:root_count]
    plan = compile_expressions(expressions)
    with CUDABuffers(
        ROOT / "build-study/libfactor_engine_cuda.so",
        field_names=("close",),
        capacity_lanes=2,
        time_length=97,
        plan=plan,
    ) as owner:
        owner._upload(0, source.panel, source.target, final=True)
        owner._prepare(2, plan)
        owner.budget = owner.payload_bytes + owner.native.requirement(owner.handle, plan)
        if root_count == 2:
            with pytest.raises(MemoryError, match="workspace"):
                owner.evaluate(0, 2, plan)
        else:
            owner.evaluate(0, 2, plan)
            assert (
                owner.payload_bytes + owner.native.requirement(owner.handle, plan) <= owner.budget
            )
