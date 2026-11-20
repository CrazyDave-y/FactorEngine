import ctypes
import ctypes as C
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from factor_engine.cuda_native import DeviceInfo, ScoreStats, UploadStats
from factor_engine.data import Panel
from factor_engine.dsl import OPERATORS, ValueType
from factor_engine.expression import add, constant, rolling_mean, terminal
from factor_engine.ir import Node, compile_expressions
from factor_engine.metric_types import CudaStatus, Metric, NativeLimit, SumCount
from factor_engine.native_abi import FloatPointer, NativeNode
from factor_engine.native_plan_backend import NativePlanBackend


def test_native_operator_codes_cover_the_dsl_and_keep_the_c_abi():
    from factor_engine.native_abi import OPCODES, NativeNode

    assert set(OPCODES) == set(OPERATORS)
    assert tuple(map(int, OPCODES.values())) == tuple(range(23))
    assert ctypes.sizeof(NativeNode) == struct.calcsize("=7if")


def test_native_encoding_preserves_float32_constant_and_slot_layout():
    from factor_engine.native_abi import PlanOp, encode_nodes

    plan = compile_expressions([add(terminal("x"), constant(0.1))], canonicalize_exprs=False)
    nodes = encode_nodes(plan, ("x",))
    assert bytes(nodes[0]) == struct.pack("=7if", PlanOp.TERMINAL, -1, -1, 0, 0, -1, 0, 0.0)
    assert bytes(nodes[1]) == struct.pack("=7if", PlanOp.CONST, -1, -1, 0, -1, -1, 1, 0.1)
    assert nodes[2].in0 == 0 and nodes[2].in1 == 1
    assert nodes[2].slot == plan.nodes[2].slot


def test_native_encoding_rejects_string_constants_before_pointer_conversion():
    from dataclasses import replace

    from factor_engine.native_abi import encode_nodes

    plan = compile_expressions([constant(1.0)])
    invalid = replace(plan, nodes=(Node(0, "const", (), ValueType.SCALAR, "1.0"),))
    with pytest.raises(TypeError, match="numeric constant"):
        encode_nodes(invalid, ("x",))


def test_cpu_memory_layout_uses_named_int_fields():
    from contextlib import closing

    from factor_engine.native_plan_backend import CPULayout, NativePlanBackend

    with closing(NativePlanBackend()) as backend:
        layout = backend.cpu_layout()
    names = tuple(name for name, *_ in CPULayout._fields_)
    assert set(layout) == set(names)
    assert ctypes.sizeof(CPULayout) == len(names) * ctypes.sizeof(ctypes.c_int)
    assert layout["float_bytes"] == ctypes.sizeof(ctypes.c_float)
    assert layout["double_bytes"] == ctypes.sizeof(ctypes.c_double)
    assert layout["int_bytes"] == ctypes.sizeof(ctypes.c_int)
    assert layout["max_threads"] >= 1


def test_cpp_layouts_and_enums_match_ctypes():
    library = os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB")
    if not library:
        pytest.skip("explicit native build required")
    executable = Path(library).parent / "factor_engine_contracts"
    if not executable.is_file():
        pytest.fail("current native build must include factor_engine_contracts")
    result = json.loads(subprocess.check_output([str(executable)], text=True))
    structures = (NativeNode, DeviceInfo, ScoreStats, UploadStats)
    for structure in structures:
        native = result["layouts"][structure.__name__]
        assert C.sizeof(structure) == native["bytes"]
        assert C.alignment(structure) == native["alignment"]
        assert {name: getattr(structure, name).offset for name, *_ in structure._fields_} == native[
            "offsets"
        ]
    assert [int(x) for x in Metric] == result["metric"]
    assert [int(x) for x in CudaStatus] == result["cuda_status"]
    assert [int(x) for x in SumCount] == result["sum_count"]
    assert NativeLimit.MAX_RANK_TIME == result["max_rank_time"]


def test_checked_abi_dimensions_products_and_rank_boundary_without_allocation():
    from factor_engine.native_abi import validate_rank_shape, validate_shape

    assert validate_shape((1, 2, 128), 1) == (2, 128, 256)
    assert validate_rank_shape(1, 1, 2**31 - 1) == (2**31 - 1, 1)
    assert validate_rank_shape(2**31 - 1, 1, 1) == (2**31 - 1, 2**31 - 1)
    for shape in ((1, 2**31, 1), (2**30, 4, 1), (1, 1, 2**31)):
        with pytest.raises(OverflowError, match="int32"):
            validate_shape(shape, 1)
    with pytest.raises(OverflowError, match="CUB"):
        validate_rank_shape(2, 1, 2**30)
    with pytest.raises(OverflowError, match="bytes"):
        validate_shape((1, 2**31 - 1, 2**31 - 1), 2)


@pytest.mark.skipif(not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="native build")
def test_native_serializer_rejects_both_wraps_before_allocation_or_indexing():
    x = np.arange(1024, dtype=np.float32).reshape(1, 1, -1)
    panel = Panel({"x": x})
    backend = NativePlanBackend()
    for window in (2**32 + 1, 2**31 + 1):
        with pytest.raises(OverflowError, match="int32"):
            backend._prepare(compile_expressions([rolling_mean(terminal("x"), window)]), panel)
    assert backend.workspace_allocation_count == 0
    with pytest.raises(OverflowError, match="int32"):
        backend.score_rank_ic_stats(
            compile_expressions([rolling_mean(terminal("x"), 2**32 + 1)]), panel, x
        )


@pytest.mark.skipif(not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="native build")
def test_native_negative_parameter_rejected_before_workspace_or_indexing():
    # 只触发参数校验，避免负窗口进入实际索引计算。

    backend = NativePlanBackend()
    x = np.arange(8, dtype=np.float32)
    ptrs = (FloatPointer * 1)(backend._ptr(x))
    bad = (NativeNode * 2)(
        NativeNode(0, -1, -1, 0, 0, -1, 0, 0), NativeNode(17, 0, -1, -1, -1, 0, 0, 0)
    )
    roots = (ctypes.c_int * 1)(1)
    out = np.empty(8, dtype=np.float32)
    assert backend._fn(bad, 2, ptrs, 1, 1, 8, 1, None, 0, roots, 1, backend._ptr(out)) == -6
