"""CPU 和 CUDA 共用的节点编号、C 结构和参数转换。"""

import ctypes as C
from enum import IntEnum
import operator
import sys

from .dsl import INTEGER_PARAMETER_OPS, OPERATORS, ValueType
from .metric_types import NativeLimit
from .ir import constant_param, native_int_param, terminal_param


class PlanOp(IntEnum):
    TERMINAL = 0
    CONST = 1
    NEG = 2
    ABS = 3
    SIGN = 4
    SIGNED_SQRT = 5
    SIGNED_LOG1P = 6
    ADD = 7
    SUB = 8
    MUL = 9
    SAFE_DIV = 10
    MIN = 11
    MAX = 12
    DELAY = 13
    DIFF = 14
    EMA = 15
    ROLL_SUM = 16
    ROLL_MEAN = 17
    ROLL_STD = 18
    ROLL_MIN = 19
    ROLL_MAX = 20
    COV = 21
    CORR = 22


class PlanValueType(IntEnum):
    SERIES = 0
    SCALAR = 1


OPCODES = {op.name.lower(): op for op in PlanOp}
FloatPointer = C.POINTER(C.c_float)


class NativeNode(C.Structure):
    _fields_ = [
        ("op", C.c_int32),
        ("in0", C.c_int32),
        ("in1", C.c_int32),
        ("param_i", C.c_int32),
        ("field_index", C.c_int32),
        ("slot", C.c_int32),
        ("value_type", C.c_int32),
        ("param_f", C.c_float),
    ]


def encode_nodes(plan, field_names):
    fields = {name: index for index, name in enumerate(field_names)}
    nodes = (NativeNode * len(plan.nodes))()
    for index, node in enumerate(plan.nodes):
        nodes[index] = NativeNode(
            OPCODES[node.op],
            node.inputs[0] if node.inputs else NativeLimit.NO_INDEX,
            node.inputs[1] if len(node.inputs) > 1 else NativeLimit.NO_INDEX,
            native_int_param(node),
            fields[terminal_param(node)] if node.op == "terminal" else NativeLimit.NO_INDEX,
            node.slot,
            PlanValueType.SERIES if node.value_type is ValueType.SERIES else PlanValueType.SCALAR,
            constant_param(node) if node.op == "const" else 0.0,
        )
    return nodes


def require_current_abi(library):
    """在绑定计算函数前确认 ABI，旧动态库不能带着错误参数布局进入计算。"""
    try:
        function = library.fe_native_abi_version
    except AttributeError as error:
        raise RuntimeError(
            "native ABI differs; rebuild the current production libraries"
        ) from error
    function.argtypes, function.restype = [], C.c_int
    actual = function()
    expected = int(NativeLimit.ABI_VERSION)
    if actual != expected:
        raise RuntimeError(
            f"native ABI differs: expected {expected}, received {actual}; "
            "rebuild the current production libraries"
        )


def checked_i32(value, name, minimum=0):
    value = operator.index(value)
    if not minimum <= value <= NativeLimit.INT32_MAX:
        raise OverflowError(f"{name} outside supported int32 range")
    return value


def validate_shape(shape, fields, *, cuda=False):
    if len(shape) != 3:
        raise ValueError("panel shape must be session/security/time")
    sessions, securities, time = [checked_i32(v, "dimension", 1) for v in shape]
    lanes = checked_i32(sessions * securities, "lanes", 1)
    fields = checked_i32(fields, "fields", 1)
    cells = lanes * time
    if cells * fields * 4 > sys.maxsize:
        raise OverflowError("panel bytes exceed addressable native storage")
    blocks = (cells + NativeLimit.CUDA_THREADS_PER_BLOCK - 1) // NativeLimit.CUDA_THREADS_PER_BLOCK
    if cuda and blocks > NativeLimit.INT32_MAX:
        raise OverflowError("CUDA launch outside supported int32 grid")
    return lanes, time, cells


def validate_rank_shape(roots, lanes, time, *, segmented=True):
    roots = checked_i32(roots, "roots", 1)
    lanes = checked_i32(lanes, "lanes", 1)
    time = checked_i32(time, "time", 1)
    items, segments = roots * lanes * time, roots * lanes
    if segments > NativeLimit.INT32_MAX or (segmented and items > NativeLimit.INT32_MAX):
        raise OverflowError("CUB item/segment counts exceed int32")
    return items, segments


def validate_plan_abi(plan, fields, shape, *, cuda=False, rank=False):
    lanes, time, cells = validate_shape(shape, len(fields), cuda=cuda)
    nodes = checked_i32(len(plan.nodes), "nodes", 1)
    roots = checked_i32(len(plan.root_ids), "roots", 1)
    slots = checked_i32(plan.slot_count, "slots")
    if cells * max(slots, roots) * 4 > sys.maxsize:
        raise OverflowError("workspace/output bytes exceed native storage")
    if rank:
        validate_rank_shape(roots, lanes, time, segmented=not cuda)
    for i, node in enumerate(plan.nodes):
        if node.node_id != i or node.op not in OPERATORS:
            raise ValueError("invalid topological node/op")
        if len(node.inputs) != OPERATORS[node.op].arity:
            raise ValueError("invalid native node arity")
        for child in node.inputs:
            checked_i32(child, "input index")
            if child >= i:
                raise ValueError("inputs must precede consumers")
        checked_i32(node.slot, "slot", -1)
        if (
            node.value_type is ValueType.SERIES
            and node.op != "terminal"
            and not 0 <= node.slot < slots
        ):
            raise ValueError("invalid Series workspace slot")
        if node.op == "terminal" and str(node.param) not in fields:
            raise KeyError(f"missing field {node.param}")
        if node.op in INTEGER_PARAMETER_OPS:
            checked_i32(node.param, "operator parameter", 2 if node.op in {"cov", "corr"} else 1)
    for root in plan.root_ids:
        checked_i32(root, "root index")
        if root >= nodes:
            raise ValueError("root outside plan")
    return lanes, time, cells


class PreparedPlan:
    def __init__(self, plan, field_names):
        self.plan = plan
        self.field_names = tuple(field_names)
        self.nodes = encode_nodes(plan, self.field_names)
        self.roots = (C.c_int * len(plan.root_ids))(*plan.root_ids)


class PlanCache:
    def __init__(self):
        self.current = None
        self.preparations = 0

    def get(self, plan, field_names, shape, *, cuda=False, rank=False):
        names = tuple(field_names)
        lanes, time, cells = validate_shape(shape, len(names), cuda=cuda)
        if cells * max(plan.slot_count, len(plan.root_ids)) * 4 > sys.maxsize:
            raise OverflowError("workspace/output bytes exceed native storage")
        if rank:
            validate_rank_shape(len(plan.root_ids), lanes, time, segmented=not cuda)
        current = self.current
        if current is None or current.plan is not plan or current.field_names != names:
            # ExecutionPlan/Node 是不可变 dataclass；只省掉同一对象的重复编码。
            validate_plan_abi(plan, names, shape, cuda=cuda, rank=rank)
            current = PreparedPlan(plan, names)
            self.current = current
            self.preparations += 1
        return current

    def clear(self):
        self.current = None
