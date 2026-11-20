"""CUDA 缓冲区的 ctypes 接口；导入模块时不加载动态库。"""

from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np
from .metric_types import MIN_CORRELATION_PAIRS, CudaRuntimeStatus, CudaStatus

from .native_abi import (
    validate_plan_abi,
    validate_shape,
    FloatPointer,
    NativeNode,
    require_current_abi,
    PlanCache,
)

_I64P = C.POINTER(C.c_longlong)
DIRECT_ROLLING_WINDOW_LIMIT = 32


class MetricResources(C.Structure):
    _fields_ = [
        (name, C.c_longlong)
        for name in (
            "workspace_bytes",
            "requested_scratch_bytes",
            "retained_scratch_bytes",
            "owned_device_bytes",
        )
    ]


class MetricCapacity(C.Structure):
    _fields_ = [
        (name, C.c_longlong)
        for name in (
            "active_storage_bytes",
            "retained_storage_bytes",
            "prepare_peak_metric_bytes",
            "prepare_peak_owned_bytes",
        )
    ]


F64P = C.POINTER(C.c_double)
U8P = C.POINTER(C.c_uint8)


class UploadStats(C.Structure):
    _fields_ = [("upload_h2d_bytes", C.c_longlong), ("sanitize_launches", C.c_longlong)]

    def as_dict(self):
        return {name: int(getattr(self, name)) for name, *_ in self._fields_}


class ScoreStats(C.Structure):
    _fields_ = [
        ("h2d_bytes", C.c_longlong),
        ("kernel_launches", C.c_longlong),
        ("root_export_bytes", C.c_longlong),
        ("rank_workspace_bytes", C.c_longlong),
        ("owned_device_bytes", C.c_longlong),
        ("input_target_h2d_bytes", C.c_longlong),
        ("lane_ic_d2h_bytes", C.c_longlong),
    ]

    def as_dict(self):
        return {name: int(getattr(self, name)) for name, *_ in self._fields_}


class DeviceInfo(C.Structure):
    _fields_ = [
        ("free_bytes", C.c_longlong),
        ("total_bytes", C.c_longlong),
        ("sm_count", C.c_int),
        ("max_shared_bytes", C.c_int),
        ("device", C.c_int),
    ]


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_upload(panel, target):
    names = tuple(panel.fields)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("CUDA inputs requires named ordered fields")
    arrays = (*panel.fields.values(), panel.valid, target)
    for index, a in enumerate(arrays):
        dtype = np.bool_ if index == len(names) else np.float32
        if (
            not isinstance(a, np.ndarray)
            or a.dtype != dtype
            or a.shape != panel.shape
            or not a.flags.c_contiguous
            or a.flags.writeable
        ):
            raise ValueError(
                "CUDA inputs requires immutable contiguous owned FP32 fields/target and bool mask"
            )
    validate_shape(panel.shape, len(names), cuda=True)


def input_payload_bytes(fields, lanes, time):
    for value, name in ((fields, "fields"), (lanes, "lanes"), (time, "time")):
        _positive(value, name)
        if value > 2**31 - 1:
            raise ValueError("input dimensions exceed native int32 ABI")
    if time < MIN_CORRELATION_PAIRS:
        raise ValueError("RankIC input needs at least three time points")
    required = lanes * time * (4 * fields + 5)
    if required > 2**63 - 1:
        raise MemoryError("input payload exceeds native int64 ABI")
    return required


class CUDANative:
    """绑定 CUDA 的 C 接口，负责数组指针和 plan 结构转换。"""

    def __init__(self, library):
        self.field_names: tuple[str, ...] = ()
        self.plans = PlanCache()
        path = Path(library).resolve(strict=True)
        self.lib = C.CDLL(str(path))
        require_current_abi(self.lib)
        definitions = {
            "device_info": ([C.POINTER(DeviceInfo)], C.c_int),
            "input_bytes": ([C.c_void_p], C.c_longlong),
            "requirement": (
                [C.c_void_p, C.POINTER(NativeNode), C.c_int, C.c_int, C.c_int, _I64P],
                C.c_int,
            ),
            "export_requirement": (
                [C.c_void_p, C.POINTER(NativeNode), C.c_int, C.c_int, C.c_int, _I64P],
                C.c_int,
            ),
            "free_bytes": ([], C.c_longlong),
            "shape_requirement": (
                [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, _I64P],
                C.c_int,
            ),
        }
        self._bind_api("runtime", definitions)

        backend_definitions = {
            "allocate_checked": (
                [C.c_int, C.c_int, C.c_int, C.c_longlong, C.POINTER(C.c_void_p)],
                C.c_int,
            ),
            "release": ([C.c_void_p], C.c_int),
            "upload_window": (
                [
                    C.c_void_p,
                    C.c_int,
                    C.POINTER(FloatPointer),
                    C.POINTER(C.c_uint8),
                    FloatPointer,
                    C.c_int,
                    C.c_int,
                    C.c_int,
                    C.c_int,
                    C.POINTER(UploadStats),
                ],
                C.c_int,
            ),
            "set_batch_width": ([C.c_void_p, C.c_int], C.c_int),
            "score_lanes": (
                [
                    C.c_void_p,
                    C.c_int,
                    C.POINTER(NativeNode),
                    C.c_int,
                    C.c_int,
                    C.POINTER(C.c_int),
                    C.c_int,
                    C.POINTER(C.c_double),
                    C.c_longlong,
                    C.POINTER(ScoreStats),
                ],
                C.c_int,
            ),
            "evaluate": (
                [
                    C.c_void_p,
                    C.c_int,
                    C.POINTER(NativeNode),
                    C.c_int,
                    C.c_int,
                    C.POINTER(C.c_int),
                    C.c_int,
                    FloatPointer,
                    C.c_longlong,
                ],
                C.c_int,
            ),
        }
        self._bind_api("backend", backend_definitions)

        definitions = {
            "prepare": ([C.c_void_p] + [C.c_int] * 2 + [C.c_longlong, C.POINTER(MetricResources)]),
            "upload": ([C.c_void_p, FloatPointer, U8P]),
            "score": (
                [
                    C.c_void_p,
                    C.c_void_p,
                    C.c_int,
                    C.POINTER(NativeNode),
                    C.c_int,
                    C.c_int,
                    C.POINTER(C.c_int32),
                    C.c_int,
                    F64P,
                ]
            ),
            "read": ([C.c_void_p, C.c_int, F64P]),
            "resources": ([C.c_void_p, C.POINTER(MetricResources)]),
            "capacity": ([C.c_void_p, C.POINTER(MetricCapacity)]),
            "inputs_ready": ([C.c_void_p, C.POINTER(C.c_int)]),
        }
        self._bind_api("metric", {name: (args, C.c_int) for name, args in definitions.items()})

    def _bind_api(self, group, definitions):
        for name, (arguments, result) in definitions.items():
            symbol = f"fe_cuda_{group}_{name}"
            try:
                function = getattr(self.lib, symbol)
            except AttributeError as error:
                raise RuntimeError(
                    f"CUDA library missing {symbol}; rebuild the current targets"
                ) from error
            function.argtypes, function.restype = arguments, result

    def shape_requirement(self, field_names, lanes, time, plan):
        validate_plan_abi(plan, tuple(field_names), (1, lanes, time), cuda=True, rank=True)
        result = C.c_longlong()
        unary = int(
            any(
                (n.op in ("roll_sum", "roll_mean") and n.param > DIRECT_ROLLING_WINDOW_LIMIT)
                or (
                    n.op in ("roll_min", "roll_max")
                    and DIRECT_ROLLING_WINDOW_LIMIT < n.param <= time
                )
                for n in plan.nodes
            )
        )
        pair = max(
            (int(n.param) for n in plan.nodes if n.op in ("cov", "corr") and n.param <= time),
            default=0,
        )
        std = max(
            (int(n.param) for n in plan.nodes if n.op == "roll_std" and n.param <= time), default=0
        )
        arguments = [len(field_names), lanes, time, plan.slot_count, len(plan.root_ids)]
        rc = self.lib.fe_cuda_runtime_shape_requirement(
            *arguments, unary, pair, std, C.byref(result)
        )
        if rc or result.value < 0:
            raise RuntimeError(f"native CUDA shape query failed: {rc}")
        return result.value

    def input_bytes(self, handle):
        return int(self.lib.fe_cuda_runtime_input_bytes(handle))

    def requirement(self, handle, plan, *, for_export=False):
        prepared = self.plans.get(plan, self.field_names, (1, 1, self.time), cuda=True)
        result = C.c_longlong()
        function = self.lib.fe_cuda_runtime_requirement
        if for_export:
            function = self.lib.fe_cuda_runtime_export_requirement
        rc = function(
            handle,
            prepared.nodes,
            len(plan.nodes),
            plan.slot_count,
            len(plan.root_ids),
            C.byref(result),
        )
        if rc:
            raise RuntimeError(f"native CUDA workspace requirement failed: {rc}")
        return result.value

    def free_bytes(self):
        available = int(self.lib.fe_cuda_runtime_free_bytes())
        if available < 0:
            raise RuntimeError("native CUDA free-memory query failed")
        return available

    def device_info(self):
        info = DeviceInfo()
        status = self.lib.fe_cuda_runtime_device_info(C.byref(info))
        if status:
            raise RuntimeError(f"CUDA device query failed: {status}")
        return {name: int(getattr(info, name)) for name, *_ in info._fields_}

    def allocate(self, names, lanes, time_length, budget):
        self.field_names, self.time = tuple(names), time_length
        self.capacity = lanes
        output = C.c_void_p()
        status = self.lib.fe_cuda_backend_allocate_checked(
            len(names), lanes, time_length, budget, C.byref(output)
        )
        if status == CudaStatus.ALLOCATION_FAILED:
            raise MemoryError("CUDA input allocation OOM or budget exceeded")
        if status:
            raise RuntimeError(f"CUDA checked input allocation failed: {status}")
        if not output.value:
            raise RuntimeError("CUDA checked allocation succeeded without a handle")
        return output.value

    def upload(self, handle, first, panel, target, *, final):
        _validate_upload(panel, target)
        if tuple(panel.fields) != self.field_names or panel.shape[-1] != self.time:
            raise ValueError("Generic CUDA upload terminal/time identity differs")
        ptrs = (FloatPointer * len(self.field_names))(
            *(a.ctypes.data_as(FloatPointer) for a in panel.fields.values())
        )
        stats = UploadStats()
        status = self.lib.fe_cuda_backend_upload_window(
            handle,
            first,
            ptrs,
            panel.valid.ctypes.data_as(C.POINTER(C.c_uint8)),
            target.ctypes.data_as(FloatPointer),
            len(self.field_names),
            panel.shape[0] * panel.shape[1],
            self.time,
            int(final),
            C.byref(stats),
        )
        if status:
            raise RuntimeError(f"Generic CUDA upload failed: {status}; preceding data invalidated")
        return stats.as_dict()

    def set_batch_width(self, handle, width):
        if self.lib.fe_cuda_backend_set_batch_width(handle, width):
            raise RuntimeError("CUDA batch configuration failed")

    def release(self, handle):
        if self.lib.fe_cuda_backend_release(handle):
            raise RuntimeError("CUDA buffer release failed; owner retained")

    def score_lanes(self, handle, first, width, plan):
        prepared = self.plans.get(
            plan, self.field_names, (1, width, self.time), cuda=True, rank=True
        )
        nodes, roots = prepared.nodes, prepared.roots
        values = np.empty((len(roots), width), dtype=np.float64)
        stats = ScoreStats()
        arguments = [handle, first, nodes, len(nodes), plan.slot_count, roots, len(roots)]
        status = self.lib.fe_cuda_backend_score_lanes(
            *arguments,
            values.ctypes.data_as(C.POINTER(C.c_double)),
            values.nbytes,
            C.byref(stats),
        )
        if status:
            if status == CudaRuntimeStatus.WORKSPACE_ALLOCATION_FAILED:
                raise MemoryError("CUDA workspace allocation OOM after memory query")
            raise RuntimeError(f"Generic CUDA exact RankIC failed: {status}; no fallback")
        return values, stats.as_dict()

    def evaluate(self, handle, first, width, plan):
        shape = (1, width, self.time)
        output = np.empty((len(plan.root_ids), *shape), dtype=np.float32)
        prepared = self.plans.get(plan, self.field_names, shape, cuda=True)
        nodes, roots = prepared.nodes, prepared.roots
        status = self.lib.fe_cuda_backend_evaluate(
            handle,
            first,
            nodes,
            len(nodes),
            plan.slot_count,
            roots,
            len(roots),
            output.ctypes.data_as(FloatPointer),
            output.nbytes,
        )
        if status:
            raise RuntimeError(f"Generic CUDA factor export failed: {status}")
        return output
