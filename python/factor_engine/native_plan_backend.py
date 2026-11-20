from __future__ import annotations

import ctypes
import os
from pathlib import Path
import numpy as np
from .metric_types import SumCount, SUM_COUNT_WIDTH, METRIC_COUNT, CORRELATION_TOLERANCE

from .ir import ExecutionPlan
from .native_abi import FloatPointer, NativeNode, require_current_abi, PlanCache


class CPULayout(ctypes.Structure):
    _fields_ = [
        ("float_bytes", ctypes.c_int),
        ("double_bytes", ctypes.c_int),
        ("int_bytes", ctypes.c_int),
        ("sort_pair_bytes", ctypes.c_int),
        ("unary_entry_bytes", ctypes.c_int),
        ("unary_moments_bytes", ctypes.c_int),
        ("pair_entry_bytes", ctypes.c_int),
        ("pair_moments_bytes", ctypes.c_int),
        ("max_threads", ctypes.c_int),
    ]


class NativePlanBackend:
    """一次原生调用执行整张 DAG。"""

    name = "native_plan_cpu"

    def __init__(self, library: str | os.PathLike[str] | None = None):
        requested = Path(library) if library else None
        if requested is None and os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"):
            requested = Path(os.environ["FACTORENGINE_NATIVE_PLAN_LIB"])
        path = (
            requested
            or Path(__file__).resolve().parents[2] / "build-study" / "libfactor_engine_plan.so"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "build libfactor_engine_plan.so or set FACTORENGINE_NATIVE_PLAN_LIB"
            )
        self._library_path = path.resolve()
        self._lib = ctypes.CDLL(str(path))
        require_current_abi(self._lib)
        fn = self._lib.fe_execute_plan_cpu
        fn.argtypes = [
            ctypes.POINTER(NativeNode),
            ctypes.c_int,
            ctypes.POINTER(FloatPointer),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            FloatPointer,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            FloatPointer,
        ]
        fn.restype = ctypes.c_int
        self._fn = fn
        self._metric_fn = self._lib.fe_score_plan_metrics_cpu
        self._metric_fn.argtypes = fn.argtypes[:-1] + [
            FloatPointer,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._metric_fn.restype = ctypes.c_int
        self._pearson_fn = self._lib.fe_pearson_ic_lanes_cpu
        self._pearson_fn.argtypes = [
            ctypes.POINTER(FloatPointer),
            ctypes.c_int,
            FloatPointer,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._pearson_fn.restype = ctypes.c_int
        score = self._lib.fe_score_plan_rank_ic_cpu
        score.argtypes = fn.argtypes[:-1] + [
            FloatPointer,
            ctypes.POINTER(ctypes.c_double),
        ]
        score.restype = ctypes.c_int
        self._score_fn = score
        self._lane_fn = self._lib.fe_score_plan_lanes_cpu
        self._lane_fn.argtypes = score.argtypes
        self._lane_fn.restype = ctypes.c_int
        # 每个 C 函数只绑定一次；评分循环里只准备数据并调用。
        self._accumulate_fn = self._lib.fe_score_plan_rank_ic_accumulate_cpu
        self._accumulate_fn.argtypes, self._accumulate_fn.restype = score.argtypes, ctypes.c_int
        self._layout_fn = self._lib.fe_cpu_layout
        self._layout_fn.argtypes = [ctypes.POINTER(CPULayout)]
        self._layout_fn.restype = ctypes.c_int
        self._scratch_fn = self._lib.fe_cpu_rank_scratch_retained_bytes
        self._scratch_fn.argtypes, self._scratch_fn.restype = [], ctypes.c_uint64
        self._workspace = np.empty(0, dtype=np.float32)
        self._workspace_allocation_count = 0
        self.plans = PlanCache()

    @staticmethod
    def _ptr(a):
        return a.ctypes.data_as(FloatPointer)

    @property
    def workspace_allocation_count(self):
        return self._workspace_allocation_count

    @property
    def workspace_capacity_floats(self):
        return int(self._workspace.size)

    @property
    def rank_scratch_retained_bytes(self):
        return int(self._scratch_fn())

    def _ensure_workspace(self, required_floats: int):
        if self._workspace.size < required_floats:
            self._workspace = np.empty(required_floats, dtype=np.float32)
            self._workspace_allocation_count += 1

    def _prepare(self, plan: ExecutionPlan, panel):
        field_names = list(panel.fields)
        prepared = self.plans.get(plan, field_names, panel.shape)
        lanes = panel.shape[0] * panel.shape[1]
        time = panel.shape[2]
        cells = lanes * time
        # 在入口统一应用 valid，并把无穷值变成 NaN，原生算子就不用反复处理。
        fields = [
            np.ascontiguousarray(
                np.where(panel.valid & np.isfinite(panel.fields[name]), panel.fields[name], np.nan),
                dtype=np.float32,
            )
            for name in field_names
        ]
        field_ptrs = (FloatPointer * len(fields))(*[self._ptr(a) for a in fields])
        required_workspace = plan.slot_count * cells
        self._ensure_workspace(required_workspace)
        args = (
            prepared.nodes,
            len(plan.nodes),
            field_ptrs,
            len(fields),
            lanes,
            time,
            plan.slot_count,
            self._ptr(self._workspace),
            self._workspace.size,
            prepared.roots,
            len(plan.root_ids),
        )
        return args, fields

    def evaluate(self, plan: ExecutionPlan, panel):
        args, fields = self._prepare(plan, panel)
        out = np.empty((len(plan.root_ids),) + panel.shape, dtype=np.float32)
        self._execute_into(args, out)
        return [out[i] for i in range(out.shape[0])]

    def evaluate_into(self, plan: ExecutionPlan, panel, out):
        """直接写入调用方的连续 FP32 输出，支持 pinned 内存持有的 ndarray。"""
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.float32
            or out.shape != (len(plan.root_ids),) + panel.shape
            or not out.flags.c_contiguous
            or not out.flags.writeable
            or not out.flags.aligned
        ):
            raise ValueError("out must be writable aligned contiguous FP32 [roots]+panel.shape")
        args, fields = self._prepare(plan, panel)
        self._execute_into(args, out)

    def _execute_into(self, args, out):
        rc = self._fn(*args, self._ptr(out))
        if rc:
            raise RuntimeError(f"fe_execute_plan_cpu failed with status {rc}")

    def score_rank_ic_stats(self, plan: ExecutionPlan, panel, target):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != panel.shape:
            raise ValueError("target must match panel shape")
        target = np.ascontiguousarray(target)
        args, fields = self._prepare(plan, panel)
        stats = np.empty((len(plan.root_ids), SUM_COUNT_WIDTH), dtype=np.float64)

        rc = self._score_fn(
            *args,
            self._ptr(target),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc:
            raise RuntimeError(f"fe_score_plan_rank_ic_cpu failed with status {rc}")
        return stats

    def score_rank_ic(self, plan: ExecutionPlan, panel, target):
        stats = self.score_rank_ic_stats(plan, panel, target)
        return np.divide(
            stats[:, SumCount.SUM],
            stats[:, SumCount.COUNT],
            out=np.full(len(stats), np.nan),
            where=stats[:, SumCount.COUNT] > 0,
        )

    def score_lanes(self, plan, panel, target):
        """逐 lane 数值对照用；正常评分继续在原生代码中累计 Sum/Count。"""
        target = np.ascontiguousarray(target, dtype=np.float32)
        if target.shape != panel.shape:
            raise ValueError("target must match panel shape")
        args, fields = self._prepare(plan, panel)
        output = np.empty((len(plan.root_ids), panel.shape[0] * panel.shape[1]), np.float64)
        status = self._lane_fn(
            *args, self._ptr(target), output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        if status:
            raise RuntimeError(f"fe_score_plan_lanes_cpu failed with status {status}")
        return output

    def accumulate_rank_ic_stats(self, plan, panel, target, stats):
        """把接下来的 lane 依次累加到调用方提供的 float64 总和与计数中。

        需要动态库提供累计接口。不能用批均值代替，否则分批方式会影响舍入结果。
        """
        if (
            not isinstance(stats, np.ndarray)
            or stats.dtype != np.float64
            or stats.shape != (len(plan.root_ids), SUM_COUNT_WIDTH)
            or not stats.flags.c_contiguous
            or not stats.flags.writeable
        ):
            raise ValueError("stats must be writable contiguous FP64 [roots,2]")
        counts = stats[:, SumCount.COUNT]
        if (
            not np.isfinite(stats).all()
            or np.any(counts < 0)
            or np.any(counts != np.floor(counts))
            or np.any(np.abs(stats[:, SumCount.SUM]) > counts + CORRELATION_TOLERANCE * counts)
        ):
            raise ValueError("invalid cumulative lane sum/count")
        target = np.ascontiguousarray(target, dtype=np.float32)
        if target.shape != panel.shape:
            raise ValueError("target must match panel shape")
        args, fields = self._prepare(plan, panel)

        rc = self._accumulate_fn(
            *args,
            self._ptr(target),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc:
            raise RuntimeError(f"fe_score_plan_rank_ic_accumulate_cpu failed with status {rc}")

    def score_metric_lanes(self, plan, panel, targets):
        """一次 DAG 评分全部期限，输出 [root,lane,horizon,PearsonIC/RankIC]。"""
        lanes = panel.shape[0] * panel.shape[1]
        if targets.values.shape[1:] != (lanes, panel.shape[2]):
            raise ValueError("multi-target shape differs from panel")
        values = np.ascontiguousarray(targets.values, dtype=np.float32)
        eligible = np.ascontiguousarray(targets.eligible, dtype=np.bool_)
        args, fields = self._prepare(plan, panel)
        output = np.empty(
            (len(plan.root_ids), lanes, len(targets.horizons), METRIC_COUNT), dtype=np.float64
        )
        status = self._metric_fn(
            *args,
            self._ptr(values),
            eligible.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(targets.horizons),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status:
            raise RuntimeError(f"fe_score_plan_metrics_cpu failed with status {status}")
        return output

    def pearson_lanes(self, factors, target, eligible=None):
        """直接调用原生 Pearson；不调用 NumPy oracle，输入为 FP32 [root,lane,time]。"""
        factors = np.ascontiguousarray(factors, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        if factors.ndim != 3 or min(factors.shape) < 1 or target.shape != factors.shape[1:]:
            raise ValueError("Pearson requires factors[root,lane,time] and target[lane,time]")
        mask = None if eligible is None else np.ascontiguousarray(eligible, dtype=np.bool_)
        if mask is not None and mask.shape != target.shape:
            raise ValueError("eligible must match target")
        roots, lanes, time = factors.shape
        pointers = (FloatPointer * roots)(*(self._ptr(row) for row in factors))
        output = np.empty((roots, lanes), dtype=np.float64)
        status = self._pearson_fn(
            pointers,
            roots,
            self._ptr(target),
            None if mask is None else mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            lanes,
            time,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status:
            raise RuntimeError(f"fe_pearson_ic_lanes_cpu failed with status {status}")
        return output

    def cpu_layout(self):
        layout = CPULayout()
        if self._layout_fn(ctypes.byref(layout)):
            raise RuntimeError("native CPU layout query failed")
        result = {name: int(getattr(layout, name)) for name, *_ in layout._fields_}
        if (layout.float_bytes, layout.double_bytes, layout.int_bytes) != (
            4,
            8,
            4,
        ) or layout.max_threads < 1:
            raise RuntimeError("unsupported native CPU layout; rebuild the current library")
        return result

    def close(self):
        """释放本对象的 DAG 缓冲区；OpenMP 线程的排序空间由运行库持有。"""
        self._workspace = np.empty(0, dtype=np.float32)
        self.plans.clear()
