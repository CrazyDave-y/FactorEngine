"""持有 GPU 输入与工作区，释放时保留共享上下文。"""

from __future__ import annotations

import ctypes as C
import numpy as np

from .cuda_native import CUDANative, MetricResources, MetricCapacity, F64P, U8P, input_payload_bytes
from .native_abi import FloatPointer
from .metric_types import CudaStatus, METRIC_COUNT, SUM_COUNT_WIDTH
from .metrics import MetricStats, LaneAccumulator, BatchTelemetry


class CUDAReleaseError(RuntimeError):
    """释放失败时保留实际 owner，调用方仍可重试 close。"""

    def __init__(self, owner, cause):
        super().__init__(f"CUDA close failure; owner retained: {cause}")
        self.owner = owner


def validate_window(capacity, first, width, *, max_workspace_lanes=None):
    if max_workspace_lanes is None:
        max_workspace_lanes = capacity
    if (
        any(type(v) is not int for v in (capacity, first, width, max_workspace_lanes))
        or capacity < 1
        or first < 0
        or width < 1
        or not 1 <= max_workspace_lanes <= 2**31 - 1
        or width > max_workspace_lanes
        or first > capacity - width
    ):
        raise ValueError("batch window exceeds input capacity")
    return first, width


class CUDABuffers:
    def __init__(
        self,
        library,
        *,
        field_names,
        capacity_lanes,
        time_length,
        plan,
        budget_bytes=7 * 1024**3,
        reserve_bytes=1024**3,
    ):
        validate_window(capacity_lanes, 0, capacity_lanes)
        if (
            type(budget_bytes) is not int
            or budget_bytes <= 0
            or type(reserve_bytes) is not int
            or reserve_bytes < 0
        ):
            raise ValueError("positive budget and nonnegative reserve required")
        self.native = CUDANative(library)
        self.capacity, self.time = capacity_lanes, time_length
        self.names = tuple(field_names)
        self.plan = plan
        self.plan_nodes = tuple(plan.nodes)
        self.plan_slots = plan.slot_count
        self.root_ids = tuple(plan.root_ids)
        self.budget = budget_bytes
        self.reserve_bytes = reserve_bytes
        self.metric_prepared = False
        self._pending_requirement = None
        self.payload_bytes = input_payload_bytes(len(self.names), self.capacity, self.time)
        self.required_bytes = self.native.shape_requirement(
            self.names, self.capacity, self.time, plan
        )
        available = self.native.free_bytes()
        if self.required_bytes > budget_bytes or self.required_bytes > available - reserve_bytes:
            raise MemoryError("CUDA inputs and workspace exceed memory budget or free reserve")
        self.handle = None
        self.batch_width = self.active_lanes = self.upload_bytes = 0
        self.failed = False

        self.last_counter = None
        self._resident_source = None
        try:
            self.handle = self.native.allocate(self.names, self.capacity, self.time, self.budget)
            if self.native.input_bytes(self.handle) != self.payload_bytes:
                raise RuntimeError("CUDA input allocation size differs")
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _check(self, plan=None):
        if self.handle is None:
            raise RuntimeError("Generic CUDA owner is closed")
        if self.failed:
            raise RuntimeError("Generic CUDA owner invalid after upload/score failure")
        if plan is not None and plan is not self.plan:
            raise ValueError("Generic CUDA prepared plan identity differs")

    def _prepare(self, width, plan, *, for_export=False):
        self._check(None if for_export else plan)
        if self.batch_width != width:
            self.native.set_batch_width(self.handle, width)
            self.batch_width = width
            self._pending_requirement = None
        pending = self._pending_requirement
        if not for_export and pending is not None and pending[0] is plan:
            required = pending[1]
        else:
            required = self.native.requirement(self.handle, plan, for_export=for_export)
        self._pending_requirement = None
        self.workspace_required_bytes = required
        if self.payload_bytes + required > self.budget:
            if for_export:
                raise MemoryError("CUDA factor output workspace exceeds memory budget")
            raise MemoryError("CUDA workspace exceeds memory budget")
        return self.handle

    def rebind_plan(self, plan):
        """更新绑定的计划，保持输入布局；已失败的设备 owner 不能重新使用。"""
        self._check()
        required = self.native.requirement(self.handle, plan)
        if self.payload_bytes + required > self.budget:
            raise MemoryError("CUDA retained workspace exceeds memory budget")
        if self.last_counter is not None:
            growth = max(0, self.payload_bytes + required - self.last_counter["owned_device_bytes"])
            if growth > self.native.free_bytes() - self.reserve_bytes:
                raise MemoryError("CUDA retained workspace growth exceeds free reserve")
        self.plan = plan
        self.plan_nodes = tuple(plan.nodes)
        self.plan_slots = plan.slot_count
        self.root_ids = tuple(plan.root_ids)
        self._pending_requirement = (plan, required)

    def _upload(self, first, panel, target, *, final):
        self._check()
        self._resident_source = None
        try:
            self.active_lanes = 0
            lanes = panel.shape[0] * panel.shape[1]
            validate_window(self.capacity, first, lanes, max_workspace_lanes=self.capacity)
            counters = self.native.upload(self.handle, first, panel, target, final=final)
            expected = input_payload_bytes(len(self.names), lanes, self.time)
            if counters["upload_h2d_bytes"] != expected:
                raise RuntimeError(
                    "Generic CUDA measured upload bytes differ from complete inputs/target"
                )
            self.upload_bytes += counters["upload_h2d_bytes"]
            if final:
                self.active_lanes = first + lanes
            return counters
        except BaseException:
            self.failed = True
            raise

    def _score_window(self, first, width, plan):
        try:
            self._check(plan)
            validate_window(self.active_lanes, first, width, max_workspace_lanes=self.capacity)
            values, counters = self.native.score_lanes(
                self._prepare(width, plan), first, width, plan
            )
            if (
                counters["input_target_h2d_bytes"] != 0
                or counters["lane_ic_d2h_bytes"] != len(plan.root_ids) * width * 8
                or values.shape != (len(plan.root_ids), width)
            ):
                raise RuntimeError("Generic CUDA resident transfer/output contract differs")
            self.last_counter = counters
            return values
        except BaseException:
            self.failed = True
            raise

    def score_stats(self, source, plan):
        self._check(plan)
        if (
            source.time != self.time
            or tuple(source.field_names) != self.names
            or tuple(plan.root_ids) != self.root_ids
        ):
            self.failed = True
            raise ValueError("Generic CUDA source/plan shape differs")
        accumulator = LaneAccumulator(len(self.root_ids), source.lane_count)
        telemetry = BatchTelemetry()
        reuse_input = source is self._resident_source
        cursor = 0
        try:
            for batch in source.iter_batches(0, self.capacity):
                if batch.first_lane != cursor:
                    raise ValueError("Generic CUDA source Lane order differs")
                width = batch.panel.shape[0] * batch.panel.shape[1]
                uploaded = (
                    {"upload_h2d_bytes": 0, "sanitize_launches": 0}
                    if reuse_input
                    else self._upload(0, batch.panel, batch.target, final=True)
                )
                values = self._score_window(0, width, plan)
                if self.last_counter is None:
                    raise RuntimeError("CUDA scoring completed without counters")
                self.last_counter.update(uploaded)
                telemetry.add(self.last_counter)
                accumulator.commit(cursor, values)
                cursor += width
            run_stats = {
                "lanes": source.lane_count,
                "batches": telemetry.batches,
                "rank_ic_sums": accumulator.stats[:, 0].tolist(),
                "finite_lane_counts": accumulator.stats[:, 1].tolist(),
                "counter_summary": telemetry.summary,
                "accumulation": "canonical-lane-order-fp64",
                "input_reused": reuse_input,
            }
            result = accumulator.finish(run_stats)
            from .session_source import ArraySource

            # 仅绑定能完整放入一个 owner 的只读数组快照；流式数据仍逐批上传。
            if isinstance(source, ArraySource) and source.lane_count <= self.capacity:
                self._resident_source = source
            return result
        except BaseException:
            accumulator.invalidate()
            self.failed = True
            raise

    def evaluate(self, first, width, plan):
        self._check()
        if (
            tuple(plan.nodes) != self.plan_nodes
            or plan.slot_count != self.plan_slots
            or any(root not in self.root_ids for root in plan.root_ids)
        ):
            raise ValueError("factor export must retain original nodes, slots and roots")
        validate_window(self.active_lanes, first, width, max_workspace_lanes=self.capacity)
        return self.native.evaluate(self._prepare(width, plan, for_export=True), first, width, plan)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            raise exc from cleanup_error

    def close(self):
        if self.handle is not None:
            # 原生释放先同步；失败时保留 handle，不能伪称释放成功。
            try:
                self.native.release(self.handle)
            except BaseException as error:
                self.failed = True
                raise CUDAReleaseError(self, error) from error
            self.handle = None
            self.active_lanes = 0
            self._resident_source = None

    def prepare_metrics(self, plan, horizons):
        self._check(plan)
        self._prepare(self.capacity, plan)
        required = self.workspace_required_bytes
        available = self.budget - self.payload_bytes - required
        current = (
            self.metric_resources()
            if self.metric_prepared
            else {"workspace_bytes": 0, "owned_device_bytes": self.payload_bytes}
        )
        runtime_bytes = (
            current["owned_device_bytes"] - self.payload_bytes - current["workspace_bytes"]
        )
        runtime_growth = max(0, required - runtime_bytes)
        # free 已扣除现有分配；仅允许新分配消耗 reserve 以上的剩余空间。
        free_limit = (
            self.native.free_bytes()
            - self.reserve_bytes
            + current["workspace_bytes"]
            - runtime_growth
        )
        available = min(available, free_limit)
        result = MetricResources()
        status = self.native.lib.fe_cuda_metric_prepare(
            self.handle,
            len(plan.root_ids),
            len(horizons),
            available,
            C.byref(result),
        )
        if status == CudaStatus.ALLOCATION_FAILED:
            raise MemoryError("multi-target workspace exceeds memory budget or free memory")
        if status:
            raise RuntimeError(f"multi-target prepare failed: {status}")
        self.metric_prepared = True
        return {name: int(getattr(result, name)) for name, *_ in result._fields_}

    def metric_resources(self):
        self._check()
        result = MetricResources()
        status = self.native.lib.fe_cuda_metric_resources(self.handle, C.byref(result))
        if status:
            raise RuntimeError(f"multi-target resource query failed: {status}")
        return {name: int(getattr(result, name)) for name, *_ in result._fields_}

    def metric_capacity(self):
        """查询当前分配、保留容量与扩容峰值。"""
        self._check()
        function = self.native.lib.fe_cuda_metric_capacity
        result = MetricCapacity()
        status = function(self.handle, C.byref(result))
        if status:
            raise RuntimeError(f"multi-target capacity query failed: {status}")
        return {name: int(getattr(result, name)) for name, *_ in result._fields_}

    def metric_inputs_ready(self):
        """查询当前目标是否已上传；原生 owner 在扩容或布局变化时使上传状态失效。"""
        self._check()
        function = self.native.lib.fe_cuda_metric_inputs_ready
        result = C.c_int()
        status = function(self.handle, C.byref(result))
        if status:
            self.failed = True
            raise RuntimeError(f"multi-target input query failed: {status}")
        return bool(result.value)

    def score_metrics(
        self,
        accumulator,
        plan,
        batch,
        targets,
        *,
        detail,
        reuse_input=False,
        reuse_targets=False,
    ):
        width = batch.panel.shape[0] * batch.panel.shape[1]
        if targets.values.shape != (len(targets.horizons), width, self.time):
            raise ValueError("multi-target batch shape differs")
        self._check(plan)
        if not reuse_input:
            self._upload(0, batch.panel, batch.target, final=True)
        status = 0
        if not reuse_targets:
            status = self.native.lib.fe_cuda_metric_upload(
                self.handle,
                targets.values.ctypes.data_as(FloatPointer),
                targets.eligible.ctypes.data_as(U8P),
            )
        if status:
            self.failed = True
            raise RuntimeError(f"multi-target upload failed: {status}")
        prepared = self.native.plans.get(plan, self.names, (1, width, self.time), cuda=True)
        output = (
            np.empty((len(plan.root_ids), width, len(targets.horizons), METRIC_COUNT), np.float64)
            if detail
            else None
        )
        status = self.native.lib.fe_cuda_metric_score(
            self.handle,
            accumulator.handle,
            batch.first_lane,
            prepared.nodes,
            len(plan.nodes),
            plan.slot_count,
            prepared.roots,
            len(plan.root_ids),
            None if output is None else output.ctypes.data_as(F64P),
        )
        if status:
            self.failed = accumulator.failed = True
            raise RuntimeError(f"multi-target score failed: {status}")
        return output

    def read_metrics(self, plan, source):
        """读取 [root,horizon,PearsonIC/RankIC,Sum/Count]，不导出辅助列。"""
        self._check(plan)
        stats = np.empty(
            (len(plan.root_ids), len(source.targets.horizons), METRIC_COUNT, SUM_COUNT_WIDTH),
            np.float64,
        )
        status = self.native.lib.fe_cuda_metric_read(
            self.handle, source.lane_count, stats.ctypes.data_as(F64P)
        )
        if status:
            self.failed = True
            raise RuntimeError(f"multi-target result failed: {status}")
        return MetricStats(source.targets.horizons, stats)
