"""按完整时间序列上传和评分，管理本后端的输入和计算工作区。"""

from pathlib import Path
from contextlib import contextmanager

from .metric_types import NativeLimit
from .cuda_buffers import CUDABuffers, CUDAReleaseError
from .session_source import ArraySource
from .native_abi import PlanCache


def _metric_workspace_key(plan, source, ranges):
    # 工作区只绑定计划和布局；流式调用每批上传，会话内才复用输入。
    return (
        plan,
        source.field_names,
        source.lane_count,
        source.time,
        source.targets.horizons,
        ranges,
    )


class CUDABackend:
    name = "generic_cuda"

    def __init__(
        self,
        library,
        *,
        batch_lanes: int | None = int(NativeLimit.CUDA_MAX_BATCH_LANES),
        memory_budget_bytes: int | None = 256 * 1024**2,
        reserve_bytes=1024**3,
    ):
        if batch_lanes is not None and (
            type(batch_lanes) is not int or not 1 <= batch_lanes <= 2**31 - 1
        ):
            raise ValueError("batch_lanes must be a positive int32 or None for automatic planning")
        self.library = Path(library).resolve(strict=True)
        self.max_batch_lanes = batch_lanes
        self.memory_budget_bytes = memory_budget_bytes
        self.reserve_bytes = reserve_bytes

        self.last_run_stats = None
        self.closed = False
        self._single_buffer: CUDABuffers | None = None
        self._metric_buffers: dict[int, CUDABuffers] = {}
        self._metric_key = None
        self._metric_storage_key = None
        self._prepared_plans = PlanCache()
        self.last_metric_details = None
        self._metric_resources = {}
        self.last_chunk_plan = None
        self._input_session = None

    @contextmanager
    def multitarget_input_session(self, source):
        """本次搜索独占后端；每个批区间持有独立输入缓冲，受同一总预算约束。"""
        if self.closed:
            raise RuntimeError("CUDA backend is closed")
        if self._input_session is not None:
            raise RuntimeError("CUDA input session already bound")
        state = {
            "source": source,
            "base": source.source,
            "targets": source.targets,
            "bindings": {},
            "ranges": None,
            "info": {
                "enabled": True,
                "mode": "resident_batches",
                "input_uploads": 0,
                "target_uploads": 0,
                "input_h2d_bytes": 0,
                "target_h2d_bytes": 0,
            },
        }
        self._input_session = state
        try:
            yield state["info"]
            self._check_input_session(source)
        finally:
            # 保留已分配空间，但搜索结束后清除输入绑定。
            self._input_session = None
            self._metric_key = None

    def _check_input_session(self, source):
        session = self._input_session
        if session is not None and (
            source is not session["source"]
            or source.source is not session["base"]
            or source.targets is not session["targets"]
        ):
            raise ValueError("source or targets changed during CUDA input session")
        return session if session is not None and session["info"]["enabled"] else None

    def score(self, plan, source):
        self.last_run_stats = None
        if self.closed:
            raise RuntimeError("CUDA backend is closed")
        if self._input_session is not None:
            raise RuntimeError(
                "single-target scoring cannot replace an active multi-target session"
            )
        # 单期限与多期限共用同一个预算，切回单期限接口时先释放缓存。
        self._close_metric_buffers()
        self._metric_key = None
        if self._single_buffer is not None and self._single_buffer.failed:
            raise RuntimeError("CUDA cleanup pending; retry backend.close() before more work")
        max_lanes = self.max_batch_lanes
        budget = self.memory_budget_bytes
        if max_lanes is None or budget is None:
            automatic = True
            from .cuda_native import CUDANative

            self.last_chunk_plan = choose_lane_chunk(
                CUDANative(self.library),
                plan,
                source.field_names,
                source.time,
                source.lane_count,
                budget_bytes=budget,
                reserve_bytes=self.reserve_bytes,
                max_lanes=max_lanes,
                host_budget_bytes=getattr(source, "max_batch_bytes", None),
            )
            width = self.last_chunk_plan["lanes"]
            budget = self.last_chunk_plan["budget_bytes"]
        else:
            automatic = False
            width = min(max_lanes, source.lane_count)

        retries = []
        while True:
            try:
                result = self._score_width(plan, source, width, budget)
                break
            except MemoryError as error:
                if not automatic or width <= 1 or self._single_buffer is not None:
                    raise
                retries.append({"lanes": width, "error": str(error)})
                width = max(1, width // 2)

        result.run_stats["allocation_retries"] = retries
        result.run_stats["batch_capacity"] = width
        result.run_stats["execution_node_count"] = len(plan.nodes)
        result.run_stats["execution_slot_count"] = plan.slot_count
        # 只有完整重跑成功才发布；失败的局部累加器已失效，不会重复提交。
        self.last_run_stats = result.run_stats
        return result

    def _score_width(self, plan, source, width, budget):
        owner = self._single_buffer
        reused = owner is not None and (
            owner.names == tuple(source.field_names)
            and owner.capacity == width
            and owner.time == source.time
            and owner.budget == budget
        )
        if owner is not None and not reused:
            self._close_single_buffer()
            owner = None
        try:
            if owner is None:
                owner = CUDABuffers(
                    self.library,
                    field_names=source.field_names,
                    capacity_lanes=width,
                    time_length=source.time,
                    plan=plan,
                    budget_bytes=budget,
                    reserve_bytes=self.reserve_bytes,
                )
                self._single_buffer = owner
            elif owner.plan is not plan:
                owner.rebind_plan(plan)
            result = owner.score_stats(source, plan)
            result.run_stats["workspace_reused"] = reused
        except BaseException as error:
            if isinstance(error.__cause__, CUDAReleaseError):
                self._single_buffer = error.__cause__.owner
            try:
                self._close_single_buffer()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        return result

    def _close_single_buffer(self):
        if self._single_buffer is not None:
            try:
                self._single_buffer.close()
            except BaseException:
                self._single_buffer.failed = True
                raise
            self._single_buffer = None

    def prepare_multitarget(self, plan, source):
        if self.max_batch_lanes is None or self.max_batch_lanes > NativeLimit.CUDA_MAX_BATCH_LANES:
            raise ValueError("multi-target scoring retains its existing 1..32 Lane contract")
        if self.memory_budget_bytes is None:
            raise ValueError("multi-target scoring requires its existing explicit budget")
        if self.closed:
            raise RuntimeError("CUDA backend is closed")
        if self._single_buffer is not None and self._single_buffer.failed:
            raise RuntimeError("CUDA cleanup pending; retry backend.close() before more work")
        self._close_single_buffer()
        session = self._check_input_session(source)
        ranges = tuple(source.batch_ranges(0, self.max_batch_lanes))
        if not ranges:
            raise ValueError("multi-target source requires nonempty batch ranges")
        if session is not None:
            if session["ranges"] is not None and session["ranges"] != ranges:
                raise ValueError("source batch ranges changed during CUDA input session")
            session["ranges"] = ranges
        key = _metric_workspace_key(plan, source, ranges)
        if session is None:
            # 流式模式为每种批宽保留一个设备缓冲 owner。
            widths = {width for _, width in ranges}
            specs = [(width, width) for width in sorted(widths, reverse=True)]
        else:
            # 常驻模式为每个输入批次保留缓冲，供不同代复用。
            specs = list(ranges)
        total_width = sum(width for _, width in specs)
        allocations = [
            (batch_key, width, self.memory_budget_bytes * width // total_width)
            for batch_key, width in specs
        ]
        storage_key = (
            session is not None,
            tuple(source.field_names),
            source.time,
            tuple(allocations),
        )
        if self._metric_storage_key != storage_key:
            self._close_metric_buffers()
            self._metric_storage_key = storage_key
        # 所有 owner 准备完成后，才记录本次计划和输入布局。
        self._metric_key = None
        resources = {}
        try:
            for batch_key, width, budget in allocations:
                if batch_key not in self._metric_buffers:
                    # 按批宽为各 owner 分配总预算。
                    try:
                        owner = CUDABuffers(
                            self.library,
                            field_names=source.field_names,
                            capacity_lanes=width,
                            time_length=source.time,
                            plan=plan,
                            budget_bytes=budget,
                            reserve_bytes=self.reserve_bytes,
                        )
                    except BaseException as error:
                        if isinstance(error.__cause__, CUDAReleaseError):
                            self._metric_buffers[batch_key] = error.__cause__.owner
                        raise
                    self._metric_buffers[batch_key] = owner
                    # 复用同一份不可变 ABI 数组，完整批和尾批不重复编码。
                    owner.native.plans = self._prepared_plans
                owner = self._metric_buffers[batch_key]
                if owner.plan is not plan:
                    owner.rebind_plan(plan)
                resources[batch_key] = owner.prepare_metrics(plan, source.targets.horizons)

            # Runtime 在 score 时才申请内存，须合计预留各缓冲 owner 尚未分配的增量。
            pending_runtime = 0
            for batch_key, owner in self._metric_buffers.items():
                resource = resources[batch_key]
                retained_runtime = (
                    resource["owned_device_bytes"]
                    - owner.payload_bytes
                    - resource["workspace_bytes"]
                )
                pending_runtime += max(0, owner.workspace_required_bytes - retained_runtime)
            if pending_runtime > owner.native.free_bytes() - self.reserve_bytes:
                raise MemoryError("all CUDA Runtime growth exceeds free memory reserve")
        except MemoryError:
            self._metric_key = None
            if session is None:
                for owner in self._metric_buffers.values():
                    owner.failed = True
                raise
            # 常驻优化不应缩小原流式接口可处理的数据规模；明确记录模式切换。
            # 只在准备阶段完整释放后重试，绝不重放已提交的批次。
            self._close_metric_buffers()
            self._metric_storage_key = None
            session["bindings"].clear()
            session["info"].update(enabled=False, mode="streaming", reason="resident_memory_budget")
            return CUDABackend.prepare_multitarget(self, plan, source)
        except BaseException:
            self._metric_key = None
            for owner in self._metric_buffers.values():
                owner.failed = True
            raise
        self._metric_key = key
        self._metric_resources = {
            "algorithm": "fused",
            "batches": resources,
        }

    def score_multitarget(self, plan, source, *, detail=False):
        import numpy as np

        if self.closed:
            raise RuntimeError("CUDA backend is closed")
        ranges = tuple(source.batch_ranges(0, self.max_batch_lanes))
        key = _metric_workspace_key(plan, source, ranges)
        if (
            self._metric_key is None
            or key[0] is not self._metric_key[0]
            or key[1:] != self._metric_key[1:]
        ):
            raise ValueError("prepare_multitarget is required for this Plan and targets")
        self.last_metric_details = None
        session = self._check_input_session(source)
        accumulator = next(iter(self._metric_buffers.values()))
        outputs = []
        cursor = 0
        for index, (batch, targets) in enumerate(source.iter_batches(0, self.max_batch_lanes)):
            width = batch.panel.shape[0] * batch.panel.shape[1]
            if index >= len(ranges) or (batch.first_lane, width) != ranges[index]:
                raise ValueError("multi-target source batches differ from prepared batch_ranges")
            owner = self._metric_buffers[cursor if session is not None else width]
            reuse_input = session is not None and session["bindings"].get(owner) == cursor
            reuse_targets = False
            if reuse_input:
                reuse_targets = owner.metric_inputs_ready()
            values = owner.score_metrics(
                accumulator,
                plan,
                batch,
                targets,
                detail=detail,
                reuse_input=reuse_input,
                reuse_targets=reuse_targets,
            )
            if session is not None:
                session["bindings"][owner] = cursor
            if self._input_session is not None:
                info = self._input_session["info"]
                info["input_uploads"] += int(not reuse_input)
                info["target_uploads"] += int(not reuse_targets)
                info["input_h2d_bytes"] += 0 if reuse_input else owner.payload_bytes
                info["target_h2d_bytes"] += 0 if reuse_targets else targets.values.size * 5
            if detail:
                outputs.append(values)
            cursor += width
        if cursor != source.lane_count:
            raise ValueError("multi-target source did not provide all lanes")
        stats = accumulator.read_metrics(plan, source)
        if detail:
            self.last_metric_details = np.concatenate(outputs, axis=1)
        return stats

    def multitarget_resources(self, *, capacity=False):
        if self.closed:
            raise RuntimeError("CUDA backend is closed")
        result = dict(self._metric_resources)
        resident = self._metric_storage_key is not None and self._metric_storage_key[0]
        result["batch_index"] = "first_lane" if resident else "capacity_lanes"
        result["batches"] = {
            key: {**owner.metric_resources(), "capacity_lanes": owner.capacity}
            for key, owner in self._metric_buffers.items()
        }
        result["owned_device_bytes"] = sum(
            value["owned_device_bytes"] for value in result["batches"].values()
        )
        if capacity:
            for width, owner in self._metric_buffers.items():
                result["batches"][width].update(owner.metric_capacity())
            result["transient_peak_device_bytes"] = result["owned_device_bytes"] + max(
                (
                    max(0, value["prepare_peak_owned_bytes"] - value["owned_device_bytes"])
                    for value in result["batches"].values()
                ),
                default=0,
            )
        return result

    def _close_metric_buffers(self):
        self._metric_key = None
        errors = []
        for width, owner in list(self._metric_buffers.items()):
            try:
                owner.close()
                del self._metric_buffers[width]
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self._metric_storage_key = None
        self._metric_resources = {}
        self.last_metric_details = None

    def score_rank_ic_source(self, plan, source):
        return self.score(plan, source).raw_ic

    def score_rank_ic(self, plan, panel, target):
        self.last_run_stats = None
        source = ArraySource(panel, target, source_identity="in-memory-call")
        return self.score(plan, source).raw_ic

    def synchronize(self):
        # 公共评分 API 返回前已经同步。
        if self.closed:
            raise RuntimeError("CUDA backend is closed")
        if self._single_buffer is not None and self._single_buffer.failed:
            raise RuntimeError("CUDA cleanup pending; retry backend.close() before more work")

    def close(self):
        self.last_run_stats = None
        errors = []
        for close in (self._close_metric_buffers, self._close_single_buffer):
            try:
                close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self.closed = True
        self._prepared_plans.clear()


_DEFAULT_LANE_CANDIDATES = (32, 64, 128, 256, 384, 448, 512, 640, 768, 1024)


def choose_lane_chunk(
    native,
    plan,
    field_names,
    time_length,
    lane_count,
    *,
    budget_bytes=None,
    reserve_bytes=1024**3,
    max_lanes=None,
    host_budget_bytes=None,
    candidates=_DEFAULT_LANE_CANDIDATES,
) -> dict:
    """返回最大资源可行容量及逐候选依据；性能选型仍须固定工作量实测。"""
    for name, value in (("time_length", time_length), ("lane_count", lane_count)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("budget_bytes", budget_bytes),
        ("max_lanes", max_lanes),
        ("host_budget_bytes", host_budget_bytes),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer or None")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    names = tuple(field_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("field_names must contain nonempty field names")
    default_grid = candidates is _DEFAULT_LANE_CANDIDATES
    requested = tuple(candidates)
    if not requested or any(type(lanes) is not int or lanes <= 0 for lanes in requested):
        raise ValueError("candidates must contain positive integer lane counts")
    source_limit = min(lane_count, max_lanes) if max_lanes is not None else lane_count
    widths = sorted({min(width, source_limit) for width in requested})
    device = dict(native.device_info())
    for key in ("free_bytes", "total_bytes", "sm_count", "max_shared_bytes"):
        value = device.get(key)
        if type(value) is not int or value < (0 if key == "free_bytes" else 1):
            raise ValueError(f"invalid device information: {key}")
    if device["free_bytes"] > device["total_bytes"]:
        raise ValueError("invalid device information: free_bytes exceeds total_bytes")
    available = device["free_bytes"] - reserve_bytes
    budget = min(available, budget_bytes) if budget_bytes is not None else available
    # FP32 字段与目标，加每时间点一个有效性字节；尾批复用同一个缓冲 owner。
    host_bytes_per_lane = (len(names) * 4 + 5) * time_length
    records = []
    feasible = []

    def query(width, *, fallback=False):
        required = native.shape_requirement(names, width, time_length, plan)
        if type(required) is not int or required < 0:
            raise ValueError("native shape requirement must be a nonnegative integer")
        host_required = width * host_bytes_per_lane
        status = "FEASIBLE"
        if required > budget:
            status = "RESOURCE_INFEASIBLE"
        elif host_budget_bytes is not None and host_required > host_budget_bytes:
            status = "HOST_BUDGET_INFEASIBLE"
        record = {
            "lanes": width,
            "required_bytes": required,
            "host_required_bytes": host_required,
            "status": status,
            "small_chunk_fallback": fallback,
        }
        records.append(record)
        if status == "FEASIBLE":
            feasible.append(record)

    for width in widths:
        query(width)
    used_fallback = default_grid and not feasible
    if used_fallback:
        # 默认资源扫表没有可行项时，只做有界小容量查询，不创建 owner。
        for candidate in (16, 8, 4, 2, 1):
            width = min(candidate, source_limit)
            if width not in widths:
                widths.append(width)
                query(width, fallback=True)
    if not feasible:
        reasons = ", ".join(
            f"L{item['lanes']}:{item['status']} "
            f"(device={item['required_bytes']},host={item['host_required_bytes']})"
            for item in records
        )
        raise MemoryError(
            f"no feasible CUDA lane capacity: budget={budget}, "
            f"free={device['free_bytes']}, reserve={reserve_bytes}, "
            f"host_budget={host_budget_bytes}; {reasons}"
        )
    selected = max(feasible, key=lambda item: item["lanes"])
    return {
        "lanes": selected["lanes"],
        "required_bytes": selected["required_bytes"],
        "budget_bytes": budget,
        "device": device,
        "candidates": records,
        "small_chunk_fallback": bool(used_fallback),
        "selection": "LARGEST_RESOURCE_FEASIBLE_NOT_THROUGHPUT_OPTIMUM",
    }
