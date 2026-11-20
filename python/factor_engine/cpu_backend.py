"""CPU 后端主接口 score(plan, source) 返回原始 Sum/Count。"""

from pathlib import Path
import numpy as np
from .metric_types import SumCount, SUM_COUNT_WIDTH, METRIC_COUNT, NativeLimit, CpuMemoryAllowance

from .metrics import MetricStats, ScoreStats
from .target_bank import MultiTargetSource

from .native_plan_backend import NativePlanBackend
from .session_source import ArraySource


class CPUBackend:
    name = "native_cpu"

    def __init__(
        self,
        library,
        *,
        workers=2,
        batch_lanes=32,
        memory_budget_bytes=1024**3,
    ):
        if type(memory_budget_bytes) is not int or memory_budget_bytes < 1:
            raise ValueError("memory budget must be a positive integer")
        if type(batch_lanes) is not int or not 1 <= batch_lanes <= NativeLimit.INT32_MAX:
            raise ValueError("batch_lanes must be a positive int32")
        self.backend = NativePlanBackend(Path(library))
        if self.backend.cpu_layout()["max_threads"] != workers:
            self.backend.close()
            raise RuntimeError("OpenMP thread limit differs from configured workers")
        self.max_batch_lanes = batch_lanes
        self.memory_budget_bytes = memory_budget_bytes

        self.last_run_stats = None
        self.closed = False
        self._metric_binding = None
        self.last_metric_details = None

    def _require_open(self):
        if self.closed:
            raise RuntimeError("CPU backend is closed")

    def _policy(self, frame_bytes, snapshot_bytes=0, reserved_bytes=0):
        budget = self.memory_budget_bytes - snapshot_bytes - reserved_bytes
        if budget <= 0:
            raise MemoryError("CPU memory budget cannot admit the input snapshot")
        return CPUMemoryBudget(
            self.backend,
            budget_bytes=budget,
            host_frame_bytes_per_lane=frame_bytes,
        )

    def _score_source(self, plan, source):
        policy = self._policy(source.bytes_per_lane, getattr(source, "snapshot_bytes", 0))
        frame_limit = getattr(source, "max_batch_bytes", self.memory_budget_bytes)
        maximum = min(self.max_batch_lanes, frame_limit // source.bytes_per_lane)
        if maximum < 1:
            raise MemoryError("CPU source frame budget cannot admit one complete lane")

        width = policy.capacity(
            plan, fields=len(source.field_names), time=source.time, maximum=maximum
        )
        batches = source.iter_batches(0, width)
        totals = np.zeros((len(plan.root_ids), SUM_COUNT_WIDTH), np.float64)
        cursor = batches_count = largest_batch = peak_bytes = 0
        peak_estimate = None
        input_bytes = 0
        try:
            for batch in batches:
                panel, target = batch.panel, batch.target
                lanes = panel.shape[0] * panel.shape[1]
                if batch.first_lane != cursor:
                    raise ValueError("source lane ordinal must be contiguous and ordered")
                if (
                    tuple(panel.fields) != tuple(source.field_names)
                    or panel.shape[-1] != source.time
                ):
                    raise ValueError("batch field schema and time length must match")
                if not 0 < lanes <= width or cursor + lanes > source.lane_count:
                    raise ValueError("source batch exceeds the complete lane range")
                estimate = policy.estimate(
                    plan, fields=len(source.field_names), time=source.time, lanes=lanes
                )
                if estimate["total"] > peak_bytes:
                    peak_bytes, peak_estimate = estimate["total"], estimate
                self.backend.accumulate_rank_ic_stats(plan, panel, target, totals)
                input_bytes += sum(value.nbytes for value in panel.fields.values()) + target.nbytes
                cursor += lanes
                batches_count += 1
                largest_batch = max(largest_batch, lanes)
                del panel, target, batch
        finally:
            close = getattr(batches, "close", None)
            if callable(close):
                close()
        if cursor != source.lane_count or batches_count == 0:
            raise ValueError("complete nonempty source required")
        return ScoreStats(
            totals,
            {
                "lanes": cursor,
                "batches": batches_count,
                "max_batch_lanes": largest_batch,
                "input_bytes": input_bytes,
                "compact_result_bytes": totals.nbytes * batches_count,
                "estimated_peak_working_bytes": peak_bytes,
                "cpu_memory_estimate": peak_estimate,
                "rank_ic_sums": totals[:, SumCount.SUM].tolist(),
                "finite_lane_counts": totals[:, SumCount.COUNT].tolist(),
                "accumulation": "canonical-lane-order-fp64-inout",
            },
        )

    def score(self, plan, source):
        self.last_run_stats = None
        self._require_open()
        try:
            result = self._score_source(plan, source)
        except BaseException:
            self.backend.close()
            raise
        self.last_run_stats = result.run_stats
        return result

    def score_rank_ic_source(self, plan, source):
        return self.score(plan, source).raw_ic

    def score_rank_ic(self, plan, panel, target):
        self.last_run_stats = None
        self._require_open()
        snapshot_bytes = ArraySource.required_bytes(panel, target)
        frame_bytes = panel.shape[2] * (4 * len(panel.fields) + 5)
        # 独立 array 调用仍先准入再复制；搜索入口只创建一次 ArraySource。
        self._policy(frame_bytes, snapshot_bytes).capacity(
            plan, fields=len(panel.fields), time=panel.shape[2], maximum=1
        )
        source = ArraySource(
            panel,
            target,
            source_identity="in-memory-call",
            snapshot_budget_bytes=self.memory_budget_bytes,
        )
        return self.score(plan, source).raw_ic

    def _metric_capacity(self, plan, source, *, detail=False):
        targets = source.targets
        roots, horizons = len(plan.root_ids), len(targets.horizons)
        # 计入不可变标签、输出、批内目标副本及累计所需临时量，避免低估多期内存。
        totals_bytes = (
            roots * horizons * METRIC_COUNT * SUM_COUNT_WIDTH * np.dtype(np.float64).itemsize
        )
        detail_bytes = (
            (roots * source.lane_count * horizons * METRIC_COUNT * np.dtype(np.float64).itemsize)
            if detail
            else 0
        )
        reserved = targets.values.nbytes + targets.eligible.nbytes + totals_bytes + detail_bytes
        frame = (
            source.source.bytes_per_lane
            + horizons * source.time * (np.dtype(np.float32).itemsize + np.dtype(np.bool_).itemsize)
            + roots
            * horizons
            * METRIC_COUNT
            * (2 * np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize)
        )
        policy = self._policy(frame, getattr(source.source, "snapshot_bytes", 0), reserved)
        frame_limit = getattr(source.source, "max_batch_bytes", self.memory_budget_bytes)
        maximum = min(
            self.max_batch_lanes, source.lane_count, frame_limit // source.source.bytes_per_lane
        )
        if maximum < 1:
            raise MemoryError("CPU source frame budget cannot admit one complete lane")
        return policy.capacity(
            plan,
            fields=len(source.field_names),
            time=source.time,
            maximum=maximum,
        )

    def prepare_multitarget(self, plan, source):
        self._require_open()
        self._metric_binding = None
        if not isinstance(source, MultiTargetSource):
            raise TypeError("CPU metrics require MultiTargetSource")
        if source.time > NativeLimit.MAX_RANK_TIME:
            raise ValueError("CPU metric integer-rank contract supports Time <= 24576")
        self._metric_capacity(plan, source)
        self._metric_binding = (plan, source)

    def score_multitarget(self, plan, source, *, detail=False):
        """返回 PearsonIC/RankIC 的 Sum/Count；detail 仅保留这两个逐 lane 结果。"""
        self._require_open()
        self.last_metric_details = None
        binding = (plan, source)
        if self._metric_binding is None or any(
            current is not prepared for current, prepared in zip(binding, self._metric_binding)
        ):
            raise ValueError("prepare_multitarget is required for this Plan and targets")
        width = self._metric_capacity(plan, source, detail=detail)
        totals = np.zeros(
            (len(plan.root_ids), len(source.targets.horizons), METRIC_COUNT, SUM_COUNT_WIDTH),
            dtype=np.float64,
        )
        details = (
            np.empty(
                (
                    len(plan.root_ids),
                    source.lane_count,
                    len(source.targets.horizons),
                    METRIC_COUNT,
                ),
                dtype=np.float64,
            )
            if detail
            else None
        )
        cursor = 0
        for batch, targets in source.iter_batches(0, width):
            if batch.first_lane != cursor:
                raise ValueError("ordered complete multi-target batches required")
            values = self.backend.score_metric_lanes(plan, batch.panel, targets)
            lanes = values.shape[1]
            if cursor + lanes > source.lane_count:
                raise ValueError("source batch exceeds lane count")
            # 与 CUDA commit 相同，逐 lane 加入，不能把批小计再相加。
            for lane in range(lanes):
                row = values[:, lane]
                valid = np.isfinite(row)
                totals[..., SumCount.SUM][valid] += row[valid]
                totals[..., SumCount.COUNT][valid] += 1.0
            if details is not None:
                details[:, cursor : cursor + lanes] = values
            cursor += lanes
        if cursor != source.lane_count:
            raise ValueError("complete multi-target source required")
        result = MetricStats(source.targets.horizons, totals)
        self.last_metric_details = details
        return result

    def synchronize(self):
        self._require_open()

    def close(self):
        if not self.closed:
            self._metric_binding = None
            self.last_metric_details = None
            self.backend.close()
            self.closed = True


class CPUMemoryBudget:
    def __init__(
        self,
        backend,
        *,
        budget_bytes,
        host_frame_bytes_per_lane,
        safety_bytes=64 * 1024**2,
    ):
        if any(
            type(v) is not int or v < 1
            for v in (budget_bytes, host_frame_bytes_per_lane, safety_bytes)
        ):
            raise ValueError("CPU budget/frame/safety must be positive integers")
        self.backend = backend
        self.budget_bytes = budget_bytes
        self.host_frame_bytes_per_lane = host_frame_bytes_per_lane
        self.safety_bytes = safety_bytes
        self.layout = backend.cpu_layout()

    def estimate(self, plan, *, fields, time, lanes):
        if min(fields, time, lanes) < 1 or not plan.root_ids:
            raise ValueError("positive CPU scoring dimensions required")
        layout = self.layout
        threads = layout["max_threads"]
        cell = (
            2 * layout["sort_pair_bytes"]
            + 2 * layout["double_bytes"]
            + layout["int_bytes"]
            + CpuMemoryAllowance.TARGET_RANK_CACHE_ENTRIES
            * (layout["int_bytes"] + layout["double_bytes"])
        )
        scratch = (
            self.backend.rank_scratch_retained_bytes
            + CpuMemoryAllowance.RANK_GROWTH_ALLOWANCE * cell * time * threads
            + CpuMemoryAllowance.RANK_THREAD_OVERHEAD_BYTES * threads
        )
        rolling = 0
        # NODE_BARRIER 按节点串行执行，rolling 的局部 vector 在算子返回时释放。
        # 取同时存活峰值；allocator 保留页与碎片由 safety 余量覆盖。
        for node in plan.nodes:
            needed = 0
            if node.op in ("roll_std", "cov", "corr") and int(node.param) <= time:
                family = "unary" if node.op == "roll_std" else "pair"
                needed = threads * (
                    2
                    * (layout[family + "_entry_bytes"] + layout[family + "_moments_bytes"])
                    * int(node.param)
                    + CpuMemoryAllowance.ROLLING_ALLOCATION_OVERHEAD_BYTES
                )
            elif node.op in ("roll_min", "roll_max"):
                needed = threads * (
                    CpuMemoryAllowance.EXTREMA_ENTRY_ALLOWANCE_BYTES * min(time, int(node.param))
                    + CpuMemoryAllowance.ROLLING_ALLOCATION_OVERHEAD_BYTES
                )
            rolling = max(rolling, needed)
        requested = layout["float_bytes"] * plan.slot_count * lanes * time
        retained = self.backend.workspace_capacity_floats * layout["float_bytes"]
        workspace = max(requested, retained) + (retained if requested > retained else 0)
        parts = {
            "source_frames": lanes * self.host_frame_bytes_per_lane * 2,
            "workspace": workspace,
            "sanitized_fields": (fields * 4 + 2) * lanes * time,
            "rank_scratch": scratch,
            "rolling_scratch": rolling,
            "compact_scores": 8 * len(plan.root_ids) * lanes + 32 * len(plan.root_ids),
            "metadata": len(plan.nodes) * CpuMemoryAllowance.NODE_METADATA_ALLOWANCE_BYTES
            + threads * CpuMemoryAllowance.THREAD_METADATA_ALLOWANCE_BYTES,
            "safety": self.safety_bytes,
        }
        parts["total"] = sum(parts.values())
        return parts

    def capacity(self, plan, *, fields, time, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("positive maximum required")
        limit = self.budget_bytes
        lo, hi = 0, maximum
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.estimate(plan, fields=fields, time=time, lanes=mid)["total"] <= limit:
                lo = mid
            else:
                hi = mid - 1
        if lo < 1:
            raise MemoryError("CPU memory budget cannot admit one complete lane")
        return lo
