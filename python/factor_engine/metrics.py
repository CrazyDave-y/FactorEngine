"""搜索与后端共享的不可变多期限原始 Sum/Count 合同。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from .metric_types import (
    SumCount,
    SUM_COUNT_WIDTH,
    CORRELATION_TOLERANCE,
    MAX_EXACT_DOUBLE_INTEGER,
    METRIC_COUNT,
)


def _real_array(values):
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError("real numeric values required")
    return np.asarray(array, dtype=np.float64)


def _immutable_array(values):
    # bytes 底层不可写，调用方也不能通过 setflags 解除保护。
    array = np.asarray(values, dtype=np.float64, order="C")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True, eq=False)
class MetricStats:
    """拥有不可变 FP64 [Root,Horizon,Metric,SumCount]，零计数均值为 NaN。"""

    horizons: Sequence[int]
    stats: np.ndarray

    def __post_init__(self):
        horizons = tuple(self.horizons)
        if (
            not horizons
            or any(
                isinstance(h, (bool, np.bool_)) or not isinstance(h, Integral) or h < 1
                for h in horizons
            )
            or len(set(horizons)) != len(horizons)
        ):
            raise ValueError("ordered unique positive integer horizons required")
        horizons = tuple(int(h) for h in horizons)
        values = _real_array(self.stats)
        if (
            values.ndim != 4
            or values.shape[0] < 1
            or values.shape[1:] != (len(horizons), METRIC_COUNT, SUM_COUNT_WIDTH)
        ):
            raise ValueError("nonempty FP64 [roots,horizons,metrics,2] stats required")
        sums = values[..., SumCount.SUM]
        counts = values[..., SumCount.COUNT]
        if (
            not np.isfinite(values).all()
            or np.any(counts < 0)
            or np.any(counts > MAX_EXACT_DOUBLE_INTEGER)
            or np.any(counts != np.floor(counts))
            or np.any((counts == 0) & (sums != 0))
        ):
            raise ValueError("finite sums and exact nonnegative counts required")
        if np.any(np.abs(sums) > counts * (1 + CORRELATION_TOLERANCE)):
            raise ValueError("correlation totals outside their mathematical ranges")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "stats", _immutable_array(values))

    @property
    def means(self) -> np.ndarray:
        return np.divide(
            self.stats[..., SumCount.SUM],
            self.stats[..., SumCount.COUNT],
            out=np.full(self.stats.shape[:-1], np.nan, dtype=np.float64),
            where=self.stats[..., SumCount.COUNT] > 0,
        )


@dataclass(frozen=True, slots=True)
class ScoreStats:
    stats: np.ndarray
    run_stats: dict = field(default_factory=dict)

    def __post_init__(self):
        values = np.array(self.stats, dtype=np.float64, order="C", copy=True)
        if values.ndim != 2 or values.shape[1] != SUM_COUNT_WIDTH or not len(values):
            raise ValueError("scores require nonempty FP64 [roots,2] totals")
        counts = values[:, SumCount.COUNT]
        if (
            not np.isfinite(values).all()
            or np.any(counts < 0)
            or np.any(counts > MAX_EXACT_DOUBLE_INTEGER)
            or np.any(counts != np.floor(counts))
            or np.any(np.abs(values[:, SumCount.SUM]) > counts + CORRELATION_TOLERANCE * counts)
        ):
            raise ValueError("invalid RankIC sums/counts")
        object.__setattr__(self, "stats", _immutable_array(values))
        object.__setattr__(self, "run_stats", dict(self.run_stats))

    @property
    def raw_ic(self):
        return np.divide(
            self.stats[:, SumCount.SUM],
            self.stats[:, SumCount.COUNT],
            out=np.full(len(self.stats), np.nan),
            where=self.stats[:, SumCount.COUNT] > 0,
        )


class LaneAccumulator:
    """每条 lane 按固定顺序累计。分批方式不改变 FP64 加法顺序。"""

    def __init__(self, roots: int, lanes: int):
        if (
            type(roots) is not int
            or roots < 1
            or type(lanes) is not int
            or not 1 <= lanes <= MAX_EXACT_DOUBLE_INTEGER
        ):
            raise ValueError("positive roots and exact lane count required")
        self.lanes = lanes
        self.stats = np.zeros((roots, SUM_COUNT_WIDTH), dtype=np.float64)
        self.next_lane = 0
        self.invalid = False

    def invalidate(self):
        self.invalid = True
        self.stats.fill(np.nan)

    def commit(self, first_lane, lane_ic):
        if self.invalid:
            raise RuntimeError("invalid score accumulator")
        values = np.asarray(lane_ic)
        if (
            type(first_lane) is not int
            or first_lane != self.next_lane
            or values.dtype != np.float64
            or values.ndim != 2
            or values.shape[0] != len(self.stats)
            or values.shape[1] < 1
            or first_lane + values.shape[1] > self.lanes
        ):
            self.invalidate()
            raise ValueError("ordered complete lane window required")
        finite = np.isfinite(values)
        if np.any(np.abs(values[finite]) > 1.0 + CORRELATION_TOLERANCE):
            self.invalidate()
            raise ValueError("lane IC outside correlation range")
        # 每个 root 仍按 lane 顺序相加；只把互不依赖的 root 一起处理。
        # 不使用 sum/cumsum，避免分批方式改变 FP64 归约顺序。
        for lane in range(values.shape[1]):
            active = finite[:, lane]
            self.stats[active, 0] += values[active, lane]
            self.stats[active, 1] += 1.0
        self.next_lane += values.shape[1]

    def finish(self, run_stats=None):
        if self.invalid:
            raise RuntimeError("invalid score accumulator")
        if self.next_lane != self.lanes:
            raise ValueError("complete Lane output has not been consumed")
        if np.any(self.stats[:, SumCount.COUNT] > self.lanes):
            raise ValueError("finite lane count exceeds the complete source")
        return ScoreStats(self.stats, {} if run_stats is None else run_stats)


class BatchTelemetry:
    """累加各批次的实际传输计数、发射数及工作区峰值。"""

    SUM_FIELDS = frozenset(
        (
            "h2d_bytes",
            "kernel_launches",
            "root_export_bytes",
            "input_target_h2d_bytes",
            "lane_ic_d2h_bytes",
            "upload_h2d_bytes",
            "sanitize_launches",
        )
    )
    PEAK_FIELDS = frozenset(("owned_device_bytes", "rank_workspace_bytes"))

    def __init__(self):
        self.summary = {}
        self.batches = 0

    def add(self, counters):
        for name in self.SUM_FIELDS:
            self.summary[name] = self.summary.get(name, 0) + int(counters.get(name, 0))
        for name in self.PEAK_FIELDS:
            self.summary[name] = max(self.summary.get(name, 0), int(counters.get(name, 0)))
        self.batches += 1
