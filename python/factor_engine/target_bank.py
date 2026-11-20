"""保存多期限标签快照，按原数据源的完整 lane 批次对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator

import numpy as np

from .data import IndexedBatch
from .session_source import IndexedLaneSource


def _horizons(values) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(
        isinstance(h, (bool, np.bool_)) or not isinstance(h, Integral) or h < 1 for h in result
    ):
        raise ValueError("horizons must be nonempty positive integers")
    if len(set(result)) != len(result):
        raise ValueError("horizons must be unique")
    return tuple(int(h) for h in result)


def _snapshot(values, dtype) -> np.ndarray:
    # bytes 持有独立副本，调用方不能重新开启写入或改动原数组来污染标签。
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class TargetBatch:
    """[H,L,T] 标签；eligible 是最终资格，mature 只记录端点成熟度。"""

    horizons: tuple[int, ...]
    values: np.ndarray
    eligible: np.ndarray
    mature: np.ndarray | None = None

    def __post_init__(self) -> None:
        horizons = _horizons(self.horizons)
        values = np.asarray(self.values)
        eligible = np.asarray(self.eligible)
        if values.ndim != 3 or min(values.shape) < 1 or values.shape[0] != len(horizons):
            raise ValueError("target values shape must be nonempty [H,L,T]")
        if values.dtype.kind not in "fiu":
            raise ValueError("target values must be real numeric arrays")
        if eligible.dtype != np.bool_:
            raise ValueError("eligible must be a boolean mask")
        if eligible.shape != values.shape:
            raise ValueError("eligible shape must match target values")
        mature = self.mature
        if mature is not None:
            mature = np.asarray(mature)
            if mature.dtype != np.bool_ or mature.shape != values.shape:
                raise ValueError("mature must be a boolean mask with target values shape")
            if np.any(eligible & ~mature):
                raise ValueError("eligible cannot extend beyond mature endpoints")
            mature = _snapshot(mature, np.bool_)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "values", _snapshot(values, np.float32))
        object.__setattr__(self, "eligible", _snapshot(eligible, np.bool_))
        object.__setattr__(self, "mature", mature)

    def _slice_lanes(self, selected: slice) -> TargetBatch:
        """复用已验证快照；原生接口要求不连续的多期限切片先打包。"""
        values = self.values[:, selected]
        eligible = self.eligible[:, selected]
        mature = None if self.mature is None else self.mature[:, selected]
        arrays = (values, eligible) if mature is None else (values, eligible, mature)
        if not all(array.flags.c_contiguous for array in arrays):
            return TargetBatch(self.horizons, values, eligible, mature)
        # 底层由不可写 bytes 持有，视图不能重新开启写入，也不会污染父批次。
        batch = object.__new__(TargetBatch)
        object.__setattr__(batch, "horizons", self.horizons)
        object.__setattr__(batch, "values", values)
        object.__setattr__(batch, "eligible", eligible)
        object.__setattr__(batch, "mature", mature)
        return batch


def targets_from_close(close, horizons, valid=None) -> TargetBatch:
    """同 lane 端点对数收益；计算顺序为 FP32→FP64→FP32。"""
    horizons = _horizons(horizons)
    original = np.asarray(close)
    if original.ndim not in (2, 3) or min(original.shape) < 1:
        raise ValueError("close must have nonempty [L,T] or [S,N,T] shape")
    if original.dtype.kind not in "fiu":
        raise ValueError("close must be real numeric")
    if valid is None:
        available = np.ones(original.shape, dtype=bool)
    else:
        available = np.asarray(valid)
        if available.dtype != np.bool_ or available.shape != original.shape:
            raise ValueError("valid must be a boolean mask matching close shape")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = np.array(original, dtype=np.float32, order="C", copy=True)
    time = stored.shape[-1]
    prices = stored.reshape(-1, time).astype(np.float64)
    observed = available.reshape(-1, time) & np.isfinite(prices) & (prices > 0)
    shape = (len(horizons), prices.shape[0], time)
    values = np.full(shape, np.nan, dtype=np.float32)
    mature = np.zeros(shape, dtype=bool)
    eligible = np.zeros(shape, dtype=bool)
    for index, horizon in enumerate(horizons):
        if horizon >= time:
            continue
        mature[index, :, :-horizon] = True
        good = observed[:, :-horizon] & observed[:, horizon:]
        eligible[index, :, :-horizon] = good
        ratio = np.divide(
            prices[:, horizon:],
            prices[:, :-horizon],
            out=np.full_like(prices[:, :-horizon], np.nan),
            where=good,
        )
        values[index, :, :-horizon] = np.log(ratio).astype(np.float32)
    return TargetBatch(horizons, values, eligible, mature)


class MultiTargetSource:
    """委托原 source 读取，按其局部 lane 顺序附加多期限标签。"""

    def __init__(self, source: IndexedLaneSource, targets: TargetBatch) -> None:
        if targets.values.shape[1:] != (source.lane_count, source.time):
            raise ValueError("target shape must match source lane_count and time")
        self.source = source
        self.targets = targets

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.source.field_names

    @property
    def source_identity(self) -> str:
        return self.source.source_identity

    @property
    def lane_count(self) -> int:
        return self.source.lane_count

    @property
    def time(self) -> int:
        return self.source.time

    def batch_ranges(self, first_lane: int, max_lanes: int) -> Iterator[tuple[int, int]]:
        """CUDA 多期预分配需要显式批次布局；查询不允许读取实际数据。"""
        describe = getattr(self.source, "batch_ranges", None)
        if not callable(describe):
            raise TypeError("CUDA multi-target source requires batch_ranges metadata")
        cursor = first_lane
        for first, count in self.source.batch_ranges(first_lane, max_lanes):
            if (
                type(first) is not int
                or type(count) is not int
                or first != cursor
                or not 0 < count <= max_lanes
                or first + count > self.lane_count
            ):
                raise ValueError("source batch_ranges must describe ordered complete lane batches")
            yield first, count
            cursor += count
        if cursor != self.lane_count:
            raise ValueError("source batch_ranges does not cover all lanes")

    def iter_batches(
        self, first_lane: int, max_lanes: int
    ) -> Iterator[tuple[IndexedBatch, TargetBatch]]:
        for batch in self.source.iter_batches(first_lane, max_lanes):
            count = batch.panel.shape[0] * batch.panel.shape[1]
            begin = batch.first_lane
            if count < 1 or begin < 0 or begin + count > self.lane_count:
                raise ValueError("source batch lane range exceeds target shape")
            yield batch, self.targets._slice_lanes(slice(begin, begin + count))
