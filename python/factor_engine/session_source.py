"""从磁盘按完整 lane 读取数据；一个 lane 是一个 session 中一只标的的时间序列。

打开时检查文件布局和尺寸；数据准备完成后，读取期间不得原地修改。
调用方负责关闭 source；view 只选择数据范围，不拥有底层文件。
"""

from __future__ import annotations

from contextlib import ExitStack
from itertools import chain
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Iterator, Protocol, cast

import numpy as np

from .data import Panel, IndexedBatch


class IndexedLaneSource(Protocol):
    """分批评分所需的数据源接口。"""

    @property
    def field_names(self) -> tuple[str, ...]: ...

    @property
    def source_identity(self) -> str: ...

    @property
    def lane_count(self) -> int: ...

    @property
    def time(self) -> int: ...

    def iter_batches(self, first_lane: int, max_lanes: int) -> Iterator[IndexedBatch]: ...

    def batch_ranges(self, first_lane: int, max_lanes: int) -> Iterator[tuple[int, int]]:
        """仅返回与 iter_batches 完全一致的 (first_lane, lane_count)，不读取数据。"""
        ...


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def lane_batch_ranges(first_lane, max_lanes, lane_count):
    """固定批宽的数据源共用的纯元数据范围。"""
    _integer(first_lane, "first_lane", 0)
    _integer(max_lanes, "max_lanes")
    if first_lane > lane_count:
        raise ValueError("first_lane beyond source")
    return (
        (first, min(max_lanes, lane_count - first))
        for first in range(first_lane, lane_count, max_lanes)
    )


def _json_bytes(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def create_session_source(path, batches, *, sessions, securities, time, provenance, byte_limit):
    """按 lane 顺序写入新数据集；关闭输入迭代器，失败时清理本次暂存目录。"""
    for value, name in (
        (sessions, "sessions"),
        (securities, "securities"),
        (time, "time"),
        (byte_limit, "byte_limit"),
    ):
        _integer(value, name)

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    iterator = iter(batches)
    stage = None
    try:
        with ExitStack() as resources:
            # 创建迭代器后，首批校验或文件准备失败也必须关闭它。
            close = getattr(iterator, "close", None)
            if callable(close):
                resources.callback(close)
            first = next(iterator, None)
            if first is None:
                raise ValueError("empty source")
            if not isinstance(first, IndexedBatch):
                raise TypeError("source requires IndexedBatch")
            names = tuple(first.panel.fields)
            if not names or any(not isinstance(n, str) or not n for n in names):
                raise ValueError("invalid fields")
            cells = sessions * securities * time
            required = cells * (4 * len(names) + 5)
            if required > byte_limit:
                raise MemoryError("persisted source exceeds byte limit")
            _json_bytes(provenance)  # 先确认 provenance 能写成 JSON，再创建文件。

            path.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
            files = [f"field-{i}.bin" for i in range(len(names))] + ["valid.bin", "target.bin"]
            dtypes = [np.dtype("<f4")] * len(names) + [np.dtype("u1"), np.dtype("<f4")]
            digests = [hashlib.sha256() for _ in files]
            cursor = 0
            with ExitStack() as outputs:
                handles = []
                for name in files:
                    handle = outputs.enter_context((stage / name).open("xb"))
                    handles.append(handle)
                for batch in chain((first,), iterator):
                    if (
                        not isinstance(batch, IndexedBatch)
                        or isinstance(batch.first_lane, bool)
                        or batch.first_lane != cursor
                    ):
                        raise ValueError("source lane ordinal must be contiguous and ordered")
                    panel = batch.panel
                    target = np.asarray(batch.target)
                    if (
                        tuple(panel.fields) != names
                        or panel.shape[2] != time
                        or target.shape != panel.shape
                    ):
                        raise ValueError("source field/time/target schema changed")
                    count = panel.shape[0] * panel.shape[1]
                    if count <= 0 or cursor + count > sessions * securities:
                        raise ValueError("source exceeds lane count")
                    arrays = list(panel.fields.values()) + [panel.valid, target]
                    for out, digest, array, dtype in zip(
                        handles, digests, arrays, dtypes, strict=True
                    ):
                        data = memoryview(np.ascontiguousarray(array, dtype=dtype)).cast("B")
                        out.write(data)
                        digest.update(data)
                    cursor += count
                if cursor != sessions * securities:
                    raise ValueError("source has missing lanes")
                for out in handles:
                    out.flush()
                    os.fsync(out.fileno())

            file_records = []
            for name, digest in zip(files, digests, strict=True):
                file_records.append(
                    {
                        "name": name,
                        "bytes": (stage / name).stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            body = {
                "version": 1,
                "layout": "session-security-time-C",
                "sessions": sessions,
                "securities": securities,
                "time": time,
                "fields": list(names),
                "provenance": provenance,
                "files": file_records,
            }
            body["source_identity"] = hashlib.sha256(_json_bytes(body)).hexdigest()
            with (stage / "manifest.json").open("xb") as out:
                out.write(_json_bytes(body) + b"\n")
                out.flush()
                os.fsync(out.fileno())
            # 不覆盖已有数据集；若其他任务先写入非空目录，rename 也会失败。
            if path.exists():
                raise FileExistsError(path)
            os.rename(stage, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        # ExitStack 已关闭所有打开的文件，再清理本次创建的目录。
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        raise


class SessionSource:
    def __init__(self, path, *, max_batch_bytes=256 * 1024**2):
        _integer(max_batch_bytes, "max_batch_bytes")
        self.path = Path(path)
        self.max_batch_bytes = max_batch_bytes
        self._fds = []
        self._closed = False
        self.stats = {"read_lanes": 0, "read_bytes": 0, "max_batch_bytes": 0}
        body = json.loads((self.path / "manifest.json").read_text())
        if body["version"] != 1 or body["layout"] != "session-security-time-C":
            raise ValueError("unsupported source layout")
        for name in ("sessions", "securities", "time"):
            _integer(body[name], name)
        self.sessions = body["sessions"]
        self.securities = body["securities"]
        self.time = body["time"]
        self.field_names = tuple(body["fields"])
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("invalid source fields")
        self.source_identity = body["source_identity"]
        self.provenance = MappingProxyType(body["provenance"])
        self.lane_count = self.sessions * self.securities
        self.bytes_per_lane = self.time * (4 * len(self.field_names) + 5)
        self.shape = (self.sessions, self.securities, self.time)
        names = [f"field-{i}.bin" for i in range(len(self.field_names))] + [
            "valid.bin",
            "target.bin",
        ]
        sizes = [self.lane_count * self.time * 4] * len(self.field_names) + [
            self.lane_count * self.time,
            self.lane_count * self.time * 4,
        ]
        if len(body["files"]) != len(names):
            raise ValueError("invalid source file count")
        try:
            for entry, name, size in zip(body["files"], names, sizes, strict=True):
                if entry["name"] != name or entry["bytes"] != size:
                    raise ValueError("source file size/schema mismatch")
                fd = os.open(self.path / name, os.O_RDONLY)
                self._fds.append(fd)
                status = os.fstat(fd)
                if not stat.S_ISREG(status.st_mode):
                    raise ValueError("source must contain regular files")
                if status.st_size != size:
                    raise ValueError("source file size mismatch")
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("source is closed")

    def view(self, sessions):
        return SessionView(self, sessions)

    def global_lanes(self, first_lane, count):
        return tuple(range(first_lane, first_lane + count))

    def iter_batches(self, first_lane, max_lanes):
        for first, count in self.batch_ranges(first_lane, max_lanes):
            yield self.read_batch(first, count)

    def batch_ranges(self, first_lane, max_lanes):
        ranges = lane_batch_ranges(first_lane, max_lanes, self.lane_count)
        if max_lanes * self.bytes_per_lane > self.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        self._ensure_open()
        yield from ranges

    def read_batch(self, first_lane, max_lanes):
        """读取连续完整 lane；只有选择部分 session 时才创建 SessionView。"""
        _integer(first_lane, "first_lane", 0)
        _integer(max_lanes, "max_lanes")
        if first_lane >= self.lane_count:
            raise ValueError("first_lane beyond source")
        if max_lanes * self.bytes_per_lane > self.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        end = min(first_lane + max_lanes, self.lane_count)
        panel, target = self._read(range(first_lane, end))
        return IndexedBatch(first_lane, panel, target)

    def _read(self, global_lanes):
        count = len(global_lanes)
        required = count * self.bytes_per_lane
        if required > self.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        self._ensure_open()
        arrays = []
        for index, fd in enumerate(self._fds):
            dtype = np.dtype("u1") if index == len(self.field_names) else np.dtype("<f4")
            array = np.empty((1, count, self.time), dtype=dtype)
            # 连续 lane 合并为一次读取；不连续的 lane 保留调用方指定的顺序。
            begin = 0
            while begin < count:
                end = begin + 1
                while end < count and global_lanes[end] == global_lanes[end - 1] + 1:
                    end += 1
                buffer = memoryview(array[:, begin:end]).cast("B")
                offset = global_lanes[begin] * self.time * dtype.itemsize
                read = 0
                while read < len(buffer):
                    n = os.preadv(fd, [buffer[read:]], offset + read)
                    if n == 0:
                        raise RuntimeError("source truncated during read")
                    read += n
                begin = end
            arrays.append(array)
        fields = dict(zip(self.field_names, arrays[: len(self.field_names)], strict=True))
        panel = Panel(fields, arrays[-2].view(np.bool_))
        # Panel 构造时已把缺省资格转换为同形布尔数组。
        valid = cast(np.ndarray, panel.valid)
        for a in (*panel.fields.values(), valid, arrays[-1]):
            a.setflags(write=False)
        object.__setattr__(panel, "fields", MappingProxyType(panel.fields))
        self.stats["read_lanes"] += count
        self.stats["read_bytes"] += required
        self.stats["max_batch_bytes"] = max(self.stats["max_batch_bytes"], required)
        return panel, arrays[-1]

    def close(self):
        if not self._closed:
            for fd in self._fds:
                os.close(fd)
            self._fds.clear()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()


class SessionView:
    def __init__(self, source, sessions):
        self.source = source
        source._ensure_open()
        self.session_indices = tuple(sessions)
        if not self.session_indices or len(set(self.session_indices)) != len(self.session_indices):
            raise ValueError("sessions must be nonempty and unique")
        if any(
            isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < source.sessions
            for s in self.session_indices
        ):
            raise ValueError("session outside source")
        self.field_names = source.field_names
        self.time = source.time
        self.securities = source.securities
        self.lane_count = len(self.session_indices) * self.securities
        self.shape = (len(self.session_indices), self.securities, self.time)
        self.source_identity = hashlib.sha256(
            _json_bytes([source.source_identity, self.session_indices])
        ).hexdigest()
        self.bytes_per_lane = source.bytes_per_lane

    @property
    def max_batch_bytes(self):
        return self.source.max_batch_bytes

    @property
    def role(self):
        return getattr(self.source, "role", None)

    def global_lanes(self, first_lane, count):
        return tuple(
            self.session_indices[i // self.securities] * self.securities + i % self.securities
            for i in range(first_lane, first_lane + count)
        )

    def read_batch(self, first_lane, max_lanes):
        """从指定 lane 开始读取一批数据，末批可以不足 max_lanes。"""
        _integer(first_lane, "first_lane", 0)
        _integer(max_lanes, "max_lanes")
        if first_lane >= self.lane_count:
            raise ValueError("first_lane beyond source")
        if max_lanes * self.bytes_per_lane > self.source.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        panel, target = self.source._read(
            self.global_lanes(first_lane, min(max_lanes, self.lane_count - first_lane))
        )
        return IndexedBatch(first_lane, panel, target)

    def iter_batches(self, first_lane, max_lanes):
        for first, count in self.batch_ranges(first_lane, max_lanes):
            yield self.read_batch(first, count)

    def batch_ranges(self, first_lane, max_lanes):
        ranges = lane_batch_ranges(first_lane, max_lanes, self.lane_count)
        if max_lanes * self.bytes_per_lane > self.source.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        self.source._ensure_open()
        yield from ranges


class ArraySource:
    """保存输入的一份只读副本，每代评分都读取同一份数据的 lane 切片。"""

    @staticmethod
    def required_bytes(panel, target):
        if np.shape(target) != panel.shape:
            raise ValueError("target shape must match panel shape")
        return int(np.prod(panel.shape, dtype=object)) * (4 * len(panel.fields) + 5)

    def __init__(
        self,
        panel,
        target,
        *,
        source_identity,
        max_batch_bytes=256 * 1024**2,
        snapshot_budget_bytes=256 * 1024**2,
    ):
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError("explicit source identity required")
        required = self.required_bytes(panel, target)
        _integer(snapshot_budget_bytes, "snapshot_budget_bytes")
        if required > snapshot_budget_bytes:
            raise MemoryError("in-memory source snapshot exceeds host budget")
        _integer(max_batch_bytes, "max_batch_bytes")
        fields = {
            name: np.array(value, dtype=np.float32, order="C", copy=True)
            for name, value in panel.fields.items()
        }
        valid = np.array(panel.valid, dtype=bool, order="C", copy=True)
        target = np.array(target, dtype=np.float32, order="C", copy=True)
        for array in (*fields.values(), valid, target):
            array.setflags(write=False)
        self.panel = Panel(fields, valid)
        object.__setattr__(self.panel, "fields", MappingProxyType(fields))
        self.target = target
        self._valid = valid
        self.source_identity = source_identity
        self.field_names = tuple(fields)
        self.time = panel.shape[2]
        self.shape = panel.shape
        self.sessions, self.securities, _ = panel.shape
        self.lane_count = self.sessions * self.securities
        self.bytes_per_lane = self.time * (4 * len(fields) + 5)
        self.max_batch_bytes = max_batch_bytes
        self.snapshot_bytes = required

    def iter_batches(self, first_lane, max_lanes):
        ranges = self.batch_ranges(first_lane, max_lanes)
        fields = {
            name: value.reshape(self.lane_count, self.time)
            for name, value in self.panel.fields.items()
        }
        valid = self._valid.reshape(self.lane_count, self.time)
        target = self.target.reshape(self.lane_count, self.time)
        for first, count in ranges:
            last = first + count
            panel = Panel(
                {name: value[first:last][None] for name, value in fields.items()},
                valid[first:last][None],
            )
            yield IndexedBatch(first, panel, target[first:last][None])

    def batch_ranges(self, first_lane, max_lanes):
        if type(first_lane) is not int or not 0 <= first_lane <= self.lane_count:
            raise ValueError("first lane outside source")
        _integer(max_lanes, "max_lanes")
        if max_lanes * self.bytes_per_lane > self.max_batch_bytes:
            raise MemoryError("source frame exceeds explicit host budget")
        yield from lane_batch_ranges(first_lane, max_lanes, self.lane_count)
