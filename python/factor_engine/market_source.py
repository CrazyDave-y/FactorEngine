"""把模拟数据存到磁盘，也支持保留原数据、只补写后面的 session。

顺序读取时，一批数据不会跨文件集合；session view 可以按需拼接前后两部分。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time as clock
from types import MappingProxyType
from typing import cast

import numpy as np

from factor_engine.data import Panel, IndexedBatch
from factor_engine.marketlike import MarketLikeSource
from factor_engine.session_source import SessionSource, SessionView, create_session_source


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(body):
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class CompositeSource:
    def __init__(self, head_path, tail_path, max_batch_bytes=256 * 1024**2):
        self.head = SessionSource(head_path, max_batch_bytes=max_batch_bytes)
        try:
            self.tail = SessionSource(tail_path, max_batch_bytes=max_batch_bytes)
            if (self.head.field_names, self.head.securities, self.head.time) != (
                self.tail.field_names,
                self.tail.securities,
                self.tail.time,
            ):
                raise ValueError("composite source schema mismatch")
            self.sessions = self.head.sessions + self.tail.sessions
            binding = self.tail.provenance.get("composite_extension")
            expected = {
                "head_source_identity": self.head.source_identity,
                "global_first_lane": self.head.lane_count,
                "total_sessions": self.sessions,
                "head_sessions": self.head.sessions,
            }
            if binding != expected:
                raise ValueError("tail extension provenance does not bind original head")
            generator_provenance = dict(self.tail.provenance)
            generator_provenance.pop("composite_extension")
            if generator_provenance != dict(self.head.provenance):
                raise ValueError("head/tail generator provenance mismatch")
        except BaseException:
            self.head.close()
            if hasattr(self, "tail"):
                self.tail.close()
            raise
        self.field_names = self.head.field_names
        self.securities = self.head.securities
        self.time = self.head.time
        self.shape = (self.sessions, self.securities, self.time)
        self.lane_count = self.head.lane_count + self.tail.lane_count
        self.bytes_per_lane = self.head.bytes_per_lane
        self.max_batch_bytes = max_batch_bytes
        provenance = dict(self.head.provenance)
        provenance.update(
            {
                "layout": "persisted-complete-lane-head-tail-v1",
                "head_source_identity": self.head.source_identity,
                "tail_source_identity": self.tail.source_identity,
                "extension": binding,
            }
        )
        self.provenance = MappingProxyType(provenance)
        self.source_identity = _identity(dict(self.provenance))
        self.stats = {"read_lanes": 0, "read_bytes": 0, "max_batch_bytes": 0}

    def _ensure_open(self):
        self.head._ensure_open()
        self.tail._ensure_open()

    def global_lanes(self, first_lane, count):
        _integer(first_lane, "first_lane", 0)
        _integer(count, "count", 0)
        if first_lane + count > self.lane_count:
            raise ValueError("lanes beyond source")
        return tuple(range(first_lane, first_lane + count))

    def read_batch(self, first_lane, max_lanes):
        """最多读 max_lanes 条；到了前后两部分的边界就结束这一批。"""
        _integer(first_lane, "first_lane", 0)
        _integer(max_lanes, "max_lanes")
        if first_lane >= self.lane_count:
            raise ValueError("first_lane beyond source")
        if max_lanes * self.bytes_per_lane > self.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        self._ensure_open()
        if first_lane < self.head.lane_count:
            owner, local = self.head, first_lane
        else:
            owner, local = self.tail, first_lane - self.head.lane_count
        batch = owner.read_batch(local, max_lanes)
        count = batch.panel.shape[0] * batch.panel.shape[1]
        required = count * self.bytes_per_lane
        self.stats["read_lanes"] += count
        self.stats["read_bytes"] += required
        self.stats["max_batch_bytes"] = max(self.stats["max_batch_bytes"], required)
        return IndexedBatch(first_lane, batch.panel, batch.target)

    def iter_batches(self, first_lane, max_lanes):
        for first, count in self.batch_ranges(first_lane, max_lanes):
            yield self.read_batch(first, count)

    def batch_ranges(self, first_lane, max_lanes):
        """保留 head/tail 文件边界，查询过程不读取任何 payload。"""
        _integer(first_lane, "first_lane", 0)
        _integer(max_lanes, "max_lanes")
        if first_lane > self.lane_count:
            raise ValueError("first_lane beyond source")
        if max_lanes * self.bytes_per_lane > self.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        self._ensure_open()
        if first_lane < self.head.lane_count:
            yield from self.head.batch_ranges(first_lane, max_lanes)
        tail_first = max(0, first_lane - self.head.lane_count)
        for first, count in self.tail.batch_ranges(tail_first, max_lanes):
            yield self.head.lane_count + first, count

    def view(self, sessions):
        return SessionView(self, sessions)

    def _read(self, global_lanes):
        lanes = tuple(global_lanes)
        if not lanes or any(type(i) is not int or not 0 <= i < self.lane_count for i in lanes):
            raise ValueError("global lane outside source")
        if len(lanes) * self.bytes_per_lane > self.max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        self._ensure_open()
        # 只分配请求的结果和一批读取缓冲区。
        shape = (1, len(lanes), self.time)
        fields = {name: np.empty(shape, dtype=np.float32) for name in self.field_names}
        valid = np.empty(shape, dtype=bool)
        target = np.empty(shape, dtype=np.float32)
        for owner, offset in ((self.head, 0), (self.tail, self.head.lane_count)):
            positions = [
                i for i, lane in enumerate(lanes) if offset <= lane < offset + owner.lane_count
            ]
            if not positions:
                continue
            panel, labels = owner._read(tuple(lanes[i] - offset for i in positions))
            for name in self.field_names:
                fields[name][:, positions] = panel.fields[name]
            valid[:, positions] = panel.valid
            target[:, positions] = labels
            del panel, labels
        for array in (*fields.values(), valid, target):
            array.setflags(write=False)
        self.stats["read_lanes"] += len(lanes)
        self.stats["read_bytes"] += len(lanes) * self.bytes_per_lane
        self.stats["max_batch_bytes"] = max(
            self.stats["max_batch_bytes"], len(lanes) * self.bytes_per_lane
        )
        return Panel(fields, valid), target

    def close(self):
        self.head.close()
        self.tail.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()


def prepare_tail(
    head_path,
    tail_path,
    *,
    sessions=10,
    securities=400,
    time=23400,
    seed=42,
    batch_lanes=4,
    max_working_bytes=256 * 1024**2,
    max_batch_bytes=256 * 1024**2,
):
    """核对原数据，只生成新增的 session。

    打开时检查文件布局和尺寸，另抽几条 lane 与生成器对照。
    这些都是准备数据的工作，不计入后续评分耗时。
    """
    started = clock.perf_counter()
    for value, name, minimum in (
        (sessions, "sessions", 1),
        (batch_lanes, "batch_lanes", 1),
    ):
        _integer(value, name, minimum)
    tail_path = Path(tail_path)
    if tail_path.exists():
        raise FileExistsError(tail_path)
    generator = MarketLikeSource(
        seed=seed,
        sessions=sessions,
        securities=securities,
        time=time,
        max_working_bytes=max_working_bytes,
    )
    with SessionSource(head_path, max_batch_bytes=max_batch_bytes) as head:
        if (head.securities, head.time, head.field_names) != (
            securities,
            time,
            generator.field_names,
        ):
            raise ValueError("head source schema mismatch")
        if dict(head.provenance) != generator.provenance:
            raise ValueError("head generator provenance mismatch")
        if head.sessions >= sessions:
            raise ValueError("total sessions must extend the head")
        if batch_lanes * head.bytes_per_lane > max_batch_bytes:
            raise MemoryError("source frame exceeds host batch budget")
        required = (sessions - head.sessions) * securities * head.bytes_per_lane
        golden_lanes = sorted({0, securities - 1, head.lane_count - 1})
        for lane in golden_lanes:
            actual = head.read_batch(lane, 1)
            panel, target = generator.read_lanes((lane,))
            actual_arrays = (
                *actual.panel.fields.values(),
                cast(np.ndarray, actual.panel.valid),
                actual.target,
            )
            expected_arrays = (*panel.fields.values(), cast(np.ndarray, panel.valid), target)
            if any(
                a.tobytes() != b.tobytes()
                for a, b in zip(actual_arrays, expected_arrays, strict=True)
            ):
                raise ValueError("head prefix generator parity mismatch")
        offset = head.lane_count
        binding = {
            "head_source_identity": head.source_identity,
            "global_first_lane": offset,
            "total_sessions": sessions,
            "head_sessions": head.sessions,
        }
        provenance = dict(generator.provenance)
        provenance["composite_extension"] = binding

        def tail_batches():
            for batch in generator.iter_batches(offset, batch_lanes):
                yield IndexedBatch(batch.first_lane - offset, batch.panel, batch.target)

        create_session_source(
            tail_path,
            tail_batches(),
            sessions=sessions - head.sessions,
            securities=securities,
            time=time,
            provenance=provenance,
            byte_limit=required,
        )
        tail_manifest = json.loads((tail_path / "manifest.json").read_text())
        atomic_json(
            tail_path / "composite.json",
            {
                "schema": "factorengine-composite-source-v1",
                "head_path": os.path.relpath(Path(head_path).resolve(), tail_path.resolve()),
                "head_source_identity": head.source_identity,
                "tail_source_identity": tail_manifest["source_identity"],
            },
        )
        return {
            "head_source_identity": head.source_identity,
            "tail_source_identity": tail_manifest["source_identity"],
            "tail_global_first_lane": offset,
            "tail_bytes": required,
            "sessions": sessions,
            "shape": [sessions, securities, time],
            "prefix_checked_lanes": len(golden_lanes),
            "prefix_verification": "file sizes checked on open; selected generator lanes compared",
            "generator_parity_lanes": golden_lanes,
            "preparation_seconds": clock.perf_counter() - started,
        }


def atomic_json(path, body):
    """写入一份新结果，目标文件已经存在时直接报错。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(body, stream, allow_nan=False, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_marketlike(path, generator, *, batch_lanes=4, max_batch_bytes=128 * 1024**2):
    """分批生成并落盘，后续搜索只读这些文件。"""
    if not isinstance(generator, MarketLikeSource):
        raise TypeError("MarketLikeSource required")
    _integer(batch_lanes, "batch_lanes")
    _integer(max_batch_bytes, "max_batch_bytes")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if batch_lanes * generator.bytes_per_lane > max_batch_bytes:
        raise MemoryError("source frame exceeds host batch budget")
    started = clock.perf_counter()
    create_session_source(
        path,
        generator.iter_batches(0, batch_lanes),
        sessions=generator.sessions,
        securities=generator.securities,
        time=generator.time,
        provenance=generator.provenance,
        byte_limit=generator.required_bytes,
    )
    body = json.loads((path / "manifest.json").read_text())
    return dict(
        source_identity=body["source_identity"],
        shape=list(generator.shape),
        payload_bytes=generator.required_bytes,
        preparation_seconds=clock.perf_counter() - started,
    )


def open_market_source(path, *, max_batch_bytes=128 * 1024**2):
    path = Path(path)
    descriptor = path / "composite.json"
    if not descriptor.exists():
        source = SessionSource(path, max_batch_bytes=max_batch_bytes)
        if "composite_extension" in source.provenance:
            source.close()
            raise ValueError("tail source requires its composite descriptor")
        return source
    body = json.loads(descriptor.read_text())
    if (
        not isinstance(body, dict)
        or body.get("schema") != "factorengine-composite-source-v1"
        or any(
            not isinstance(body.get(name), str) or not body[name]
            for name in ("head_path", "head_source_identity", "tail_source_identity")
        )
    ):
        raise ValueError("composite descriptor schema differs or required fields are missing")
    source = CompositeSource(path / body["head_path"], path, max_batch_bytes=max_batch_bytes)
    if (
        source.head.source_identity != body["head_source_identity"]
        or source.tail.source_identity != body["tail_source_identity"]
    ):
        source.close()
        raise ValueError("composite descriptor identity differs")
    return source
