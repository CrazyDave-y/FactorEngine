"""生成用来测试引擎的行情形状数据和未来收益标签。

没有交易的时点不伪造 OHLC；相同种子和 lane 总能生成相同数据。
这套模拟分布用于工程测试，不能用它判断策略收益。
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .data import Panel, IndexedBatch
from .session_source import lane_batch_ranges

TERMINALS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "range_pct",
    "body_pct",
    "close_location",
    "relative_volume_60",
    "realized_vol_60",
)
SCHEMA = "factorengine-marketlike-12-v2.0"
_UINT64_MAX = 2**64 - 1


def _integer(value, name, minimum=0, maximum=_UINT64_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def counter_uniform(positions, *, seed, stream):
    """用位置、种子和 stream 生成可重复的均匀随机数。

    SplitMix64 的整数运算按 2**64 回绕。取高 53 位加半格转为浮点，
    只在末端避开 1，避免后续对数等运算遇到端点。
    """
    _integer(seed, "seed")
    _integer(stream, "stream")
    pos = np.asarray(positions)
    if pos.dtype != np.uint64:
        raise ValueError("counter positions must be explicit uint64")
    domain = (seed + stream * 0x9E3779B97F4A7C15) & _UINT64_MAX
    z = pos + np.uint64(domain)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.minimum(
        np.ldexp((z >> np.uint64(11)).astype(np.float64) + 0.5, -53),
        np.nextafter(np.float64(1), np.float64(0)),
    )


def _window_mean(values, window):
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.shape[-1] < window:
        return out
    finite = np.isfinite(values)
    prefix = np.concatenate(
        (
            np.zeros((*values.shape[:-1], 1)),
            np.cumsum(np.where(finite, values, 0), axis=-1, dtype=np.float64),
        ),
        axis=-1,
    )
    counts = np.concatenate(
        (
            np.zeros((*values.shape[:-1], 1), dtype=np.int64),
            np.cumsum(finite, axis=-1, dtype=np.int64),
        ),
        axis=-1,
    )
    sums = prefix[..., window:] - prefix[..., :-window]
    valid = counts[..., window:] - counts[..., :-window] == window
    out[..., window - 1 :] = np.where(valid, sums / window, np.nan)
    return out


def derive_marketlike(ohlcv: Mapping[str, np.ndarray], available):
    """从 OHLCV 计算历史特征，再生成未来 30 点的收益标签。

    先把 OHLCV 存成 float32；派生计算用 float64，结果再转回 float32。
    波动率使用已存下来的 ret_1。窗口不跨 session，缺失值也不前向填充。
    """
    if tuple(ohlcv) != TERMINALS[:5]:
        raise ValueError("ordered OHLCV schema required")
    valid = np.asarray(available)
    if valid.dtype != np.bool_ or valid.ndim != 3 or min(valid.shape) < 1:
        raise ValueError("source availability must be a nonempty3D boolean mask")
    valid = valid.copy()
    fields = {
        name: np.array(value, dtype=np.float32, order="C", copy=True)
        for name, value in ohlcv.items()
    }
    if any(a.shape != valid.shape for a in fields.values()):
        raise ValueError("OHLCV shape mismatch")
    volume = fields["volume"]
    observed_volume = volume[valid]
    if not np.isfinite(observed_volume).all() or np.any(observed_volume < 0):
        raise ValueError("available volume must be finite and nonnegative")
    traded = valid & (volume > 0)
    zero = valid & (volume == 0)
    o, h, l, c = (fields[n] for n in TERMINALS[:4])
    if any(np.any(np.isfinite(a[zero])) for a in (o, h, l, c)):
        raise ValueError("zero trade must not fabricate OHLC")
    if any(not np.isfinite(a[traded]).all() or np.any(a[traded] <= 0) for a in (o, h, l, c)):
        raise ValueError("trade OHLC must be finite and positive")
    if np.any(h[traded] < np.maximum(o, c)[traded]) or np.any(l[traded] > np.minimum(o, c)[traded]):
        raise ValueError("illegal OHLC bounds")
    for a in fields.values():
        a[~valid] = np.nan
    o, h, l, c, v = (fields[n].astype(np.float64) for n in TERMINALS[:5])
    for lag, name in ((1, "ret_1"), (5, "ret_5")):
        result = np.full(c.shape, np.nan)
        if c.shape[-1] > lag:
            good = np.isfinite(c[..., lag:]) & np.isfinite(c[..., :-lag])
            ratio = np.divide(
                c[..., lag:], c[..., :-lag], out=np.full_like(c[..., lag:], np.nan), where=good
            )
            result[..., lag:] = np.log(ratio)
        fields[name] = result.astype(np.float32)
    fields["range_pct"] = ((h - l) / c).astype(np.float32)
    fields["body_pct"] = ((c - o) / o).astype(np.float32)
    amplitude = h - l
    location = np.full_like(c, np.nan)
    nonzero = np.isfinite(amplitude) & (amplitude != 0) & np.isfinite(c)
    np.divide(c - l, amplitude, out=location, where=nonzero)
    location[np.isfinite(amplitude) & (amplitude == 0) & np.isfinite(c)] = 0.5
    fields["close_location"] = location.astype(np.float32)
    mean_volume = _window_mean(v, 60)
    relative = np.full_like(c, np.nan)
    np.divide(v, mean_volume, out=relative, where=np.isfinite(mean_volume) & (mean_volume != 0))
    fields["relative_volume_60"] = relative.astype(np.float32)
    fields["realized_vol_60"] = np.sqrt(
        _window_mean(fields["ret_1"].astype(np.float64) ** 2, 60)
    ).astype(np.float32)
    target = np.full(c.shape, np.nan, dtype=np.float32)
    if c.shape[-1] > 30:
        good = np.isfinite(c[..., :-30]) & np.isfinite(c[..., 30:])
        ratio = np.divide(
            c[..., 30:], c[..., :-30], out=np.full_like(c[..., :-30], np.nan), where=good
        )
        target[..., :-30] = np.log(ratio).astype(np.float32)
    # 这个时点有原始观测，也可能因为窗口不足而缺少某个派生字段。
    for a in fields.values():
        a[~valid] = np.nan
        a.setflags(write=False)
    valid.setflags(write=False)
    target.setflags(write=False)
    panel = Panel(fields, valid)
    object.__setattr__(panel, "fields", MappingProxyType(panel.fields))
    return panel, target


class MarketLikeSource:
    """每次只生成请求的完整 lane，结果不随读取顺序或分批大小改变。"""

    field_names = TERMINALS

    def __init__(self, *, seed, sessions, securities, time, max_working_bytes=256 * 1024**2):
        self.seed = _integer(seed, "seed")
        self.sessions = _integer(sessions, "sessions", 1)
        self.securities = _integer(securities, "securities", 1)
        self.time = _integer(time, "time", 1, 2**31 - 1)
        self.lane_count = self.sessions * self.securities
        if self.lane_count * self.time > 2**64:
            raise ValueError("global cell index exceeds uint64")
        self.max_working_bytes = _integer(max_working_bytes, "working budget", 1)
        self.bytes_per_lane = self.time * 53
        self.required_bytes = self.lane_count * self.bytes_per_lane
        self.shape = (self.sessions, self.securities, self.time)
        self.provenance = {
            "schema": SCHEMA,
            "seed": seed,
            "generator": "splitmix64-domain-fir1-v1",
            "zero_trade_rule": "uniform-domain7 < 1/16384",
            "missing_rule": "uniform-domain8 < 1/262144",
            "price": "positive latent log-price; FIR white+0.15previous, shared session shock; no emitted no-trade OHLC",
            "rounding": "observationsFP32; derivedFP64-on-stored-input thenFP32; realized_vol usesstoredret1",
            "windows": "full finite trailing60 includingcurrent; reset eachlane/session",
            "target": "h30 log observed close endpoints; same session; last30 NaN; notterminal",
            "research_status": "SYNTHETIC_ENGINEERING_ONLY_NOT_FINANCIAL_OOS",
        }

    def read_lanes(self, indices):
        lanes = tuple(indices)
        if not lanes or any(type(x) is not int or not 0 <= x < self.lane_count for x in lanes):
            raise ValueError("global lane outside source")
        # 生成时输入、累计值和输出会同时存在，按它们的合计大小限制这一批。
        if len(lanes) * self.time * 512 > self.max_working_bytes:
            raise MemoryError("generator complete-lane working preflight rejected")
        t = np.arange(self.time, dtype=np.uint64)
        lane = np.asarray(lanes, dtype=np.uint64)[:, None]
        positions = lane * np.uint64(self.time) + t[None, :]
        session = lane // np.uint64(self.securities)
        shared = session * np.uint64(self.time) + t[None, :]
        white = (counter_uniform(positions, seed=self.seed, stream=1) - 0.5) * 0.002
        innovation = white.copy()
        innovation[:, 1:] += 0.15 * white[:, :-1]
        innovation += (counter_uniform(shared, seed=self.seed, stream=2) - 0.5) * 0.0003
        base = 10 + (counter_uniform(lane[:, 0], seed=self.seed, stream=3) * 90)
        log_price = np.log(base[:, None]) + np.cumsum(innovation, axis=-1, dtype=np.float64)
        close = np.exp(log_price)
        opening = np.exp(log_price - 0.75 * innovation)
        wick = counter_uniform(positions, seed=self.seed, stream=4) * 0.0007
        high = np.maximum(opening, close) * (1 + wick)
        low = np.minimum(opening, close) / (1 + wick)
        # 用预先设定的 U 形曲线模拟盘中成交活跃度。
        activity = 1 + 3 * (2 * t.astype(np.float64) / max(1, self.time - 1) - 1) ** 2
        volume = np.floor(
            (50 + 950 * counter_uniform(positions, seed=self.seed, stream=5)) * activity
        )
        zero = counter_uniform(positions, seed=self.seed, stream=7) < 1 / 16384
        missing = counter_uniform(positions, seed=self.seed, stream=8) < 1 / 262144
        available = ~missing
        for a in (opening, high, low, close):
            a[zero | missing] = np.nan
        volume[zero] = 0
        volume[missing] = np.nan
        values = (opening, high, low, close, volume)
        raw = {
            name: a[None].astype(np.float32) for name, a in zip(TERMINALS[:5], values, strict=True)
        }
        return derive_marketlike(raw, available[None])

    def iter_batches(self, first_lane, max_lanes):
        for first, count in self.batch_ranges(first_lane, max_lanes):
            panel, target = self.read_lanes(range(first, first + count))
            yield IndexedBatch(first, panel, target)

    def batch_ranges(self, first_lane, max_lanes):
        _integer(first_lane, "first lane", 0, self.lane_count)
        _integer(max_lanes, "batch lanes", 1)
        yield from lane_batch_ranges(first_lane, max_lanes, self.lane_count)


def input_coverage(panel, target):
    """检查各字段的有效点和完整窗口占比。

    至少三个共同有效点，且字段和 target 都非常数时，RankIC 才有定义。
    这里检查的是输入字段，实际生成因子的有效数仍以评分结果为准。
    """
    if tuple(panel.fields) != TERMINALS or np.shape(target) != panel.shape:
        raise ValueError("coverage requires V2 schema and aligned target")
    fields = panel.fields
    valid = panel.valid
    volume = fields["volume"]
    traded = valid & (volume > 0)
    zero = valid & (volume == 0)
    n = valid.size
    o, h, l, c = (fields[x] for x in TERMINALS[:4])
    good = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(l)
        & np.isfinite(c)
        & (l > 0)
        & (h >= np.maximum(o, c))
        & (l <= np.minimum(o, c))
    )
    windows = {}
    rank = {}
    for name, values in fields.items():
        finite = np.isfinite(values) & valid
        windows[name] = {}
        for width in (60, 300, 900):
            if width > values.shape[-1]:
                fraction = 0.0
            else:
                count = np.concatenate(
                    (
                        np.zeros((*finite.shape[:-1], 1), dtype=np.int64),
                        np.cumsum(finite, axis=-1, dtype=np.int64),
                    ),
                    axis=-1,
                )
                fraction = float(
                    np.count_nonzero(count[..., width:] - count[..., :-width] == width) / n
                )
            windows[name][str(width)] = fraction
        pair = finite & np.isfinite(target)
        a = np.max(np.where(pair, values, -np.inf), axis=-1) > np.min(
            np.where(pair, values, np.inf), axis=-1
        )
        b = np.max(np.where(pair, target, -np.inf), axis=-1) > np.min(
            np.where(pair, target, np.inf), axis=-1
        )
        rank[name] = float(np.mean((np.sum(pair, axis=-1) >= 3) & a & b))
    return {
        "cells": n,
        "lanes": panel.shape[0] * panel.shape[1],
        "ohlc_legal_fraction": float(np.mean(good[traded])) if np.any(traded) else None,
        "volume_nonnegative_fraction": float(
            np.mean(np.isfinite(volume[valid]) & (volume[valid] >= 0))
        )
        if np.any(valid)
        else None,
        "zero_trade_fraction": float(np.count_nonzero(zero) / n),
        "missing_source_fraction": float(np.count_nonzero(~valid) / n),
        "terminal_finite_fraction": {
            name: float(np.mean(np.isfinite(value))) for name, value in fields.items()
        },
        "terminal_nonfinite_fraction": {
            name: float(np.mean(~np.isfinite(value))) for name, value in fields.items()
        },
        "target_finite_fraction": float(np.mean(np.isfinite(target))),
        "window_valid_fraction": windows,
        "rank_ic_valid_lane_fraction": rank,
        "rank_validity_method": "finite intersection>=3 and nonconstant factor/target; actual GP roots validated separately",
    }
