from __future__ import annotations

import numpy as np
from collections.abc import Sequence

EPS = np.float32(1e-12)


def _sanitize(x):
    a = np.asarray(x)
    if a.ndim == 0:
        v = np.float32(a)
        return v if np.isfinite(v) else np.float32(np.nan)
    out = np.asarray(a, dtype=np.float32)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32, copy=False)


def _array(x):
    return np.asarray(x, dtype=np.float32)


def apply_elementwise(op: str, args: Sequence[object]):
    with np.errstate(all="ignore"):
        if op == "neg":
            out = -_array(args[0])
        elif op == "abs":
            out = np.abs(_array(args[0]))
        elif op == "sign":
            out = np.sign(_array(args[0]))
        elif op == "signed_sqrt":
            x = _array(args[0])
            out = np.sign(x) * np.sqrt(np.abs(x))
        elif op == "signed_log1p":
            x = _array(args[0])
            out = np.sign(x) * np.log1p(np.abs(x))
        elif op == "add":
            out = _array(args[0]) + _array(args[1])
        elif op == "sub":
            out = _array(args[0]) - _array(args[1])
        elif op == "mul":
            out = _array(args[0]) * _array(args[1])
        elif op == "safe_div":
            a, b = _array(args[0]), _array(args[1])
            shape = np.broadcast_shapes(a.shape, b.shape)
            aa = np.broadcast_to(a, shape)
            bb = np.broadcast_to(b, shape)
            out = np.full(shape, np.nan, dtype=np.float32)
            good = np.isfinite(aa) & np.isfinite(bb) & (np.abs(bb) > EPS)
            np.divide(aa, bb, out=out, where=good)
        elif op == "min":
            out = np.minimum(_array(args[0]), _array(args[1]))
        elif op == "max":
            out = np.maximum(_array(args[0]), _array(args[1]))
        else:
            raise ValueError(f"unsupported elementwise op {op!r}")
    return _sanitize(out)


def delay(x: np.ndarray, lag: int) -> np.ndarray:
    x = _array(x)
    out = np.full_like(x, np.nan)
    if lag < x.shape[-1]:
        src = x[..., :-lag]
        out[..., lag:] = np.where(np.isfinite(src), src, np.nan)
    return out


def diff(x: np.ndarray, lag: int) -> np.ndarray:
    x = _array(x)
    out = np.full_like(x, np.nan)
    if lag < x.shape[-1]:
        a, b = x[..., lag:], x[..., :-lag]
        good = np.isfinite(a) & np.isfinite(b)
        with np.errstate(all="ignore"):
            d = a - b
        out[..., lag:] = np.where(good & np.isfinite(d), d, np.nan)
    return out


def ema(x: np.ndarray, span: int) -> np.ndarray:
    x = _array(x)
    out = np.full_like(x, np.nan)
    alpha = np.float32(2.0 / (span + 1.0))
    state = np.zeros(x.shape[:-1], dtype=np.float32)
    initialized = np.zeros(x.shape[:-1], dtype=bool)
    for t in range(x.shape[-1]):
        cur = x[..., t]
        good = np.isfinite(cur)
        first = good & ~initialized
        state = np.where(first, cur, state)
        updated = alpha * cur + (np.float32(1.0) - alpha) * state
        state = np.where(good & initialized & np.isfinite(updated), updated, state)
        initialized |= good
        out[..., t] = np.where(good & initialized, state, np.nan)
    return out


def rolling_unary(x: np.ndarray, window: int, op: str) -> np.ndarray:
    """逐窗口计算的参考实现，复杂度 O(T*W)，方便核对结果。"""
    x = _array(x)
    out = np.full_like(x, np.nan)
    if window > x.shape[-1]:
        return out
    for t in range(window - 1, x.shape[-1]):
        chunk = x[..., t - window + 1 : t + 1]
        good = np.all(np.isfinite(chunk), axis=-1)
        with np.errstate(all="ignore"):
            if op == "roll_sum":
                value = np.sum(chunk, axis=-1, dtype=np.float64)
            elif op == "roll_mean":
                value = np.mean(chunk, axis=-1, dtype=np.float64)
            elif op == "roll_std":
                value = np.std(chunk.astype(np.float64), axis=-1, ddof=0)
            elif op == "roll_min":
                value = np.min(chunk, axis=-1)
            elif op == "roll_max":
                value = np.max(chunk, axis=-1)
            else:
                raise ValueError(f"unsupported rolling op {op!r}")
        out[..., t] = np.where(good & np.isfinite(value), value, np.nan).astype(np.float32)
    return out


def rolling_pair(a: np.ndarray, b: np.ndarray, window: int, op: str) -> np.ndarray:
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        raise ValueError("pair rolling inputs must have identical shapes")
    out = np.full_like(a, np.nan)
    if window > a.shape[-1]:
        return out
    for t in range(window - 1, a.shape[-1]):
        aa = a[..., t - window + 1 : t + 1].astype(np.float64)
        bb = b[..., t - window + 1 : t + 1].astype(np.float64)
        good = np.all(np.isfinite(aa) & np.isfinite(bb), axis=-1)
        ma = np.mean(aa, axis=-1)
        mb = np.mean(bb, axis=-1)
        cov = np.mean((aa - ma[..., None]) * (bb - mb[..., None]), axis=-1)
        if op == "cov":
            value = cov
        elif op == "corr":
            va = np.mean((aa - ma[..., None]) ** 2, axis=-1)
            vb = np.mean((bb - mb[..., None]) ** 2, axis=-1)
            denom = np.sqrt(va * vb)
            value = np.full_like(cov, np.nan)
            np.divide(cov, denom, out=value, where=denom > float(EPS))
        else:
            raise ValueError(f"unsupported pair rolling op {op!r}")
        out[..., t] = np.where(good & np.isfinite(value), value, np.nan).astype(np.float32)
    return out


def evaluate_op(op: str, args: Sequence[object], param: int | None = None):
    if op in {
        "neg",
        "abs",
        "sign",
        "signed_sqrt",
        "signed_log1p",
        "add",
        "sub",
        "mul",
        "safe_div",
        "min",
        "max",
    }:
        return apply_elementwise(op, args)
    if type(param) is not int or param <= 0:
        raise TypeError(f"{op} requires a positive integer window")
    if op == "delay":
        return delay(np.asarray(args[0]), param)
    if op == "diff":
        return diff(np.asarray(args[0]), param)
    if op == "ema":
        return ema(np.asarray(args[0]), param)
    if op in {"roll_sum", "roll_mean", "roll_std", "roll_min", "roll_max"}:
        return rolling_unary(np.asarray(args[0]), param, op)
    if op in {"cov", "corr"}:
        return rolling_pair(np.asarray(args[0]), np.asarray(args[1]), param, op)
    raise ValueError(f"unsupported op {op!r}")
