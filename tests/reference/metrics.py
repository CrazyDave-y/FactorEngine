"""独立 NumPy 数学参考，不参与生产评分。"""

from __future__ import annotations

import numpy as np
from factor_engine.metric_types import MIN_CORRELATION_PAIRS, METRIC_COUNT, Metric
from factor_engine.target_bank import TargetBatch


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [values.size]))
    for begin, end in zip(starts, ends, strict=True):
        ranks[order[begin:end]] = (int(begin) + int(end) - 1) * 0.5
    return ranks


def _centered_for_pearson(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in "iu":
        # 先用 Python 整数精确平移，避免高偏置整数转 FP64 丢掉单位差或原 dtype 相减溢出。
        origin = int(values[0])
        result = np.fromiter(
            (int(value) - origin for value in values), dtype=np.float64, count=values.size
        )
    else:
        result = np.asarray(values, dtype=np.float64).copy()
    magnitude = float(np.max(np.abs(result)))
    limits = np.finfo(np.float64)
    # |x-mean|≤2*maxabs，平方和≤4*n*maxabs²；再留四倍余量，也保证求均值不溢出。
    upper = np.sqrt(limits.max / values.size) / 4
    # 近常数的最小非零差约为 maxabs*eps/2；此下界避免其平方落入次正规区。
    lower = 2 * np.sqrt(limits.tiny) / limits.eps
    if magnitude > upper or 0 < magnitude < lower:
        # 仅危险范围按 maxabs 的指数缩放，二的幂不会给近常数差值增加除法舍入。
        _, exponent = np.frexp(magnitude)
        result = np.ldexp(result, -int(exponent))
    result -= result.mean()
    return result


def pearson_finite(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < MIN_CORRELATION_PAIRS or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    a = _centered_for_pearson(x)
    b = _centered_for_pearson(y)
    aa, bb = float(a @ a), float(b @ b)
    if aa == 0 or bb == 0:
        return float("nan")
    # 分开开方，避免两个平方和的乘积溢出。
    return float(np.clip((a @ b) / np.sqrt(aa) / np.sqrt(bb), -1, 1))


def pearson_ic(factor: np.ndarray, target: np.ndarray) -> float:
    factor, target = np.asarray(factor), np.asarray(target)
    mask = np.isfinite(factor) & np.isfinite(target)
    return pearson_finite(factor[mask], target[mask])


def rank_ic(factor: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(factor) & np.isfinite(target)
    if int(mask.sum()) < MIN_CORRELATION_PAIRS:
        return float("nan")
    return pearson_ic(average_ranks(factor[mask]), average_ranks(target[mask]))


def mean_lane_rank_ic(factor: np.ndarray, target: np.ndarray) -> float:
    """每个 session、每只标的各算时间序列 RankIC，再对有效结果取平均。"""
    factor, target = np.asarray(factor), np.asarray(target)
    if factor.shape != target.shape or factor.ndim != 3:
        raise ValueError("factor and target must share [sessions, securities, time] shape")
    scores = []
    for session in range(factor.shape[0]):
        for security in range(factor.shape[1]):
            value = rank_ic(factor[session, security], target[session, security])
            if np.isfinite(value):
                scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")


def reference_conditional_rank(values, order, mask) -> np.ndarray:
    """返回原时间位置的双倍零基平均秩；未选中或非有限点为 -1。"""
    values = np.asarray(values)
    order = np.asarray(order)
    mask = np.asarray(mask)
    if values.ndim != 1 or values.dtype.kind not in "fiu":
        raise ValueError("values must be a one-dimensional real array")
    if mask.dtype != np.bool_ or mask.shape != values.shape:
        raise ValueError("mask must be boolean and match values shape")
    if (
        order.shape != values.shape
        or order.dtype.kind not in "iu"
        or np.any(order < 0)
        or np.any(order >= values.size)
    ):
        raise ValueError("order must be a complete sorted index permutation")
    indices = order.astype(np.int64, copy=False)
    if not np.array_equal(
        np.bincount(indices, minlength=values.size), np.ones(values.size, dtype=np.int64)
    ):
        raise ValueError("order must be a complete sorted index permutation")
    ordered = values[indices]
    isnan = np.isnan(ordered)
    non_nan = ordered[~isnan]
    if np.any(non_nan[1:] < non_nan[:-1]) or (
        np.any(isnan) and not np.all(isnan[np.flatnonzero(isnan)[0] :])
    ):
        raise ValueError("order must sort values ascending with NaNs last")
    valid = mask & np.isfinite(values)
    result = np.full(values.shape, -1, dtype=np.int64)
    # 独立重排有效子集，不复用 GPU 的前缀和与 ties 边界公式。
    result[valid] = (2 * average_ranks(values[valid])).astype(np.int64)
    return result


def reference_multitarget(factors, targets: TargetBatch) -> np.ndarray:
    """按共同有效集计算 FP64 [R,L,H,2]，不做跨 lane 汇总。"""
    factors = np.asarray(factors)
    if (
        factors.ndim != 3
        or min(factors.shape) < 1
        or factors.shape[1:] != targets.values.shape[1:]
        or factors.dtype.kind not in "fiu"
    ):
        raise ValueError("factors must have nonempty aligned real [R,L,T] shape")
    roots, lanes, _ = factors.shape
    result = np.empty((roots, lanes, len(targets.horizons), METRIC_COUNT), dtype=np.float64)
    for root in range(roots):
        for lane in range(lanes):
            factor = factors[root, lane]
            for horizon in range(len(targets.horizons)):
                target = targets.values[horizon, lane]
                valid = targets.eligible[horizon, lane] & np.isfinite(factor) & np.isfinite(target)
                x, y = factor[valid], target[valid]
                row = result[root, lane, horizon]
                row[Metric.PEARSON] = pearson_finite(x, y)
                row[Metric.RANK_IC] = pearson_finite(average_ranks(x), average_ranks(y))
    return result
