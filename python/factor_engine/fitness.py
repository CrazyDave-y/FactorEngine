"""多期限原始 Sum/Count 与搜索层评分，不改变后端统计口径。"""

from __future__ import annotations


from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
import math

import numpy as np

from .gp import Candidate
from .metrics import MetricStats, _immutable_array, _real_array
from .metric_types import Metric


def _metric_means(stats: MetricStats, metric: Metric):
    values = stats.stats[..., metric, :]
    return np.divide(
        values[..., 0],
        values[..., 1],
        out=np.full(values.shape[:-1], np.nan, dtype=np.float64),
        where=values[..., 1] > 0,
    )


def _complexities(values, roots):
    array = _real_array(values)
    if array.shape != (roots,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("finite nonnegative complexities must align with roots")
    return _immutable_array(array)


def _weights(values, horizons):
    array = np.ones(horizons, dtype=np.float64) if values is None else _real_array(values)
    if (
        array.shape != (horizons,)
        or not np.isfinite(array).all()
        or np.any(array < 0)
        or not np.any(array > 0)
    ):
        raise ValueError("finite nonnegative horizon weights with positive total required")
    active = array > 0
    # 先缩放避免合法大权重的总和溢出；活跃期限按原权重判定。
    normalized = array / np.max(array)
    normalized /= np.sum(normalized)
    return normalized, active


def _validate_complexity_penalty(complexity_penalty: float):
    if (
        not isinstance(complexity_penalty, Real)
        or not math.isfinite(complexity_penalty)
        or complexity_penalty < 0
    ):
        raise ValueError("finite nonnegative complexity_penalty required")


def multitarget_fitness(
    stats: MetricStats,
    complexities,
    weights=None,
    complexity_penalty: float = 0.0,
    custom=None,
) -> np.ndarray:
    """按期限加权 RankIC，再扣一次复杂度惩罚；坏根为 -inf。

    默认评分要求每个正权重期限的 RankIC 有定义。
    custom(stats, complexities) 接收原始统计与只读复杂度，必须返回 [R]。
    custom 自行选择指标和复杂度惩罚，不会被再次扣罚；非有限输出仍为
    -inf。外部评分异常直接传播。
    """
    if not isinstance(stats, MetricStats):
        raise TypeError("MetricStats required")
    roots, horizon_count = stats.stats.shape[:2]
    complexities = _complexities(complexities, roots)
    normalized, active = _weights(weights, horizon_count)
    _validate_complexity_penalty(complexity_penalty)
    if custom is None:
        rank_ic = _metric_means(stats, Metric.RANK_IC)[:, active]
        valid = np.all(np.isfinite(rank_ic), axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.sum(rank_ic * normalized[active], axis=1)
            scores -= complexity_penalty * complexities
    else:
        if not callable(custom):
            raise TypeError("custom must be callable")
        scores = _real_array(custom(stats, complexities))
        if scores.shape != (roots,):
            raise ValueError("custom fitness must return exactly [roots]")
        valid = np.ones(roots, dtype=bool)
    return np.where(valid & np.isfinite(scores), scores, -np.inf)


@dataclass(frozen=True, slots=True, eq=False)
class MultiTargetRanking:
    """排名始终索引原 population；stats 也保持原 root 次序。"""

    population: tuple[Candidate, ...]
    stats: MetricStats
    fitness: np.ndarray
    order: tuple[int, ...]
    top_indices: tuple[int, ...]
    horizon_top_indices: dict[int, tuple[int, ...]]


def rank_multitarget_candidates(
    population: Sequence[Candidate],
    stats: MetricStats,
    *,
    weights=None,
    complexity_penalty: float = 0.0,
    custom=None,
    top_k: int | None = 10,
) -> MultiTargetRanking:
    """整体按 fitness 降序稳定排名，每期限独立提供 RankIC Top-K。

    top_k=None 保留全部，超出种群大小时截断。每期限 Top-K 使用其有限
    RankIC 与复杂度惩罚，独立于整体权重和 custom；坏根排在末尾，
    所有同分根保持输入顺序。原始统计不随排序重排。
    """
    if not isinstance(stats, MetricStats):
        raise TypeError("MetricStats required")
    candidates = tuple(population)
    roots = stats.stats.shape[0]
    if len(candidates) != roots:
        raise ValueError("population must align with stats roots")
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer or None")
    limit = roots if top_k is None else min(top_k, roots)
    complexities = [candidate.expr.complexity for candidate in candidates]
    fitness = multitarget_fitness(
        stats,
        complexities,
        weights=weights,
        complexity_penalty=complexity_penalty,
        custom=custom,
    )
    order = tuple(int(i) for i in np.argsort(-fitness, kind="stable"))
    rank_ic = _metric_means(stats, Metric.RANK_IC)
    with np.errstate(over="ignore", invalid="ignore"):
        horizon_scores = rank_ic - complexity_penalty * np.asarray(complexities)[:, None]
    valid = np.isfinite(horizon_scores)
    horizon_scores[~valid] = -np.inf
    horizon_top = {
        horizon: tuple(int(i) for i in np.argsort(-horizon_scores[:, h], kind="stable")[:limit])
        for h, horizon in enumerate(stats.horizons)
    }
    return MultiTargetRanking(
        candidates, stats, _immutable_array(fitness), order, order[:limit], horizon_top
    )


def fitness_and_order(stats, complexities, penalty):
    if len(stats) != len(complexities):
        raise ValueError("complexities must align with original root order")
    raw = np.divide(
        stats[:, 0], stats[:, 1], out=np.full(len(stats), np.nan), where=stats[:, 1] > 0
    )
    fitness_values = []
    for ic, complexity in zip(raw, complexities, strict=True):
        if np.isfinite(ic):
            score = float(ic - penalty * complexity)
        else:
            score = -np.inf
        fitness_values.append(score)
    fitness = np.array(fitness_values, dtype=np.float64)
    # 与 FactorEngine.score_population 相同：按分数降序，同分时保留输入顺序。
    order = sorted(range(len(fitness)), key=lambda root: float(fitness[root]), reverse=True)
    return raw, fitness, order
