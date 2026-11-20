from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable
import time
import numpy as np

from .canonicalize import canonicalize
from .data import Panel
from .session_source import ArraySource, IndexedLaneSource
from .expression import Expr
from .gp import Candidate, GPConfig, GeneticProgram
from .ir import compile_expressions
from .metrics import ScoreStats
from .fitness import fitness_and_order

EVOLUTION_SEED_OFFSET = 1009
OFFSPRING_ATTEMPTS_PER_SLOT = 300
IN_MEMORY_SOURCE_BUDGET_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class SearchConfig:
    population_size: int = 128
    generations: int = 8
    elite_size: int = 16
    max_depth: int = 5
    max_nodes: int = 96
    seed: int = 1
    complexity_penalty: float = 0.001
    canonicalize_population: bool = True
    cse: bool = True
    windows: tuple[int, ...] = (5, 10, 20, 60, 120)
    lags: tuple[int, ...] = (1, 2, 5, 10, 20)

    def __post_init__(self):
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("generations must be a nonnegative integer")
        for name, values in (("windows", self.windows), ("lags", self.lags)):
            if (
                not isinstance(values, tuple)
                or not values
                or any(type(x) is not int or x < 1 for x in values)
            ):
                raise ValueError(f"{name} must be a nonempty tuple of positive integers")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    expression: str
    score: float
    complexity: int
    key: tuple


@dataclass(frozen=True, slots=True)
class SearchResult:
    best: ScoredCandidate
    best_expr: Expr
    history: tuple[ScoredCandidate, ...]


class FactorEngine:
    def __init__(self, *, plan_backend):
        self.plan_backend = plan_backend
        self.last_search_stats = None

    def score_population_multitarget(
        self,
        source,
        population,
        config,
        *,
        weights=None,
        custom=None,
        top_k=10,
    ):
        """同一种群只编译一个 DAG，指标汇总后再做 Fitness 和各期限 Top-K。"""
        from .fitness import rank_multitarget_candidates

        stats, _ = self._score_multitarget_raw(source, population, config)
        return rank_multitarget_candidates(
            population,
            stats,
            weights=weights,
            complexity_penalty=config.complexity_penalty,
            custom=custom,
            top_k=top_k,
        )

    def _score_multitarget_raw(self, source, population, config):
        """仅计算本批原始统计；fitness 始终按调用方的完整种群计算。"""
        started = time.perf_counter()
        plan = compile_expressions(
            [candidate.expr for candidate in population],
            canonicalize_exprs=config.canonicalize_population,
            cse=config.cse,
        )
        compiled = time.perf_counter()
        self.plan_backend.prepare_multitarget(
            plan,
            source,
        )
        prepared = time.perf_counter()
        stats = self.plan_backend.score_multitarget(plan, source)
        finished = time.perf_counter()
        return stats, {
            "compile_seconds": compiled - started,
            "prepare_seconds": prepared - compiled,
            "score_seconds": finished - prepared,
        }

    def score_population(
        self,
        panel: Panel | IndexedLaneSource,
        target: np.ndarray | None,
        population: list[Candidate],
        config: SearchConfig,
        *,
        search_stats=None,
        capture_generations=False,
    ):
        expressions = [candidate.expr for candidate in population]
        result = self._raw_scores(panel, target, expressions, config)
        batch_stats = result.run_stats
        rank_ics, fitness, order = fitness_and_order(
            result.stats,
            [candidate.expr.complexity for candidate in population],
            config.complexity_penalty,
        )
        scored = [(float(fitness[index]), population[index]) for index in order]
        if search_stats is not None:
            search_stats["logical_scoring_calls"] += 1
            search_stats["scored_roots"] += len(population)
            search_stats["compiled_unique_nodes"] += batch_stats["plan"]["unique_nodes"]
            if capture_generations:
                self._record_generation(search_stats, population, rank_ics, scored, batch_stats)
        return scored

    def _record_generation(self, stats, population, rank_ics, scored, batch_stats):
        # 评分已结束；这里只把原始 IC 和排序后的候选一起记下来。
        if "finite_lane_counts" not in batch_stats or "rank_ic_sums" not in batch_stats:
            raise RuntimeError("generation capture requires backend run statistics")
        details = {}
        for candidate, ic, count, total in zip(
            population,
            rank_ics,
            batch_stats["finite_lane_counts"],
            batch_stats["rank_ic_sums"],
            strict=True,
        ):
            candidate_id = id(candidate)
            raw_ic = None
            if np.isfinite(ic):
                raw_ic = float(ic)
            details[candidate_id] = {
                "raw_ic": raw_ic,
                "finite_lanes": count,
                "rank_ic_sum": total,
            }

        ordered = []
        for score, candidate in scored:
            record = {"expression": str(candidate.expr), "key": candidate.key}
            if np.isfinite(score):
                record["score"] = float(score)
            else:
                record["score"] = None
            record.update(details[id(candidate)])
            record["operation"] = candidate.operation
            record["parents"] = candidate.parents
            ordered.append(record)
        stats["generations"].append(
            {
                "generation": stats["logical_scoring_calls"] - 1,
                "lanes": batch_stats["lanes"],
                "ordered": ordered,
            }
        )
        stats["execution"].append(dict(batch_stats))

    def _raw_scores(self, panel, target, expressions, config):
        started = time.perf_counter()
        plan = compile_expressions(
            expressions,
            canonicalize_exprs=config.canonicalize_population,
            cse=config.cse,
        )
        compile_seconds = time.perf_counter() - started
        source = (
            panel
            if target is None
            else ArraySource(panel, target, source_identity="in-memory-score")
        )
        result = self.plan_backend.score(plan, source)
        if not isinstance(result, ScoreStats) or result.stats.shape != (len(expressions), 2):
            raise TypeError("score backend must return one ScoreStats row per expression")
        stats = result.run_stats
        stats["plan"] = {
            "ast_nodes": plan.ast_nodes,
            "unique_nodes": plan.unique_nodes,
            "slot_count": plan.slot_count,
            "roots": len(plan.root_ids),
        }
        stats["compile_seconds"] = compile_seconds
        return result

    def _dedup_key(self, expr: Expr, config: SearchConfig) -> tuple:
        return canonicalize(expr).key if config.canonicalize_population else expr.key

    def _next_population(
        self,
        gp: GeneticProgram,
        ranked: list[tuple[float, Candidate]],
        config: SearchConfig,
        rng: np.random.Generator,
        *,
        expression_keys=None,
    ) -> list[Candidate]:
        elites = [candidate for _, candidate in ranked[: config.elite_size]]
        result = list(elites)
        next_keys = {}

        def key_for(expr):
            previous = None if expression_keys is None else expression_keys.get(id(expr))
            return previous[1] if previous is not None else self._dedup_key(expr, config)

        def accepted(expr, key):
            if expression_keys is not None:
                next_keys[id(expr)] = (expr, key)

        seen = set()
        for candidate in result:
            key = key_for(candidate.expr)
            seen.add(key)
            accepted(candidate.expr, key)
        attempts = 0
        crossover_cutoff = gp.config.crossover_probability
        subtree_cutoff = crossover_cutoff + gp.config.subtree_mutation_probability
        point_cutoff = subtree_cutoff + gp.config.point_mutation_probability
        hoist_cutoff = point_cutoff + gp.config.hoist_mutation_probability

        while (
            len(result) < config.population_size
            and attempts < config.population_size * OFFSPRING_ATTEMPTS_PER_SLOT
        ):
            attempts += 1
            u = float(rng.random())
            if u < crossover_cutoff:
                parent_a = gp.tournament(ranked, rng)
                parent_b = gp.tournament(ranked, rng)
                expr = gp.crossover_one(parent_a.expr, parent_b.expr, rng)
                parents, operation = (parent_a.key, parent_b.key), "crossover"
            elif u < subtree_cutoff:
                parent_a = gp.tournament(ranked, rng)
                expr = gp.subtree_mutate(parent_a.expr, rng)
                parents, operation = (parent_a.key,), "subtree_mutation"
            elif u < point_cutoff:
                parent_a = gp.tournament(ranked, rng)
                expr = gp.point_mutate(parent_a.expr, rng)
                parents, operation = (parent_a.key,), "point_mutation"
            elif u < hoist_cutoff:
                parent_a = gp.tournament(ranked, rng)
                expr = gp.hoist_mutate(parent_a.expr, rng)
                parents, operation = (parent_a.key,), "hoist_mutation"
            else:
                parent_a = gp.tournament(ranked, rng)
                expr = parent_a.expr
                parents, operation = (parent_a.key,), "reproduction"

            key = key_for(expr)
            if key in seen:
                continue
            seen.add(key)
            result.append(Candidate(expr, expr.key, parents, operation))
            accepted(expr, key)

        # 交叉和变异总是撞到已有个体时，用语法枚举补齐这一代。
        gp.fill_unique(result, seen, config.population_size, key_for, accepted=accepted)
        if expression_keys is not None:
            # 只保留本代已接收对象，不额外持有被拒绝或淘汰个体的引用。
            expression_keys.clear()
            expression_keys.update(next_keys)
        return result

    def search(
        self, panel: Panel, target: np.ndarray, terminals: list[str], config: SearchConfig
    ) -> SearchResult:
        self.last_search_stats = None
        if target.shape != panel.shape:
            raise ValueError("target shape must match panel shape")
        if not 0 < config.elite_size <= config.population_size:
            raise ValueError("elite_size must be in (0, population_size]")
        source = ArraySource(
            panel,
            target,
            source_identity="in-memory-search",
            snapshot_budget_bytes=IN_MEMORY_SOURCE_BUDGET_BYTES,
        )
        return self._search_single(source, None, terminals, config)

    def _search_single(
        self,
        panel,
        target,
        terminals,
        config,
        generation_context=None,
        *,
        capture_generations=False,
    ):
        # 本次搜索独有的统计随回调显式传递；失败时不发布半份报告。
        stats = {
            "data_identity": panel.source_identity,
            "logical_scoring_calls": 0,
            "scored_roots": 0,
            "compiled_unique_nodes": 0,
            "generations": [],
            "execution": [],
        }

        def score_generation(population):
            return self.score_population(
                panel,
                target,
                population,
                config,
                search_stats=stats,
                capture_generations=capture_generations,
            )

        result = self._search_impl(
            terminals,
            config,
            generation_context,
            population_scorer=score_generation,
            timings=stats if capture_generations else None,
        )
        self.last_search_stats = stats
        return result

    def _search_impl(
        self,
        terminals,
        config,
        generation_context=None,
        *,
        population_scorer: Callable[[list[Candidate]], list[tuple[float, Candidate]]],
        timings=None,
        expression_keys=None,
    ):
        gp_config = GPConfig(
            population_size=config.population_size,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            seed=config.seed,
            windows=config.windows,
            lags=config.lags,
        )
        gp = GeneticProgram(terminals, gp_config)
        initial_start = time.perf_counter()

        def accepted(expr, key, storage=expression_keys):
            if storage is not None:
                storage[id(expr)] = (expr, key)

        population = gp.initial_population(
            lambda expr: self._dedup_key(expr, config),
            accepted=accepted if expression_keys is not None else None,
        )
        if timings is not None:
            timings["generate_seconds"] = time.perf_counter() - initial_start
            timings["evolve_seconds"] = []
        rng = np.random.default_rng(config.seed + EVOLUTION_SEED_OFFSET)
        history: list[ScoredCandidate] = []
        global_best: tuple[float, Candidate] | None = None

        for generation in range(config.generations):
            with nullcontext() if generation_context is None else generation_context(generation):
                ranked = population_scorer(population)
                if global_best is None or ranked[0][0] > global_best[0]:
                    global_best = ranked[0]
                score, candidate = ranked[0]
                history.append(
                    ScoredCandidate(
                        str(candidate.expr), score, candidate.expr.complexity, candidate.key
                    )
                )
                evolve_start = time.perf_counter()
                population = self._next_population(
                    gp, ranked, config, rng, expression_keys=expression_keys
                )
                if timings is not None:
                    timings["evolve_seconds"].append(time.perf_counter() - evolve_start)

        with (
            nullcontext() if generation_context is None else generation_context(config.generations)
        ):
            final_ranked = population_scorer(population)
        if global_best is None or final_ranked[0][0] > global_best[0]:
            global_best = final_ranked[0]
        best_score, best_candidate = global_best
        if not np.isfinite(best_score):
            raise ValueError(
                "search produced no finite RankIC; check target alignment, valid points and windows"
            )
        best = ScoredCandidate(
            str(best_candidate.expr), best_score, best_candidate.expr.complexity, best_candidate.key
        )
        return SearchResult(best, best_candidate.expr, tuple(history))

    def search_multitarget(
        self,
        source,
        terminals: list[str],
        config: SearchConfig,
        *,
        weights=None,
        custom=None,
        top_k=10,
        generation_context=None,
        cache_scores=True,
        reuse_inputs=True,
    ) -> SearchResult:
        """复用同一 GP 循环搜索显式多期限 source，默认等权 MeanRankIC。

        原表达式复杂度只扣一次；custom 使用原有多期评分契约。调用方负责
        source 和后端生命周期。每代及最后一次评分的原 root 顺序 Sum/Count、
        排名和遗传操作保留在 last_search_stats；history 延续单期接口，仅含
        进化前各代的最优项，最后一次评分仍参与全程最优。

        原始统计只在本次搜索中复用；generation_context 可能改变窗口，
        因此设置该回调时关闭缓存。
        """
        from .target_bank import MultiTargetSource
        from .metrics import MetricStats
        from .fitness import rank_multitarget_candidates

        self.last_search_stats = None
        if not isinstance(source, MultiTargetSource):
            raise TypeError("multi-target search requires an explicit MultiTargetSource")
        if not 0 < config.elite_size <= config.population_size:
            raise ValueError("invalid elite size")
        if generation_context is not None and not callable(generation_context):
            raise TypeError("generation_context must be a context-manager factory")
        if type(cache_scores) is not bool:
            raise TypeError("cache_scores must be bool")
        if type(reuse_inputs) is not bool:
            raise TypeError("reuse_inputs must be bool")
        role = getattr(source, "role", getattr(source.source, "role", None))
        if role is not None and role != "search":
            raise ValueError(f"{role} source is evaluation-only, not GP evolution")
        if not terminals or any(t not in source.field_names for t in terminals):
            raise ValueError("terminals must belong to source schema")
        stats = {
            "logical_scoring_calls": 0,
            "scored_roots": 0,
            "evaluated_roots": 0,
            "source_identity": source.source_identity,
            "horizons": source.targets.horizons,
            "algorithm": "fused",
            "generations": [],
        }
        score_cache = {}
        cache_enabled = cache_scores and generation_context is None
        expression_keys = {}

        def score_generation(population):
            lookup_start = time.perf_counter()
            hits = 0
            keys, missing_keys, missing = [], [], []
            if cache_enabled:
                pending = set()
                for candidate in population:
                    previous = expression_keys.get(id(candidate.expr))
                    key = (
                        previous[1]
                        if previous is not None
                        else self._dedup_key(candidate.expr, config)
                    )
                    keys.append(key)
                    if key in score_cache:
                        hits += 1
                    elif key not in pending:
                        pending.add(key)
                        missing_keys.append(key)
                        missing.append(candidate)
            else:
                missing = population
            lookup_seconds = time.perf_counter() - lookup_start

            elapsed = {"compile_seconds": 0.0, "prepare_seconds": 0.0, "score_seconds": 0.0}
            if missing:
                raw, elapsed = self._score_multitarget_raw(source, missing, config)
                if cache_enabled:
                    # 每个条目独立持有 Sum/Count，避免下一批覆盖设备输出。
                    for key, values in zip(missing_keys, raw.stats, strict=True):
                        score_cache[key] = values.copy()

            if cache_enabled:
                restore_start = time.perf_counter()
                raw = MetricStats(
                    source.targets.horizons, np.stack([score_cache[key] for key in keys])
                )
                lookup_seconds += time.perf_counter() - restore_start
            ranking = rank_multitarget_candidates(
                population,
                raw,
                weights=weights,
                complexity_penalty=config.complexity_penalty,
                custom=custom,
                top_k=top_k,
            )
            ranked, report = _multitarget_generation_report(
                ranking, stats["logical_scoring_calls"], source.lane_count
            )
            report["cache"] = {
                "enabled": cache_enabled,
                "hits": hits,
                "misses": len(population) - hits,
                "unique_misses": len(missing),
                "lookup_seconds": lookup_seconds,
                **elapsed,
            }
            stats["generations"].append(report)
            stats["logical_scoring_calls"] += 1
            stats["scored_roots"] += len(population)
            stats["evaluated_roots"] += len(missing)
            return ranked

        session_factory = getattr(self.plan_backend, "multitarget_input_session", None)
        if session_factory is not None and reuse_inputs and generation_context is None:
            session = session_factory(source)
        else:
            if not reuse_inputs:
                reason = "disabled_by_caller"
            elif generation_context is not None:
                reason = "generation_context"
            else:
                reason = "backend_has_no_input_session"
            session = nullcontext({"enabled": False, "reason": reason})
        with session as input_info:
            stats["input_session"] = input_info
            result = self._search_impl(
                terminals,
                config,
                generation_context,
                population_scorer=score_generation,
                timings=stats,
                expression_keys=expression_keys if cache_enabled else None,
            )
        self.last_search_stats = stats
        return result

    def search_indexed(
        self,
        source,
        terminals: list[str],
        config: SearchConfig,
        *,
        generation_context=None,
        capture_generations=False,
    ) -> SearchResult:
        """在全部数据 lane 上搜索一组因子。

        调用方负责保持 source 打开；每代评分按批读取完整 lane。
        """
        self.last_search_stats = None
        if generation_context is not None and not callable(generation_context):
            raise TypeError("generation_context must be a context-manager factory")
        if not 0 < config.elite_size <= config.population_size:
            raise ValueError("invalid elite size")
        role = getattr(source, "role", None)
        if role is not None and role != "search":
            raise ValueError(f"{role} source is evaluation-only, not GP evolution")
        if not terminals or any(t not in source.field_names for t in terminals):
            raise ValueError("terminals must belong to source schema")
        return self._search_single(
            source,
            None,
            terminals,
            config,
            generation_context,
            capture_generations=bool(capture_generations),
        )


def _multitarget_generation_report(ranking, generation, lanes):
    """保留原 Root 次序统计，按稳定排名组装本代诊断；不修改评分结果。"""
    ordered = []
    ranked = []
    for index in ranking.order:
        candidate = ranking.population[index]
        score = float(ranking.fitness[index])
        ranked.append((score, candidate))
        ordered.append(
            {
                "root_index": index,
                "expression": str(candidate.expr),
                "key": candidate.key,
                "score": score if np.isfinite(score) else None,
                "complexity": candidate.expr.complexity,
                "operation": candidate.operation,
                "parents": candidate.parents,
            }
        )
    return ranked, {
        "generation": generation,
        "lanes": lanes,
        "ordered": ordered,
        "metric_stats": ranking.stats.stats.tolist(),
        "top_indices": ranking.top_indices,
        "horizon_top_indices": ranking.horizon_top_indices,
    }
