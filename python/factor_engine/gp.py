from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Callable, Iterator
import numpy as np

from .dsl import (
    BINARY_NUMERIC_OPS,
    PAIR_ROLLING_OPS,
    ROLLING_UNARY_OPS,
    TEMPORAL_PARAM_OPS,
    UNARY_NUMERIC_OPS,
    ValueType,
)
from .expression import Expr, constant, make_expr, terminal

INITIAL_ATTEMPTS_PER_SLOT = 200
SCALAR_UNARY_PROBABILITY = 0.45


class SeriesFamily(IntEnum):
    UNARY = 0
    BINARY = 1
    ROLLING = 2
    TEMPORAL = 3
    ROLLING_PAIR = 4


@dataclass(frozen=True, slots=True)
class GPConfig:
    population_size: int = 128
    max_depth: int = 5
    max_nodes: int = 96
    seed: int = 1
    windows: tuple[int, ...] = (5, 10, 20, 60, 120)
    lags: tuple[int, ...] = (1, 2, 5, 10, 20)
    scalar_constants: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.25, 0.5, 1.0, 2.0)
    terminal_probability: float = 0.20
    scalar_leaf_probability: float = 0.35
    tournament_size: int = 5
    crossover_probability: float = 0.45
    subtree_mutation_probability: float = 0.25
    point_mutation_probability: float = 0.12
    hoist_mutation_probability: float = 0.08
    reproduction_probability: float = 0.10

    def __post_init__(self) -> None:
        ps = (
            self.crossover_probability,
            self.subtree_mutation_probability,
            self.point_mutation_probability,
            self.hoist_mutation_probability,
            self.reproduction_probability,
        )
        if self.population_size <= 0 or self.max_depth <= 0 or self.max_nodes <= 0:
            raise ValueError("population_size/max_depth/max_nodes must be positive")
        if type(self.tournament_size) is not int or self.tournament_size <= 0:
            raise ValueError("tournament_size must be a positive integer")
        if any(
            not 0.0 <= p <= 1.0
            for p in (*ps, self.terminal_probability, self.scalar_leaf_probability)
        ):
            raise ValueError("GP probabilities must be finite and between zero and one")
        if abs(sum(ps) - 1.0) > 1e-9:
            raise ValueError("genetic operator probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class Candidate:
    expr: Expr
    key: tuple
    parents: tuple[tuple, ...] = ()
    operation: str = "seed"


def _pick(rng, choices):
    """均匀选取一个值；与无权重标量 choice 消耗相同随机流，省去数组转换。"""
    return choices[int(rng.integers(len(choices)))]


class GeneticProgram:
    """生成表达式树，并对树执行交叉和变异。"""

    def __init__(self, terminals: list[str] | tuple[str, ...], config: GPConfig):
        if not terminals:
            raise ValueError("at least one terminal is required")
        self.terminals = tuple(terminals)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.pair_windows = tuple(window for window in config.windows if window > 1)

    def _stop(self, rng: np.random.Generator, depth: int, method: str) -> bool:
        if depth <= 0:
            return True
        if method == "full":
            return False
        if method != "grow":
            raise ValueError(f"unknown grow method {method!r}")
        return bool(rng.random() < self.config.terminal_probability)

    def generate(
        self,
        rng: np.random.Generator,
        depth: int,
        target: ValueType = ValueType.SERIES,
        method: str = "grow",
    ) -> Expr:
        if target is ValueType.SCALAR:
            if self._stop(rng, depth, method) or rng.random() < self.config.scalar_leaf_probability:
                return constant(float(_pick(rng, self.config.scalar_constants)))
            if rng.random() < SCALAR_UNARY_PROBABILITY:
                op = _pick(rng, UNARY_NUMERIC_OPS)
                return make_expr(op, self.generate(rng, depth - 1, ValueType.SCALAR, method))
            op = _pick(rng, BINARY_NUMERIC_OPS)
            return make_expr(
                op,
                self.generate(rng, depth - 1, ValueType.SCALAR, method),
                self.generate(rng, depth - 1, ValueType.SCALAR, method),
            )

        if self._stop(rng, depth, method):
            return terminal(_pick(rng, self.terminals))

        family_count = len(SeriesFamily) if self.pair_windows else int(SeriesFamily.ROLLING_PAIR)
        family = SeriesFamily(rng.integers(0, family_count))
        if family is SeriesFamily.UNARY:
            op = _pick(rng, UNARY_NUMERIC_OPS)
            return make_expr(op, self.generate(rng, depth - 1, ValueType.SERIES, method))
        if family is SeriesFamily.BINARY:
            op = _pick(rng, BINARY_NUMERIC_OPS)
            combo = int(rng.integers(0, 3))
            left_type, right_type = (
                (ValueType.SERIES, ValueType.SERIES),
                (ValueType.SERIES, ValueType.SCALAR),
                (ValueType.SCALAR, ValueType.SERIES),
            )[combo]
            return make_expr(
                op,
                self.generate(rng, depth - 1, left_type, method),
                self.generate(rng, depth - 1, right_type, method),
            )
        if family is SeriesFamily.ROLLING:
            op = _pick(rng, ROLLING_UNARY_OPS)
            return make_expr(
                op,
                self.generate(rng, depth - 1, ValueType.SERIES, method),
                param=_pick(rng, self.config.windows),
            )
        if family is SeriesFamily.TEMPORAL:
            op = _pick(rng, TEMPORAL_PARAM_OPS)
            p = _pick(rng, self.config.windows if op == "ema" else self.config.lags)
            return make_expr(op, self.generate(rng, depth - 1, ValueType.SERIES, method), param=p)
        op = _pick(rng, PAIR_ROLLING_OPS)
        return make_expr(
            op,
            self.generate(rng, depth - 1, ValueType.SERIES, method),
            self.generate(rng, depth - 1, ValueType.SERIES, method),
            param=_pick(rng, self.pair_windows),
        )

    def initial_population(
        self, key: Callable[[Expr], tuple] | None = None, *, accepted=None
    ) -> list[Candidate]:
        key = (lambda expr: expr.key) if key is None else key
        result: list[Candidate] = []
        seen: set[tuple] = set()
        attempts = 0
        # 随机选树深度，并交替使用 full 和 grow 两种生成方式。
        while (
            len(result) < self.config.population_size
            and attempts < self.config.population_size * INITIAL_ATTEMPTS_PER_SLOT
        ):
            attempts += 1
            target_depth = int(self.rng.integers(1, self.config.max_depth + 1))
            method = "full" if attempts % 2 == 0 else "grow"
            expr = self.generate(self.rng, target_depth, ValueType.SERIES, method)
            if not self._within_limits(expr):
                continue
            identity = key(expr)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(Candidate(expr, expr.key))
            if accepted is not None:
                accepted(expr, identity)
        # 随机碰撞太多时，按语法枚举剩余表达式，直到填满或确认空间耗尽。
        self.fill_unique(result, seen, self.config.population_size, key, accepted=accepted)
        return result

    def fill_unique(
        self,
        result: list[Candidate],
        seen: set[tuple],
        size: int,
        key: Callable[[Expr], tuple],
        *,
        accepted=None,
    ) -> None:
        """按节点数从小到大补充未出现过的表达式。"""
        if len(result) >= size:
            return
        # 每个节点最多两个子节点，深度固定时树的节点数也有上限。
        max_nodes = min(self.config.max_nodes, (1 << (self.config.max_depth + 1)) - 1)
        for nodes in range(1, max_nodes + 1):
            for expr in self._enumerate(ValueType.SERIES, nodes, self.config.max_depth + 1):
                identity = key(expr)
                if identity in seen:
                    continue
                seen.add(identity)
                result.append(Candidate(expr, expr.key, operation="enumerated"))
                if accepted is not None:
                    accepted(expr, identity)
                if len(result) == size:
                    return
        raise ValueError(
            f"expression space exhausted: requested {size} distinct candidates, found {len(result)}"
        )

    def _enumerate(self, target: ValueType, nodes: int, depth: int) -> Iterator[Expr]:
        if nodes == 1:
            if target is ValueType.SERIES:
                yield from (terminal(name) for name in self.terminals)
            else:
                yield from (constant(value) for value in self.config.scalar_constants)
            return
        if depth <= 1 or nodes < 2:
            return
        for child in self._enumerate(target, nodes - 1, depth - 1):
            for op in UNARY_NUMERIC_OPS:
                yield make_expr(op, child)
            if target is ValueType.SERIES:
                for op in ROLLING_UNARY_OPS:
                    for window in self.config.windows:
                        yield make_expr(op, child, param=window)
                for op in TEMPORAL_PARAM_OPS:
                    for param in self.config.windows if op == "ema" else self.config.lags:
                        yield make_expr(op, child, param=param)
        type_pairs = (
            ((ValueType.SCALAR, ValueType.SCALAR),)
            if target is ValueType.SCALAR
            else (
                (ValueType.SERIES, ValueType.SERIES),
                (ValueType.SERIES, ValueType.SCALAR),
                (ValueType.SCALAR, ValueType.SERIES),
            )
        )
        for left_nodes in range(1, nodes - 1):
            right_nodes = nodes - 1 - left_nodes
            for left_type, right_type in type_pairs:
                for left in self._enumerate(left_type, left_nodes, depth - 1):
                    for right in self._enumerate(right_type, right_nodes, depth - 1):
                        for op in BINARY_NUMERIC_OPS:
                            yield make_expr(op, left, right)
                        if left_type is ValueType.SERIES and right_type is ValueType.SERIES:
                            for op in PAIR_ROLLING_OPS:
                                for window in self.config.windows:
                                    if window > 1:
                                        yield make_expr(op, left, right, param=window)

    def _within_limits(self, expr: Expr) -> bool:
        return expr.depth <= self.config.max_depth + 1 and expr.complexity <= self.config.max_nodes

    def subtree_mutate(self, expr: Expr, rng: np.random.Generator | None = None) -> Expr:
        rng = self.rng if rng is None else rng
        paths = _paths(expr)
        path = paths[int(rng.integers(len(paths)))]
        old = _at(expr, path)
        remaining = max(0, self.config.max_depth - len(path) - 1)
        replacement = self.generate(rng, remaining, old.value_type, "grow")
        child = _replace(expr, path, replacement)
        return child if self._within_limits(child) else expr

    def point_mutate(self, expr: Expr, rng: np.random.Generator | None = None) -> Expr:
        rng = self.rng if rng is None else rng
        candidates = [p for p in _paths(expr) if _at(expr, p).op not in {"terminal", "const"}]
        if not candidates:
            return expr
        path = candidates[int(rng.integers(len(candidates)))]
        node = _at(expr, path)
        if node.op in UNARY_NUMERIC_OPS:
            pool = [x for x in UNARY_NUMERIC_OPS if x != node.op]
            repl = make_expr(_pick(rng, pool), *node.args)
        elif node.op in BINARY_NUMERIC_OPS:
            pool = [x for x in BINARY_NUMERIC_OPS if x != node.op]
            repl = make_expr(_pick(rng, pool), *node.args)
        elif node.op in ROLLING_UNARY_OPS:
            pool = [x for x in ROLLING_UNARY_OPS if x != node.op]
            repl = make_expr(_pick(rng, pool), *node.args, param=node.param)
        elif node.op in PAIR_ROLLING_OPS:
            repl = make_expr("corr" if node.op == "cov" else "cov", *node.args, param=node.param)
        elif node.op in {"delay", "diff"}:
            repl = make_expr(
                "diff" if node.op == "delay" else "delay", *node.args, param=node.param
            )
        elif node.op == "ema":
            repl = make_expr("ema", *node.args, param=_pick(rng, self.config.windows))
        else:
            return expr
        child = _replace(expr, path, repl)
        return child if self._within_limits(child) else expr

    def hoist_mutate(self, expr: Expr, rng: np.random.Generator | None = None) -> Expr:
        rng = self.rng if rng is None else rng
        paths = _paths(expr)
        outer_path = paths[int(rng.integers(len(paths)))]
        subtree = _at(expr, outer_path)
        inner_paths = [
            p for p in _paths(subtree) if p and _at(subtree, p).value_type is subtree.value_type
        ]
        if not inner_paths:
            return expr
        inner = inner_paths[int(rng.integers(len(inner_paths)))]
        child = _replace(expr, outer_path, _at(subtree, inner))
        return child if self._within_limits(child) else expr

    def crossover(
        self, a: Expr, b: Expr, rng: np.random.Generator | None = None
    ) -> tuple[Expr, Expr]:
        rng = self.rng if rng is None else rng
        selection = _crossover_selection(a, b, rng)
        if selection is None:
            return a, b
        path_a, sub_a, path_b, sub_b = selection
        ca = _replace(a, path_a, sub_b)
        cb = _replace(b, path_b, sub_a)
        return (
            ca if self._within_limits(ca) else a,
            cb if self._within_limits(cb) else b,
        )

    def crossover_one(self, a: Expr, b: Expr, rng: np.random.Generator | None = None) -> Expr:
        """搜索只使用首子代；选择顺序和随机消耗与双子代接口一致。"""
        rng = self.rng if rng is None else rng
        selection = _crossover_selection(a, b, rng)
        if selection is None:
            return a
        path_a, _, _, sub_b = selection
        child = _replace(a, path_a, sub_b)
        return child if self._within_limits(child) else a

    def tournament(
        self, ranked: list[tuple[float, Candidate]], rng: np.random.Generator
    ) -> Candidate:
        k = min(self.config.tournament_size, len(ranked))
        idx = rng.integers(0, len(ranked), size=k)
        return max((ranked[int(i)] for i in idx), key=lambda pair: pair[0])[1]


def _path_nodes(expr: Expr, prefix: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], Expr]]:
    yield prefix, expr
    for index, child in enumerate(expr.args):
        yield from _path_nodes(child, prefix + (index,))


def _crossover_selection(a: Expr, b: Expr, rng: np.random.Generator):
    # 按先序收集路径和节点，避免逐路径再次从根遍历。
    left = list(_path_nodes(a))
    right: dict[ValueType, list[tuple[tuple[int, ...], Expr]]] = {
        ValueType.SERIES: [],
        ValueType.SCALAR: [],
    }
    for path, node in _path_nodes(b):
        right[node.value_type].append((path, node))
    path_a, sub_a = left[int(rng.integers(len(left)))]
    options = right[sub_a.value_type]
    if not options:
        return None
    path_b, sub_b = options[int(rng.integers(len(options)))]
    return path_a, sub_a, path_b, sub_b


def _paths(expr: Expr, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    result = [prefix]
    for i, child in enumerate(expr.args):
        result.extend(_paths(child, prefix + (i,)))
    return result


def _at(expr: Expr, path: tuple[int, ...]) -> Expr:
    cur = expr
    for index in path:
        cur = cur.args[index]
    return cur


def _replace(expr: Expr, path: tuple[int, ...], replacement: Expr) -> Expr:
    if not path:
        if replacement.value_type is not expr.value_type:
            raise TypeError("replacement must preserve subtree output type")
        return replacement
    index = path[0]
    args = list(expr.args)
    args[index] = _replace(args[index], path[1:], replacement)
    return make_expr(expr.op, *args, param=expr.param)
