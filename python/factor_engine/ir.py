from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import hashlib
import math
from typing import TypeAlias

from .canonicalize import canonicalize
from .dsl import INTEGER_PARAMETER_OPS, ValueType
from .expression import Expr, _numeric_param


NodeParam: TypeAlias = str | int | float | None


@dataclass(frozen=True, slots=True)
class Node:
    node_id: int
    op: str
    inputs: tuple[int, ...]
    value_type: ValueType
    param: NodeParam = None
    slot: int = -1


def terminal_param(node: Node) -> str:
    """校验 terminal 节点并返回字段名。"""
    if node.op != "terminal" or not isinstance(node.param, str) or not node.param:
        raise TypeError("terminal node requires a non-empty terminal name")
    return node.param


def constant_param(node: Node) -> float:
    """返回有限数值常数；不接受布尔值或数字字符串。"""
    value = node.param
    if node.op != "const" or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("constant node requires a finite numeric constant")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TypeError("constant node requires a finite numeric constant")
    return numeric


def window_param(node: Node) -> int:
    """返回时间算子的窗口或滞后长度。"""
    value = node.param
    if node.op not in INTEGER_PARAMETER_OPS or type(value) is not int or value <= 0:
        raise TypeError(f"{node.op} requires a positive integer window")
    if node.op in {"cov", "corr"} and value <= 1:
        raise TypeError(f"{node.op} requires a window greater than one")
    return value


def native_int_param(node: Node) -> int:
    """生成原生节点的整数参数；不用窗口的算子填 0。"""
    return window_param(node) if node.op in INTEGER_PARAMETER_OPS else 0


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    nodes: tuple[Node, ...]
    root_ids: tuple[int, ...]
    ast_nodes: int
    unique_nodes: int
    canonicalized: bool
    cse_enabled: bool
    last_use: tuple[int, ...]
    slot_count: int

    @property
    def fingerprint(self) -> str:
        """按需计算执行计划的存档标识。"""
        return _fingerprint(self.nodes, self.root_ids)

    @property
    def cse_ratio(self) -> float:
        return 1.0 - self.unique_nodes / self.ast_nodes if self.ast_nodes else 0.0


def _assign_slots(
    nodes: list[Node], roots: tuple[int, ...]
) -> tuple[list[Node], tuple[int, ...], int]:
    node_count = len(nodes)
    last_use = list(range(node_count))
    for node_id, node in enumerate(nodes):
        for child in node.inputs:
            last_use[child] = max(last_use[child], node_id)
    for root in roots:
        last_use[root] = node_count

    release_at: list[list[int]] = [[] for _ in range(node_count + 2)]
    free_slots: list[int] = []
    slot_count = 0
    planned: list[Node] = []

    for node_id, node in enumerate(nodes):
        # 同一边界按节点创建顺序归还槽位，固定 LIFO 复用顺序。
        free_slots.extend(release_at[node_id])
        slot = -1
        if node.value_type is ValueType.SERIES and node.op != "terminal":
            if free_slots:
                slot = free_slots.pop()
            else:
                slot = slot_count
                slot_count += 1
            release_at[last_use[node_id] + 1].append(slot)
        planned.append(
            Node(
                node_id=node.node_id,
                op=node.op,
                inputs=node.inputs,
                value_type=node.value_type,
                param=node.param,
                slot=slot,
            )
        )
    return planned, tuple(last_use), slot_count


def _fingerprint(nodes: Sequence[Node], roots: tuple[int, ...]) -> str:
    payload = repr(
        tuple((n.op, n.inputs, n.value_type.value, n.param, n.slot) for n in nodes)
    ) + repr(roots)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_expressions(
    expressions: list[Expr] | tuple[Expr, ...],
    *,
    canonicalize_exprs: bool = True,
    cse: bool = True,
) -> ExecutionPlan:
    normalized = []
    for expression in expressions:
        if canonicalize_exprs:
            expression = canonicalize(expression)
        normalized.append(expression)

    raw_nodes: list[Node] = []
    key_to_id: dict[tuple, int] = {}
    key_memo: dict[int, tuple[Expr, tuple]] = {}

    def structural_key(expr: Expr) -> tuple:
        cached = key_memo.get(id(expr))
        if cached is not None:
            return cached[1]
        parameter = (
            _numeric_param(expr.param, operator="const") if expr.op == "const" else expr.param
        )
        key = (
            expr.op,
            expr.value_type.value,
            parameter,
            tuple(structural_key(child) for child in expr.args),
        )
        key_memo[id(expr)] = (expr, key)
        return key

    def emit(expr: Expr) -> int:
        key = structural_key(expr) if cse else None
        if cse and key in key_to_id:
            return key_to_id[key]
        inputs = tuple(emit(child) for child in expr.args)
        node_id = len(raw_nodes)
        raw_nodes.append(Node(node_id, expr.op, inputs, expr.value_type, expr.param))
        if key is not None:
            key_to_id[key] = node_id
        return node_id

    roots = tuple(emit(expression) for expression in normalized)
    nodes, last_use, slot_count = _assign_slots(raw_nodes, roots)
    return ExecutionPlan(
        nodes=tuple(nodes),
        root_ids=roots,
        ast_nodes=sum(expression.complexity for expression in normalized),
        unique_nodes=len(nodes),
        canonicalized=canonicalize_exprs,
        cse_enabled=cse,
        last_use=last_use,
        slot_count=slot_count,
    )
