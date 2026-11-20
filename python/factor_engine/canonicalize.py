from __future__ import annotations

import math
import numpy as np
from .dsl import SAFE_DIVISION_EPSILON, COMMUTATIVE_OPS

from .expression import Expr, constant, make_expr


def _const_value(expr: Expr) -> float:
    if expr.op != "const" or expr.param is None:
        raise TypeError("constant expression requires a numeric parameter")
    return float(expr.param)


def _window_value(expr: Expr) -> int:
    if type(expr.param) is not int or expr.param <= 0:
        raise TypeError(f"{expr.op} requires a positive integer window")
    return expr.param


def _is_const(expr: Expr, value: float) -> bool:
    return expr.op == "const" and _const_value(expr) == float(value)


def _scalar_fold(op: str, args: tuple[Expr, ...]) -> Expr | None:
    if not args or any(arg.op != "const" for arg in args):
        return None
    # 原生节点按 float32 保存常数，每步运算后都会舍入，折叠时也要遵守这个顺序。
    with np.errstate(all="ignore"):
        values = [np.float32(_const_value(arg)) for arg in args]
    if not all(np.isfinite(value) for value in values):
        return None
    try:
        with np.errstate(all="ignore"):
            if op == "neg":
                folded_value = -values[0]
            elif op == "abs":
                folded_value = abs(values[0])
            elif op == "sign":
                folded_value = np.float32(int(values[0] > 0) - int(values[0] < 0))
            elif op == "signed_sqrt":
                folded_value = np.copysign(np.sqrt(abs(values[0])), values[0])
            elif op == "signed_log1p":
                folded_value = np.float32(
                    math.copysign(math.log1p(float(abs(values[0]))), float(values[0]))
                )
            elif op == "add":
                folded_value = values[0] + values[1]
            elif op == "sub":
                folded_value = values[0] - values[1]
            elif op == "mul":
                folded_value = values[0] * values[1]
            elif op == "safe_div":
                if abs(values[1]) <= np.float32(SAFE_DIVISION_EPSILON):
                    return None
                folded_value = values[0] / values[1]
            elif op == "min":
                folded_value = min(values)
            elif op == "max":
                folded_value = max(values)
            else:
                return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return constant(float(folded_value)) if np.isfinite(folded_value) else None


def canonicalize(expr: Expr) -> Expr:
    if not expr.args:
        if expr.op == "const":
            # 标量常数的结构 key 不区分 +0 和 -0。
            value = _const_value(expr)
            return constant(0.0 if value == 0.0 else value)
        return expr

    args = tuple(canonicalize(arg) for arg in expr.args)
    if expr.op in COMMUTATIVE_OPS:
        args = tuple(sorted(args, key=lambda arg: repr(arg.key)))

    folded = _scalar_fold(expr.op, args)
    if folded is not None:
        return folded

    # 后端已将序列中的无穷值转为 NaN，下面只做保留当前计算语义的改写。
    if expr.op == "neg" and args[0].op == "neg":
        return args[0].args[0]
    if expr.op == "abs" and args[0].op in {"abs", "neg"}:
        return args[0] if args[0].op == "abs" else make_expr("abs", args[0].args[0])
    if expr.op in {"min", "max"} and args[0].key == args[1].key:
        return args[0]
    if expr.op == "mul":
        if _is_const(args[0], 1.0):
            return args[1]
        if _is_const(args[1], 1.0):
            return args[0]
    if expr.op == "sub" and _is_const(args[1], 0.0):
        return args[0]
    if expr.op == "safe_div" and _is_const(args[1], 1.0):
        return args[0]
    if expr.op == "delay" and args[0].op == "delay":
        return make_expr(
            "delay", args[0].args[0], param=_window_value(args[0]) + _window_value(expr)
        )

    return make_expr(expr.op, *args, param=expr.param)
