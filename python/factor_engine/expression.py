from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

from .dsl import INTEGER_PARAMETER_OPS, OperatorKind, ValueType, canonical_op_name, operator_spec


ExpressionParam: TypeAlias = str | int | float | None


def _numeric_param(value: ExpressionParam, *, operator: str) -> float:
    if value is None:
        raise TypeError(f"{operator} requires a numeric parameter")
    return float(value)


def _integer_param(value: ExpressionParam, *, operator: str) -> int:
    if value is None:
        raise TypeError(f"{operator} requires an integer parameter")
    return int(value)


@dataclass(frozen=True, slots=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value_type: ValueType = ValueType.SERIES
    param: ExpressionParam = None

    @property
    def key(self) -> tuple:
        """描述这棵表达式树的结构；等价改写由 canonicalize 另外处理。"""
        parameter = (
            _numeric_param(self.param, operator="const") if self.op == "const" else self.param
        )
        return (self.op, self.value_type.value, parameter, tuple(arg.key for arg in self.args))

    @property
    def complexity(self) -> int:
        return 1 + sum(arg.complexity for arg in self.args)

    @property
    def depth(self) -> int:
        return 1 if not self.args else 1 + max(arg.depth for arg in self.args)

    def __str__(self) -> str:
        if self.op == "terminal":
            return str(self.param)
        if self.op == "const":
            return f"{_numeric_param(self.param, operator='const'):g}"
        if self.op in INTEGER_PARAMETER_OPS:
            arguments = ", ".join(
                str(self.args[index]) for index in range(operator_spec(self.op).arity)
            )
            return f"{self.op}({arguments}, {self.param})"
        return f"{self.op}({', '.join(map(str, self.args))})"


def make_expr(op: str, *args: Expr, param: ExpressionParam = None) -> Expr:
    op = canonical_op_name(op)
    spec = operator_spec(op)
    if len(args) != spec.arity:
        raise TypeError(f"{op} expects {spec.arity} arguments, got {len(args)}")

    if spec.kind is OperatorKind.TERMINAL:
        if not isinstance(param, str) or not param:
            raise ValueError("terminal name must be a non-empty string")
        return Expr(op, (), ValueType.SERIES, param)

    if spec.kind is OperatorKind.CONSTANT:
        value = _numeric_param(param, operator=op)
        if not math.isfinite(value):
            raise ValueError("FactorDSL constants must be finite")
        return Expr(op, (), ValueType.SCALAR, value)

    if spec.kind is OperatorKind.UNARY_NUMERIC:
        return Expr(op, tuple(args), args[0].value_type, None)

    if spec.kind is OperatorKind.BINARY_NUMERIC:
        left, right = args
        result_type = (
            ValueType.SERIES
            if ValueType.SERIES in (left.value_type, right.value_type)
            else ValueType.SCALAR
        )
        return Expr(op, tuple(args), result_type, None)

    if spec.kind is OperatorKind.SERIES_PARAM:
        if args[0].value_type is not ValueType.SERIES:
            raise TypeError(f"{op} requires a Series input")
        value = _integer_param(param, operator=op)
        if value <= 0:
            raise ValueError(f"{op} parameter must be positive")
        return Expr(op, tuple(args), ValueType.SERIES, value)

    if spec.kind is OperatorKind.SERIES_PAIR_PARAM:
        if any(arg.value_type is not ValueType.SERIES for arg in args):
            raise TypeError(f"{op} requires two Series inputs")
        value = _integer_param(param, operator=op)
        if value <= 1:
            raise ValueError(f"{op} window must be greater than one")
        return Expr(op, tuple(args), ValueType.SERIES, value)

    raise AssertionError(f"unhandled operator kind {spec.kind}")


def terminal(name: str) -> Expr:
    return make_expr("terminal", param=name)


def constant(value: float) -> Expr:
    return make_expr("const", param=value)


def neg(x: Expr) -> Expr:
    return make_expr("neg", x)


def abs_(x: Expr) -> Expr:
    return make_expr("abs", x)


def sign(x: Expr) -> Expr:
    return make_expr("sign", x)


def signed_sqrt(x: Expr) -> Expr:
    return make_expr("signed_sqrt", x)


def signed_log1p(x: Expr) -> Expr:
    return make_expr("signed_log1p", x)


def add(a: Expr, b: Expr) -> Expr:
    return make_expr("add", a, b)


def sub(a: Expr, b: Expr) -> Expr:
    return make_expr("sub", a, b)


def mul(a: Expr, b: Expr) -> Expr:
    return make_expr("mul", a, b)


def safe_div(a: Expr, b: Expr) -> Expr:
    return make_expr("safe_div", a, b)


def minimum(a: Expr, b: Expr) -> Expr:
    return make_expr("min", a, b)


def maximum(a: Expr, b: Expr) -> Expr:
    return make_expr("max", a, b)


def delay(x: Expr, lag: int = 1) -> Expr:
    return make_expr("delay", x, param=lag)


def diff(x: Expr, lag: int = 1) -> Expr:
    return make_expr("diff", x, param=lag)


def ema(x: Expr, span: int) -> Expr:
    return make_expr("ema", x, param=span)


def rolling_sum(x: Expr, window: int) -> Expr:
    return make_expr("roll_sum", x, param=window)


def rolling_mean(x: Expr, window: int) -> Expr:
    return make_expr("roll_mean", x, param=window)


def rolling_std(x: Expr, window: int) -> Expr:
    return make_expr("roll_std", x, param=window)


def rolling_min(x: Expr, window: int) -> Expr:
    return make_expr("roll_min", x, param=window)


def rolling_max(x: Expr, window: int) -> Expr:
    return make_expr("roll_max", x, param=window)


def cov(a: Expr, b: Expr, window: int) -> Expr:
    return make_expr("cov", a, b, param=window)


def corr(a: Expr, b: Expr, window: int) -> Expr:
    return make_expr("corr", a, b, param=window)
