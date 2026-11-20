from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factor_engine.data import Panel
from factor_engine.expression import Expr


ALL_DSL_OPERATORS = {
    "terminal",
    "const",
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
    "delay",
    "diff",
    "ema",
    "roll_sum",
    "roll_mean",
    "roll_std",
    "roll_min",
    "roll_max",
    "cov",
    "corr",
}


@dataclass(frozen=True, slots=True)
class ExpressionCase:
    operator: str
    expression: Expr


@dataclass(frozen=True, slots=True)
class CorrectnessCase:
    name: str
    panel: Panel
    expressions: tuple[ExpressionCase, ...]
    windows: tuple[int, ...]
    has_nan: bool
    has_inf: bool
    has_zero: bool


def build_case(name: str) -> CorrectnessCase:
    """构造固定的对照数据，让各后端计算同一组表达式。"""
    from factor_engine.expression import (
        abs_,
        add,
        constant,
        corr,
        cov,
        delay,
        diff,
        ema,
        maximum,
        minimum,
        mul,
        neg,
        rolling_max,
        rolling_mean,
        rolling_min,
        rolling_std,
        rolling_sum,
        safe_div,
        sign,
        signed_log1p,
        signed_sqrt,
        sub,
        terminal,
    )

    if name == "normal":
        shape, seed, windows = (2, 3, 96), 1101, (5, 20)
    elif name == "edge":
        shape, seed, windows = (2, 3, 32), 1102, (5, 20)
    elif name == "short":
        shape, seed, windows = (2, 3, 4), 1103, (5, 20)
    elif name == "long_window":
        shape, seed, windows = (2, 3, 1024), 1104, (300, 900)
    else:
        raise KeyError(f"unknown correctness case {name!r}")

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.25, size=shape).astype(np.float32)
    y = (0.35 * x + rng.normal(0.0, 0.8, size=shape)).astype(np.float32)
    valid = np.ones(shape, dtype=bool)
    has_nan = has_inf = has_zero = False
    if name == "edge":
        x[0, 0, 0] = np.float32(-4.0)
        x[0, 0, 1] = np.float32(0.0)
        x[0, 0, 2] = np.float32(np.nan)
        x[0, 1, 7] = np.float32(np.inf)
        y[0, 0, 1] = np.float32(0.0)
        y[1, 2, 11] = np.float32(-np.inf)
        valid[1, 0, 5] = False
        has_nan = has_inf = has_zero = True

    panel = Panel({"x": x, "y": y}, valid)
    a, b, c = terminal("x"), terminal("y"), constant(-2.5)
    primary_window, secondary_window = windows
    expressions = (
        ExpressionCase("terminal", a),
        ExpressionCase("const", add(a, c)),
        ExpressionCase("neg", neg(a)),
        ExpressionCase("abs", abs_(a)),
        ExpressionCase("sign", sign(a)),
        ExpressionCase("signed_sqrt", signed_sqrt(a)),
        ExpressionCase("signed_log1p", signed_log1p(a)),
        ExpressionCase("add", add(a, b)),
        ExpressionCase("sub", sub(a, b)),
        ExpressionCase("mul", mul(a, b)),
        ExpressionCase("safe_div", safe_div(a, b)),
        ExpressionCase("min", minimum(a, b)),
        ExpressionCase("max", maximum(a, b)),
        ExpressionCase("delay", delay(a, 2)),
        ExpressionCase("diff", diff(a, 2)),
        ExpressionCase("ema", ema(a, 5)),
        ExpressionCase("roll_sum", rolling_sum(a, primary_window)),
        ExpressionCase("roll_mean", rolling_mean(a, secondary_window)),
        ExpressionCase("roll_std", rolling_std(a, primary_window)),
        ExpressionCase("roll_min", rolling_min(a, secondary_window)),
        ExpressionCase("roll_max", rolling_max(a, primary_window)),
        ExpressionCase("cov", cov(a, b, secondary_window)),
        ExpressionCase("corr", corr(a, b, primary_window)),
    )
    assert {item.operator for item in expressions} == ALL_DSL_OPERATORS
    return CorrectnessCase(name, panel, expressions, windows, has_nan, has_inf, has_zero)
