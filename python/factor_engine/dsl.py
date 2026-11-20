from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class ValueType(str, Enum):
    SERIES = "series"
    SCALAR = "scalar"


class OperatorKind(str, Enum):
    TERMINAL = "terminal"
    CONSTANT = "constant"
    UNARY_NUMERIC = "unary_numeric"
    BINARY_NUMERIC = "binary_numeric"
    SERIES_PARAM = "series_param"
    SERIES_PAIR_PARAM = "series_pair_param"


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    kind: OperatorKind
    arity: int
    parameter: str | None = None
    commutative: bool = False


_SPECS = [
    OperatorSpec("terminal", OperatorKind.TERMINAL, 0, "name"),
    OperatorSpec("const", OperatorKind.CONSTANT, 0, "float"),
    OperatorSpec("neg", OperatorKind.UNARY_NUMERIC, 1),
    OperatorSpec("abs", OperatorKind.UNARY_NUMERIC, 1),
    OperatorSpec("sign", OperatorKind.UNARY_NUMERIC, 1),
    OperatorSpec("signed_sqrt", OperatorKind.UNARY_NUMERIC, 1),
    OperatorSpec("signed_log1p", OperatorKind.UNARY_NUMERIC, 1),
    OperatorSpec("add", OperatorKind.BINARY_NUMERIC, 2, commutative=True),
    OperatorSpec("sub", OperatorKind.BINARY_NUMERIC, 2),
    OperatorSpec("mul", OperatorKind.BINARY_NUMERIC, 2, commutative=True),
    OperatorSpec("safe_div", OperatorKind.BINARY_NUMERIC, 2),
    OperatorSpec("min", OperatorKind.BINARY_NUMERIC, 2, commutative=True),
    OperatorSpec("max", OperatorKind.BINARY_NUMERIC, 2, commutative=True),
    OperatorSpec("delay", OperatorKind.SERIES_PARAM, 1, "lag"),
    OperatorSpec("diff", OperatorKind.SERIES_PARAM, 1, "lag"),
    OperatorSpec("ema", OperatorKind.SERIES_PARAM, 1, "span"),
    OperatorSpec("roll_sum", OperatorKind.SERIES_PARAM, 1, "window"),
    OperatorSpec("roll_mean", OperatorKind.SERIES_PARAM, 1, "window"),
    OperatorSpec("roll_std", OperatorKind.SERIES_PARAM, 1, "window"),
    OperatorSpec("roll_min", OperatorKind.SERIES_PARAM, 1, "window"),
    OperatorSpec("roll_max", OperatorKind.SERIES_PARAM, 1, "window"),
    OperatorSpec(
        "cov",
        OperatorKind.SERIES_PAIR_PARAM,
        2,
        "window",
        commutative=True,
    ),
    OperatorSpec(
        "corr",
        OperatorKind.SERIES_PAIR_PARAM,
        2,
        "window",
        commutative=True,
    ),
]

OPERATORS: Final[dict[str, OperatorSpec]] = {spec.name: spec for spec in _SPECS}

ALIASES: Final[dict[str, str]] = {
    "mean": "roll_mean",
    "std": "roll_std",
    "ts_mean": "roll_mean",
    "ts_std": "roll_std",
    "minimum": "min",
    "maximum": "max",
}

COMMUTATIVE_OPS: Final[frozenset[str]] = frozenset(spec.name for spec in _SPECS if spec.commutative)

# lag、span、window 都是正整数参数；各算子的下限仍由 make_expr 检查。
INTEGER_PARAMETER_OPS: Final[frozenset[str]] = frozenset(
    spec.name
    for spec in _SPECS
    if spec.kind in (OperatorKind.SERIES_PARAM, OperatorKind.SERIES_PAIR_PARAM)
)

UNARY_NUMERIC_OPS: Final[tuple[str, ...]] = tuple(
    spec.name for spec in _SPECS if spec.kind is OperatorKind.UNARY_NUMERIC
)
BINARY_NUMERIC_OPS: Final[tuple[str, ...]] = tuple(
    spec.name for spec in _SPECS if spec.kind is OperatorKind.BINARY_NUMERIC
)
ROLLING_UNARY_OPS: Final[tuple[str, ...]] = (
    "roll_sum",
    "roll_mean",
    "roll_std",
    "roll_min",
    "roll_max",
)
TEMPORAL_PARAM_OPS: Final[tuple[str, ...]] = ("delay", "diff", "ema")
PAIR_ROLLING_OPS: Final[tuple[str, ...]] = ("cov", "corr")


def canonical_op_name(name: str) -> str:
    return ALIASES.get(name, name)


def operator_spec(name: str) -> OperatorSpec:
    name = canonical_op_name(name)
    try:
        return OPERATORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown FactorDSL operator {name!r}") from exc


SAFE_DIVISION_EPSILON = 1e-12
