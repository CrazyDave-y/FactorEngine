import numpy as np
import pytest
from factor_engine.canonicalize import canonicalize
from factor_engine.dsl import ValueType
from factor_engine.expression import (
    abs_,
    add,
    constant,
    make_expr,
    mul,
    neg,
    rolling_mean,
    terminal,
)
from factor_engine.ir import Node, compile_expressions, constant_param, terminal_param, window_param


def test_commutative_order_and_idempotence():
    a, b = terminal("a"), terminal("b")
    x = canonicalize(add(b, a))
    y = canonicalize(add(a, b))
    assert x.key == y.key
    assert canonicalize(x).key == x.key


def test_strict_safe_rewrites_only():
    x = terminal("x")
    assert canonicalize(neg(neg(x))).key == x.key
    assert canonicalize(abs_(abs_(x))).key == abs_(x).key
    assert canonicalize(add(constant(2), constant(3))).param == 5.0


def test_scalar_folding_uses_native_float32_rounding_at_each_step():
    large = constant(16777216)
    expr = make_expr("sub", add(large, constant(1)), large)
    assert canonicalize(expr).param == float(np.float32(0))


def test_safe_div_fold_uses_float32_denominator_threshold():
    expr = make_expr("safe_div", constant(1), constant(1.00000001e-12))
    assert canonicalize(expr).op == "safe_div"


def test_sign_fold_preserves_negative_sign():
    assert canonicalize(make_expr("sign", constant(-2))).param == -1


@pytest.mark.parametrize(
    "expr",
    [
        make_expr("mul", constant(1e-30), constant(1e-20)),
        make_expr("add", constant(1e-100), constant(1)),
    ],
)
def test_scalar_folding_does_not_inherit_numpy_underflow_errors(expr):
    with np.errstate(under="raise"):
        folded = canonicalize(expr)
        assert np.geterr()["under"] == "raise"
    assert folded.op == "const"
    assert folded.param == (0 if expr.op == "mul" else 1)


def test_population_cse_shares_subexpression():
    x, y = terminal("x"), terminal("y")
    shared = rolling_mean(x, 5)
    exprs = [add(shared, y), add(y, shared)]
    raw = compile_expressions(exprs, canonicalize_exprs=False, cse=False)
    opt = compile_expressions(exprs, canonicalize_exprs=True, cse=True)
    assert raw.unique_nodes == raw.ast_nodes
    assert opt.unique_nodes < raw.unique_nodes
    assert opt.root_ids[0] == opt.root_ids[1]


def test_ir_parameter_guards_reject_cross_operator_parameter_mismatches() -> None:
    terminal = Node(0, "terminal", (), ValueType.SERIES, "close")
    constant = Node(1, "const", (), ValueType.SCALAR, 1.25)
    window = Node(2, "roll_mean", (0,), ValueType.SERIES, 20)

    assert terminal_param(terminal) == "close"
    assert constant_param(constant) == 1.25
    assert window_param(window) == 20

    with pytest.raises(TypeError, match="terminal name"):
        terminal_param(Node(3, "terminal", (), ValueType.SERIES, 3))
    with pytest.raises(TypeError, match="finite numeric constant"):
        constant_param(Node(4, "const", (), ValueType.SCALAR, "1.25"))
    with pytest.raises(TypeError, match="positive integer window"):
        window_param(Node(5, "roll_mean", (0,), ValueType.SERIES, True))


def test_buffer_slots_are_reused_after_last_use():
    x = terminal("x")
    a = rolling_mean(x, 5)
    b = add(a, x)
    c = mul(b, x)
    plan = compile_expressions([c], canonicalize_exprs=True, cse=True)
    series_nonterm = [
        n for n in plan.nodes if n.value_type.value == "series" and n.op != "terminal"
    ]
    assert plan.slot_count <= len(series_nonterm)
    assert all(n.slot >= 0 for n in series_nonterm)


def test_type_rules_and_scalar_broadcast():
    x = terminal("x")
    c = constant(0.5)
    assert add(x, c).value_type is ValueType.SERIES
    assert add(c, c).value_type is ValueType.SCALAR
    with pytest.raises(TypeError):
        rolling_mean(c, 5)
