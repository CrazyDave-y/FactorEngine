import os

import numpy as np
import pytest
from factor_engine.data import Panel
from factor_engine.expression import (
    abs_,
    add,
    corr,
    ema,
    mul,
    rolling_mean,
    rolling_std,
    safe_div,
    terminal,
)
from factor_engine.ir import compile_expressions
from factor_engine.native_plan_backend import NativePlanBackend

from tests.reference.numpy_backend import NumpyPlanBackend


@pytest.mark.skipif(
    not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="set FACTORENGINE_NATIVE_PLAN_LIB"
)
def test_full_native_plan_matches_numpy():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 3, 64)).astype(np.float32)
    y = rng.normal(size=x.shape).astype(np.float32)
    panel = Panel({"x": x, "y": y})
    m = rolling_mean(terminal("x"), 7)
    s = rolling_std(terminal("x"), 11)
    c = corr(terminal("x"), terminal("y"), 9)
    exprs = [abs_(add(m, s)), safe_div(m, c), mul(ema(terminal("x"), 5), terminal("y"))]
    plan = compile_expressions(exprs, canonicalize_exprs=True, cse=True)
    ref = NumpyPlanBackend().evaluate(plan, panel)
    got = NativePlanBackend().evaluate(plan, panel)
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-5, equal_nan=True)


@pytest.mark.skipif(
    not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="set FACTORENGINE_NATIVE_PLAN_LIB"
)
def test_full_native_plan_reuses_caller_owned_workspace():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(2, 3, 64)).astype(np.float32)
    y = rng.normal(size=x.shape).astype(np.float32)
    panel = Panel({"x": x, "y": y})
    exprs = [
        rolling_mean(terminal("x"), 7),
        rolling_std(terminal("x"), 11),
        corr(terminal("x"), terminal("y"), 9),
    ]
    plan = compile_expressions(exprs, canonicalize_exprs=True, cse=True)
    backend = NativePlanBackend()
    first = backend.evaluate(plan, panel)
    second = backend.evaluate(plan, panel)
    assert backend.workspace_allocation_count == 1
    assert backend.workspace_capacity_floats >= plan.slot_count * int(np.prod(panel.shape))
    for a, b in zip(second, first, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="native CPU")
@pytest.mark.parametrize("offset,quantum", [(1e8, 8.0), (1e12, 65536.0)])
def test_native_std_matches_centered_reference_for_large_offset_small_variation(offset, quantum):
    from factor_engine.data import Panel
    from factor_engine.expression import rolling_std, terminal
    from factor_engine.ir import compile_expressions
    from factor_engine.native_plan_backend import NativePlanBackend

    from tests.reference.numpy_backend import NumpyPlanBackend

    rng = np.random.default_rng(317)
    x = np.float32(offset) + np.float32(quantum) * rng.integers(-3, 4, (1, 4, 512)).astype(
        np.float32
    )
    x[0, 1] = np.float32(offset)
    x[0, 2, 79] = np.nan
    x[0, 3, ::83] = np.inf
    panel = Panel({"x": x})
    plan = compile_expressions([rolling_std(terminal("x"), 60)])
    expected = NumpyPlanBackend().evaluate(plan, panel)[0]
    actual = NativePlanBackend().evaluate(plan, panel)[0]
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=8e-5, equal_nan=True)


@pytest.mark.skipif(not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="native CPU")
def test_native_log_uses_one_fp32_rounding_after_double_evaluation():
    from factor_engine.data import Panel
    from factor_engine.expression import constant, signed_log1p, sub, terminal
    from factor_engine.ir import compile_expressions
    from factor_engine.native_plan_backend import NativePlanBackend

    x = np.linspace(-0.005, 0.005, 156000, dtype=np.float32).reshape(1, 400, 390)
    shifted = (x - np.float32(2)).astype(np.float32)
    expected = (
        np.sign(shifted.astype(np.float64)) * np.log1p(np.abs(shifted.astype(np.float64)))
    ).astype(np.float32)
    actual = NativePlanBackend().evaluate(
        compile_expressions([signed_log1p(sub(terminal("x"), constant(2)))]), Panel({"x": x})
    )[0]
    np.testing.assert_array_equal(actual, expected)
