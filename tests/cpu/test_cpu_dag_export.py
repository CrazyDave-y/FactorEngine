"""调用方连续输出、尾批布局与原生计时合同。"""

import os
from contextlib import closing

import numpy as np
import pytest
from factor_engine.data import Panel
from factor_engine.expression import add, constant, delay, rolling_mean, terminal
from factor_engine.ir import compile_expressions
from factor_engine.native_plan_backend import NativePlanBackend

from tests.reference.numpy_backend import NumpyPlanBackend

pytestmark = pytest.mark.skipif(
    not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="set FACTORENGINE_NATIVE_PLAN_LIB"
)


def test_export_preserves_root_order_masks_signed_zero_and_tail_batch():
    # 连续输出必须使用当前批大小，尾批不能沿用 32 lane 的根间距。
    x = np.resize(
        np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 2, -3, 7, 1], np.float32),
        (1, 32, 19),
    )
    valid = np.ones_like(x, dtype=bool)
    valid[..., 8] = False
    plan = compile_expressions(
        [
            terminal("x"),
            constant(-0.0),
            add(terminal("x"), constant(2)),
            delay(terminal("x"), 1),
            rolling_mean(terminal("x"), 3),
        ],
        canonicalize_exprs=False,
    )
    capacity = len(plan.root_ids) * x.size
    storage = np.full(capacity + 2, 12345, np.float32)
    with closing(NativePlanBackend()) as backend:
        for lanes in (32, 16, 32):
            panel = Panel({"x": x[:, :lanes]}, valid[:, :lanes])
            size = len(plan.root_ids) * int(np.prod(panel.shape))
            storage.fill(12345)
            out = storage[1 : 1 + size].reshape((len(plan.root_ids),) + panel.shape)
            address = out.ctypes.data
            backend.evaluate_into(plan, panel, out)
            assert out.ctypes.data == address
            assert storage[0] == 12345
            assert np.all(storage[1 + size :] == 12345)
            expected = np.where(
                panel.valid & np.isfinite(panel.fields["x"]), panel.fields["x"], np.nan
            )
            np.testing.assert_array_equal(out[0], expected)
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(np.signbit(out[0][finite]), np.signbit(expected[finite]))
            assert np.all(out[1] == 0)
            assert np.signbit(out[1]).all()
            reference = NumpyPlanBackend().evaluate(plan, panel)
            original = backend.evaluate(plan, panel)
            for root, ref, old in zip(out, reference, original, strict=True):
                np.testing.assert_allclose(root, ref, rtol=3e-5, atol=3e-5, equal_nan=True)
                np.testing.assert_array_equal(root, old)
        assert backend.workspace_allocation_count == 1


@pytest.mark.parametrize("invalid", ["shape", "dtype", "readonly", "strided", "list", "unaligned"])
def test_export_rejects_invalid_output_before_workspace_allocation(invalid):
    panel = Panel({"x": np.arange(16, dtype=np.float32).reshape(1, 2, 8)})
    plan = compile_expressions([rolling_mean(terminal("x"), 2)])
    shape = (1,) + panel.shape
    out = np.empty(shape, np.float32)
    if invalid == "shape":
        out = np.empty(panel.shape, np.float32)
    elif invalid == "dtype":
        out = np.empty(shape, np.float64)
    elif invalid == "readonly":
        out.flags.writeable = False
    elif invalid == "strided":
        out = np.empty(shape[:-1] + (16,), np.float32)[..., ::2]
    elif invalid == "list":
        out = out.tolist()
    elif invalid == "unaligned":
        out = np.ndarray(shape, np.float32, buffer=bytearray(65), offset=1)
    with closing(NativePlanBackend()) as backend:
        with pytest.raises(ValueError, match="out must be"):
            backend.evaluate_into(plan, panel, out)
        assert backend.workspace_allocation_count == 0
