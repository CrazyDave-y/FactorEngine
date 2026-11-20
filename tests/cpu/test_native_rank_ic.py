import os

import numpy as np
import pytest

from factor_engine.data import Panel
from factor_engine.expression import corr, mul, neg, rolling_mean, terminal, constant
from factor_engine.ir import compile_expressions
from tests.reference.metrics import mean_lane_rank_ic, rank_ic
from factor_engine.native_plan_backend import NativePlanBackend
from tests.reference.numpy_backend import NumpyPlanBackend

pytestmark = pytest.mark.skipif(
    not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="native plan build required"
)


def test_exact_native_rank_stats_exclude_constant_and_short_lanes():
    # 各批有效数不同，直接平均批均值会算错；无效 lane 也不能记成零。
    x = np.array(
        [
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 2, 3, 3]],
            [[2, 2, 2, 2, 2, 2, 2, 2], [0, 1, np.nan, np.inf, np.nan, np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [[7, 6, 5, 4, 3, 2, 1, 0], [0, 0, 1, 1, 2, 2, 3, 3]],
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
        ],
        dtype=np.float32,
    )
    panel = Panel({"x": x})
    plan = compile_expressions([terminal("x"), mul(terminal("x"), constant(0))])
    backend = NativePlanBackend()
    stats = backend.score_rank_ic_stats(plan, panel, target)
    np.testing.assert_allclose(stats, [[0, 2], [0, 0]], atol=1e-15)
    np.testing.assert_allclose(
        backend.score_rank_ic(plan, panel, target), [0, np.nan], equal_nan=True
    )


def test_native_rank_scores_match_reference_with_candidate_specific_masks():
    # 因子缺失位置不同，target 的排序样本也会不同。
    rng = np.random.default_rng(44)
    x = rng.integers(-3, 4, (3, 5, 97)).astype(np.float32)
    y = rng.normal(size=x.shape).astype(np.float32)
    target = rng.integers(-2, 3, x.shape).astype(np.float32)
    x[0, 0, 3:9] = np.nan
    y[1, 2, 6] = np.inf
    target[2, 1, :20] = np.nan
    valid = np.ones(x.shape, bool)
    valid[0, 2, :3] = False
    panel = Panel({"x": x, "y": y}, valid)
    plan = compile_expressions(
        [
            terminal("x"),
            rolling_mean(terminal("x"), 5),
            corr(terminal("x"), terminal("y"), 7),
        ]
    )
    outputs = NumpyPlanBackend().evaluate(plan, panel)
    expected = [mean_lane_rank_ic(root, target) for root in outputs]
    backend = NativePlanBackend()
    for _ in range(2):
        np.testing.assert_allclose(backend.score_rank_ic(plan, panel, target), expected, atol=1e-12)
    assert backend.workspace_allocation_count == 1
    stats = backend.score_rank_ic_stats(plan, panel, target)
    counts = [
        sum(
            np.isfinite(rank_ic(a, b)) for a, b in zip(root.reshape(-1, 97), target.reshape(-1, 97))
        )
        for root in outputs
    ]
    np.testing.assert_array_equal(stats[:, 1], counts)


def test_native_rank_rejects_target_shape_mismatch():
    # target 长度不够时，应在原生读取指针前报错。
    panel = Panel({"x": np.arange(16, dtype=np.float32).reshape(1, 2, 8)})
    plan = compile_expressions([terminal("x")])
    with pytest.raises(ValueError, match="target.*shape"):
        NativePlanBackend().score_rank_ic_stats(plan, panel, np.ones((1, 1, 8)))


def test_native_target_rank_reuse_requires_identical_mask_and_resets_per_call():
    # 换了 target 后不能继续用上一次的秩缓存。
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 16)
    target = x.copy()
    target[..., :2] = np.nan
    plan = compile_expressions([terminal("x"), neg(terminal("x")), rolling_mean(terminal("x"), 5)])
    backend = NativePlanBackend()
    panel = Panel({"x": x})
    np.testing.assert_allclose(backend.score_rank_ic(plan, panel, target), [1, -1, 1], atol=1e-14)
    target = -target
    np.testing.assert_allclose(backend.score_rank_ic(plan, panel, target), [-1, 1, -1], atol=1e-14)
