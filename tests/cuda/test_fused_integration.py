"""验证生产多目标融合入口；真实 GPU 验证只读取显式提供的 P256 数据。"""

import os
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest


def _real_input(time):
    """只取原始前 16 root 的 32 条 lane，检查目标与掩码跨 root 一致。"""
    data_path = os.environ.get("FACTORENGINE_TEST_REAL_P256")
    if os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("真实 GPU 测试须设置 FACTORENGINE_TEST_GPU=1")
    if not data_path:
        pytest.skip("真实 P256 测试须设置 FACTORENGINE_TEST_REAL_P256 为 npz 路径")
    from factor_engine.data import Panel
    from factor_engine.expression import terminal
    from factor_engine.ir import compile_expressions
    from factor_engine.session_source import ArraySource
    from factor_engine.target_bank import MultiTargetSource, TargetBatch

    with np.load(Path(data_path), allow_pickle=False) as data:
        x = data["x"]
        assert x.shape == (8192, 23400) and x.dtype == np.float32
        fields = {
            f"x{i}": np.ascontiguousarray(x[i * 32 : (i + 1) * 32, :time][None]) for i in range(16)
        }
        del x
        y = data["y"]
        assert y.shape == (8192, 23400) and y.dtype == np.float32
        for i in range(1, 16):
            np.testing.assert_array_equal(y[i * 32 : (i + 1) * 32, :time], y[:32, :time])
        target = np.ascontiguousarray(y[:32, :time][None])
        del y
        mask = data["mask"]
        assert mask.shape == (8192, 23400)
        assert np.all((mask[:512, :time] == 0) | (mask[:512, :time] == 1))
        for i in range(1, 16):
            np.testing.assert_array_equal(mask[i * 32 : (i + 1) * 32, :time], mask[:32, :time])
        eligible = np.ascontiguousarray(mask[:32, :time][None], dtype=bool)
        del mask
    panel = Panel(fields)
    source = ArraySource(panel, target, source_identity="real-p256-fused-test")
    targets = TargetBatch((1,), target, eligible)
    return MultiTargetSource(source, targets), compile_expressions(
        [terminal(name) for name in fields]
    )


@pytest.mark.parametrize("time", [3600, 23400])
def test_real_p256_fused_metrics_and_workspace(time, record_property):
    source, plan = _real_input(time)
    from factor_engine.cuda_backend import CUDABackend
    from factor_engine.runtime import ROOT

    from tests.reference.metric_checks import (
        assert_metric_parity,
        assert_stats_parity,
        stats_from_details,
    )

    library = (
        ROOT
        / os.environ.get("FACTORENGINE_TEST_CUDA_BUILD_DIR", "build-study")
        / "libfactor_engine_cuda.so"
    )
    with closing(CUDABackend(library, batch_lanes=32, memory_budget_bytes=4 * 1024**3)) as backend:
        backend.prepare_multitarget(plan, source)
        fused_resource = backend.multitarget_resources(capacity=True)["batches"][32]
        requested = 0 if time == 3600 else 16 * 32 * 23552 * 2
        assert fused_resource["requested_scratch_bytes"] == requested
        assert fused_resource["retained_scratch_bytes"] >= requested
        fused = backend.score_multitarget(plan, source, detail=True)
        assert backend.last_metric_details is not None
        fused_detail = backend.last_metric_details.copy()
        assert_stats_parity(fused, stats_from_details((1,), fused_detail), exact=True)
        from factor_engine.cpu_backend import CPUBackend

        with closing(
            CPUBackend(
                library.parent / "libfactor_engine_plan.so",
                workers=2,
                batch_lanes=32,
                memory_budget_bytes=4 * 1024**3,
            )
        ) as cpu:
            cpu.prepare_multitarget(plan, source)
            cpu_stats = cpu.score_multitarget(plan, source, detail=True)
            errors = assert_metric_parity(fused_detail, cpu.last_metric_details, fused=True)
            sum_errors = assert_stats_parity(fused, cpu_stats, fused=True)
        # 记录原始误差，同时检查 FP32 的绝对、相对联合容差。
        record_property("pearson_detail_errors", str(errors))
        record_property("pearson_sum_errors", str(sum_errors))


def test_real_p256_capacity_and_horizon_layout():
    """真实输入派生的容量/尾批检查；重复目标仅验证布局，不是新增金融标签。"""
    source, _ = _real_input(23400)
    from factor_engine.cuda_backend import CUDABackend
    from factor_engine.data import Panel
    from factor_engine.expression import terminal
    from factor_engine.ir import compile_expressions
    from factor_engine.runtime import ROOT
    from factor_engine.session_source import ArraySource
    from factor_engine.target_bank import MultiTargetSource, TargetBatch

    from tests.reference.metric_checks import (
        assert_metric_parity,
        assert_stats_parity,
        stats_from_details,
    )

    batch = next(source.iter_batches(0, 32))[0]
    raw = batch.panel.fields["x0"]
    library = (
        ROOT
        / os.environ.get("FACTORENGINE_TEST_CUDA_BUILD_DIR", "build-study")
        / "libfactor_engine_cuda.so"
    )
    for time in (3840, 3841, 23552, 23553):
        # 超出捕获长度的尾部统一填 NaN/不合资格，只检验容量边界。
        def resized(values, fill):
            result = np.full((*values.shape[:-1], time), fill, dtype=values.dtype)
            result[..., : min(time, 23400)] = values[..., : min(time, 23400)]
            return result

        x = resized(raw, np.nan)
        target = resized(source.targets.values, np.nan)
        mask = resized(source.targets.eligible, False)
        targets = TargetBatch((1, 2, 3), np.repeat(target, 3, axis=0), np.repeat(mask, 3, axis=0))
        current = MultiTargetSource(
            ArraySource(Panel({"x": x}), target, source_identity=f"real-p256-layout-{time}"),
            targets,
        )
        plan = compile_expressions([terminal("x")])
        with closing(
            CUDABackend(library, batch_lanes=31, memory_budget_bytes=4 * 1024**3)
        ) as backend:
            if time == 23553:
                with pytest.raises(RuntimeError, match="multi-target prepare failed"):
                    backend.prepare_multitarget(plan, current)
                continue
            backend.prepare_multitarget(plan, current)
            fused = backend.score_multitarget(plan, current, detail=True)
            assert backend.last_metric_details is not None
            details = backend.last_metric_details.copy()
            assert_stats_parity(fused, stats_from_details((1, 2, 3), details), exact=True)
            for width, row in backend.multitarget_resources()["batches"].items():
                requested = 0 if time <= 3840 else width * 3 * 23552 * 2
                assert row["requested_scratch_bytes"] == requested
            from factor_engine.cpu_backend import CPUBackend

            with closing(
                CPUBackend(
                    library.parent / "libfactor_engine_plan.so",
                    workers=2,
                    batch_lanes=31,
                    memory_budget_bytes=4 * 1024**3,
                )
            ) as cpu:
                cpu.prepare_multitarget(plan, current)
                expected = cpu.score_multitarget(plan, current, detail=True)
                assert_metric_parity(details, cpu.last_metric_details, fused=True)
                assert_stats_parity(fused, expected, fused=True)
