"""测试使用 build-study；真实 GPU 测试需要单独开启。"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
CPU_LIBRARY = ROOT / "build-study/libfactor_engine_plan.so"
if CPU_LIBRARY.is_file():
    os.environ.setdefault("FACTORENGINE_NATIVE_PLAN_LIB", str(CPU_LIBRARY))


@pytest.fixture
def gpu_backend():
    if os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("run the complete validation task for the real GPU window")
    from factor_engine.runtime import open_backend

    build_dir = os.environ.get("FACTORENGINE_TEST_CUDA_BUILD_DIR")
    with open_backend(
        "gpu",
        build_dir=build_dir,
        batch_lanes=32,
    ) as backend:
        yield backend
