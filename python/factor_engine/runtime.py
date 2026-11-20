"""显式选择后端并管理其生命周期。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .cpu_backend import CPUBackend

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def open_backend(
    kind,
    *,
    build_dir=None,
    batch_lanes=32,
    workers=2,
    memory_budget_bytes=None,
):
    if memory_budget_bytes is not None and (
        type(memory_budget_bytes) is not int or memory_budget_bytes <= 0
    ):
        raise ValueError("memory budget must be a positive integer")
    if kind not in ("cpu", "gpu"):
        raise ValueError("backend must be explicitly cpu or gpu")
    directory = ROOT / "build-study" if build_dir is None else Path(build_dir)
    library = directory / (
        "libfactor_engine_plan.so" if kind == "cpu" else "libfactor_engine_cuda.so"
    )
    if not library.is_file():
        raise FileNotFoundError(f"Native library missing: {library}; build the current targets")
    if kind == "cpu":
        owner = CPUBackend(
            library,
            workers=workers,
            batch_lanes=batch_lanes,
            memory_budget_bytes=1024**3 if memory_budget_bytes is None else memory_budget_bytes,
        )
    else:
        from .cuda_backend import CUDABackend

        budget = memory_budget_bytes
        if budget is None and batch_lanes is not None:
            budget = 256 * 1024**2
        owner = CUDABackend(
            library,
            batch_lanes=batch_lanes,
            memory_budget_bytes=budget,
        )
    try:
        yield owner
    except BaseException as error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        owner.close()
