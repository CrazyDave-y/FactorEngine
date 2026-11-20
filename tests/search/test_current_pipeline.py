"""检查几个入口是否走同一条原生搜索流程。"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from factor_engine.data import Panel
from factor_engine.engine import FactorEngine, SearchConfig


def test_engine_requires_an_explicit_backend():
    with pytest.raises(TypeError):
        arguments = {}
        FactorEngine(**arguments)


def test_zero_budget_is_rejected_instead_of_selecting_a_default():
    from factor_engine.runtime import open_backend

    with pytest.raises(ValueError, match="budget"):
        with open_backend("cpu", memory_budget_bytes=0):
            pass


def test_array_source_owns_inputs_and_keeps_complete_tail_lanes():
    from factor_engine.session_source import ArraySource

    original = np.arange(35, dtype=np.float32).reshape(1, 5, 7)
    source = ArraySource(Panel({"x": original}), original, source_identity="owned")
    original.fill(-99)
    batches = list(source.iter_batches(0, 3))
    assert [(batch.first_lane, batch.panel.shape) for batch in batches] == [
        (0, (1, 3, 7)),
        (3, (1, 2, 7)),
    ]
    np.testing.assert_array_equal(batches[0].target.ravel(), np.arange(21, dtype=np.float32))
    assert not batches[0].target.flags.writeable


def test_native_cpu_runs_complete_indexed_search():
    from factor_engine.runtime import open_backend
    from factor_engine.session_source import ArraySource
    from factor_engine.synthetic import make_synthetic_panel

    panel, target = make_synthetic_panel(sessions=2, securities=3, time=64, seed=7)
    source = ArraySource(panel, target, source_identity="search-regression")
    config = SearchConfig(population_size=10, generations=2, elite_size=2, max_depth=2)
    with open_backend("cpu", batch_lanes=4, workers=2) as backend:
        engine = FactorEngine(plan_backend=backend)
        result = engine.search_indexed(source, ["return_1", "volume_change"], config)
        assert result.best.expression
        assert len(result.history) == 2
        stats = engine.last_search_stats
        assert stats is not None
        assert stats["logical_scoring_calls"] == 3
        assert all(row["lanes"] == 6 for row in stats["execution"])


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_search_does_not_report_a_winner_when_every_score_is_undefined(kind):
    if kind == "gpu" and os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("real GPU window")
    from factor_engine.runtime import open_backend
    from factor_engine.session_source import ArraySource

    x = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
    source = ArraySource(Panel({"x": x}), np.full_like(x, np.nan), source_identity="no-targets")
    config = SearchConfig(population_size=2, generations=1, elite_size=1, max_depth=1)
    with open_backend(kind, batch_lanes=2) as backend:
        engine = FactorEngine(plan_backend=backend)
        with pytest.raises(ValueError, match="no finite RankIC"):
            engine.search_indexed(source, ["x"], config)
        assert engine.last_search_stats is None


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_search_report_keeps_undefined_history_when_a_later_generation_succeeds(kind):
    if kind == "gpu" and os.environ.get("FACTORENGINE_TEST_GPU") != "1":
        pytest.skip("real GPU window")
    from factor_engine.runtime import ROOT
    from factor_engine.session_source import ArraySource

    from examples.gp_search import run_search

    x = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
    source = ArraySource(Panel({"x": x}), x, source_identity="valid-offspring")
    # 初代只有超过数据长度的时间算子；子代会生成可用的字段 x。
    config = SearchConfig(
        population_size=2,
        generations=1,
        elite_size=1,
        max_depth=1,
        windows=(60,),
        lags=(60,),
        seed=6,
    )
    report = run_search(
        kind,
        source,
        config,
        build_dir=ROOT / "build-study",
        batch_lanes=2,
        workers=2,
        terminals=("x",),
    )
    assert report["history"][0]["score"] is None
    assert np.isfinite(report["winner"]["score"])
    json.dumps(report, allow_nan=False)


def test_search_entry_requires_native_library_and_has_no_reference_fallback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.gp_search",
            "--backend",
            "cpu",
            "--build-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "Native library missing" in result.stderr


@pytest.mark.skipif(os.environ.get("FACTORENGINE_TEST_GPU") != "1", reason="real GPU window")
def test_same_native_gp_runs_on_cpu_and_generic_cuda(tmp_path):
    reports = {}
    for kind in ("cpu", "gpu"):
        output = tmp_path / f"search-{kind}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.gp_search",
                "--backend",
                kind,
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        receipt = json.loads(output.read_text())
        assert receipt["status"] == "NATIVE_SEARCH_PASS"
        reports[kind] = receipt[kind]
    assert reports["cpu"] == reports["gpu"]


def _scoring_fixture():
    from factor_engine.expression import terminal
    from factor_engine.ir import compile_expressions
    from factor_engine.session_source import ArraySource

    x = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    panel = Panel({"x": x})
    plan = compile_expressions([terminal("x")])
    source = ArraySource(panel, x, source_identity="positive")
    return panel, plan, source


def test_cpu_array_scoring_enforces_budget_before_snapshot_allocation(monkeypatch):
    from factor_engine.runtime import open_backend
    from factor_engine.session_source import ArraySource

    panel, plan, source = _scoring_fixture()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("snapshot allocated before admission")

    monkeypatch.setattr(ArraySource, "__init__", unexpected_copy)
    with open_backend("cpu", memory_budget_bytes=1) as backend:
        with pytest.raises(MemoryError, match="budget"):
            backend.score_rank_ic(plan, panel, source.target)


def test_cpu_invalid_source_clears_prior_completion_evidence():
    from factor_engine.runtime import open_backend

    panel, plan, source = _scoring_fixture()
    with open_backend("cpu") as backend:
        backend.score(plan, source)
        source.time = 999
        with pytest.raises(ValueError):
            backend.score(plan, source)
        assert backend.last_run_stats is None


def test_array_source_snapshot_budget_precedes_copy(monkeypatch):
    from factor_engine.session_source import ArraySource

    panel, plan, source = _scoring_fixture()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("copy before budget admission")

    monkeypatch.setattr(np, "array", unexpected_copy)
    with pytest.raises(MemoryError, match="snapshot.*budget"):
        ArraySource(panel, source.target, source_identity="bounded", snapshot_budget_bytes=1)
