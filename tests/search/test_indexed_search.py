import os
from dataclasses import replace

import numpy as np
import pytest
from factor_engine.cpu_backend import CPUBackend
from factor_engine.data import IndexedBatch, Panel
from factor_engine.engine import FactorEngine, SearchConfig
from factor_engine.expression import rolling_mean, terminal
from factor_engine.ir import compile_expressions
from factor_engine.session_source import SessionSource, create_session_source

pytestmark = pytest.mark.skipif(
    not os.environ.get("FACTORENGINE_NATIVE_PLAN_LIB"), reason="own native CPU library required"
)


def fixture(tmp_path):
    rng = np.random.default_rng(621)
    x = rng.normal(size=(3, 5, 67)).astype(np.float32)
    y = rng.normal(size=x.shape).astype(np.float32)
    x[0, 1] = 1
    y[1, 3] = np.nan
    valid = np.ones(x.shape, bool)
    valid[2, 4, 9:12] = False
    panel = Panel({"x": x, "y": -x}, valid)
    create_session_source(
        tmp_path / "source",
        [IndexedBatch(0, panel, y)],
        sessions=3,
        securities=5,
        time=67,
        provenance={"engineering_only": True},
        byte_limit=20000,
    )
    return panel, y, SessionSource(tmp_path / "source", max_batch_bytes=1024**2)


def engine(source, width):
    return FactorEngine(
        plan_backend=CPUBackend(
            os.environ["FACTORENGINE_NATIVE_PLAN_LIB"],
            batch_lanes=width,
            memory_budget_bytes=256 * 1024**2,
        )
    )


def test_joint_indexed_gp_matches_materialized_full_history_and_all_generations(tmp_path):
    panel, target, source = fixture(tmp_path)
    cfg = SearchConfig(
        population_size=12, generations=3, elite_size=3, max_depth=2, max_nodes=24, seed=703
    )
    from contextlib import closing

    from factor_engine.cpu_backend import CPUBackend

    with closing(CPUBackend(os.environ["FACTORENGINE_NATIVE_PLAN_LIB"])) as cpu:
        expected = FactorEngine(plan_backend=cpu).search(panel, target, ["x", "y"], cfg)
    traces = []
    with source:
        for width in (1, 4, 7, 15):
            model = engine(source, width)
            result = model.search_indexed(source, ["x", "y"], cfg, capture_generations=True)
            assert result == expected
            stats = model.last_search_stats
            assert stats is not None
            assert stats["logical_scoring_calls"] == 4 and stats["scored_roots"] == 48
            assert len(stats["generations"]) == 4
            assert all(len(g["ordered"]) == 12 for g in stats["generations"])
            assert all(g["lanes"] == 15 for g in stats["generations"])
            assert all(
                0 <= c["finite_lanes"] <= 15 for g in stats["generations"] for c in g["ordered"]
            )
            assert stats["data_identity"] == source.source_identity
            traces.append(stats["generations"])
        assert all(trace == traces[0] for trace in traces[1:])
        assert source.stats["read_lanes"] == 4 * 4 * 15
        # 数据只准备一次；四次搜索各读 G+1 轮，不是为每个 session 单独搜索。


def test_generation_context_released_before_next_factory(tmp_path):
    import gc
    import weakref

    _, _, source = fixture(tmp_path)
    contexts = []

    class GenerationContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def generation_context(generation):
        gc.collect()
        assert all(ref() is None for ref in contexts), "previous generation still retained"
        context = GenerationContext()
        contexts.append(weakref.ref(context))
        return context

    config = SearchConfig(population_size=8, generations=2, elite_size=2, max_depth=2)
    with source:
        model = engine(source, 3)
        model.search_indexed(
            source,
            ["x", "y"],
            config,
            generation_context=generation_context,
            capture_generations=True,
        )
    gc.collect()
    assert len(contexts) == config.generations + 1
    assert all(ref() is None for ref in contexts)


def test_indexed_search_rejects_tree_and_evaluation_only_sources(tmp_path):
    _, _, source = fixture(tmp_path)
    cfg = SearchConfig(population_size=8, generations=1, elite_size=2, max_depth=2)
    with source:
        model = engine(source, 3)
        with pytest.raises(TypeError):
            replace(cfg, execution_mode="tree")
        for role in ("validation", "forward"):
            setattr(source, "role", role)
            with pytest.raises(ValueError, match=role):
                model.search_indexed(source, ["x"], cfg, capture_generations=True)
        delattr(source, "role")
        assert source.stats["read_lanes"] == 0


def test_indexed_source_ordinals_are_validated(tmp_path):
    _, _, source = fixture(tmp_path)
    with source:
        model = engine(source, 4)
        stream = model.plan_backend
        plan = compile_expressions([terminal("x"), rolling_mean(terminal("x"), 3)])

        class Reordered:
            field_names = source.field_names
            time = source.time
            lane_count = source.lane_count
            source_identity = source.source_identity
            bytes_per_lane = source.bytes_per_lane

            def iter_batches(self, first, width):
                batch = next(source.iter_batches(first, width))
                yield IndexedBatch(1, batch.panel, batch.target)

        with pytest.raises(ValueError, match="ordinal"):
            stream.score_rank_ic_source(plan, Reordered())
        assert stream.last_run_stats is None


def test_failed_search_discards_partial_stats_and_closes_source_iterator(tmp_path):
    _, _, source = fixture(tmp_path)
    with source:
        model = engine(source, 3)
        cfg = SearchConfig(population_size=8, generations=1, elite_size=2, max_depth=2)

        class Failing:
            field_names = source.field_names
            time = source.time
            lane_count = source.lane_count
            source_identity = source.source_identity
            bytes_per_lane = source.bytes_per_lane
            closed = False

            def iter_batches(self, first, width):
                try:
                    yield next(source.iter_batches(first, width))
                    raise OSError("read failure")
                finally:
                    self.closed = True

        bad = Failing()
        with pytest.raises(OSError, match="read failure"):
            model.search_indexed(bad, ["x"], cfg, capture_generations=True)
        assert bad.closed and model.last_search_stats is None
        assert model.plan_backend.backend.workspace_capacity_floats == 0
        assert model.search_indexed(source, ["x"], cfg, capture_generations=True).best.expression
