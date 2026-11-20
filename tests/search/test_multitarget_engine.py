"""多期限沿用同一个 GP 种群和 DAG，不按期限重复构建引擎。"""

import json
from dataclasses import replace

import numpy as np
import pytest
from factor_engine.data import Panel
from factor_engine.engine import FactorEngine, SearchConfig
from factor_engine.metric_types import METRIC_COUNT, Metric
from factor_engine.session_source import ArraySource
from factor_engine.target_bank import MultiTargetSource, targets_from_close

H9 = (2, 5, 8, 15, 30, 60, 120, 300, 900)


def search_source(horizons=H9, lanes=3, time=1025):
    rng = np.random.default_rng(91)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, (1, lanes, time)), axis=-1))
    panel = Panel({"close": close, "volume": rng.uniform(1, 3, close.shape)})
    targets = targets_from_close(panel.fields["close"], horizons, panel.valid)
    source = ArraySource(panel, targets.values[0][None], source_identity="multi-gp-test")
    return MultiTargetSource(source, targets)


def small_search_config(**changes):
    config = SearchConfig(
        population_size=8,
        generations=2,
        elite_size=2,
        max_depth=3,
        max_nodes=32,
        seed=11,
        complexity_penalty=0.01,
    )
    return replace(config, **changes)


def assert_search_diagnostics(result, stats, config, source):
    assert stats["logical_scoring_calls"] == config.generations + 1
    assert stats["scored_roots"] == config.population_size * (config.generations + 1)
    generations = stats["generations"]
    assert [g["generation"] for g in generations] == list(range(config.generations + 1))
    assert stats["horizons"] == source.targets.horizons
    assert stats["algorithm"] == "fused"
    for generation in generations:
        ordered = generation["ordered"]
        assert len(ordered) == config.population_size
        finite = [c["score"] for c in ordered if c["score"] is not None]
        assert finite == sorted(finite, reverse=True)
        raw = np.asarray(generation["metric_stats"])
        assert raw.shape == (config.population_size, len(source.targets.horizons), METRIC_COUNT, 2)
        for row in ordered:
            assert row["key"]
            assert 0 <= row["root_index"] < config.population_size
        assert len(generation["top_indices"]) <= config.population_size
    assert len(result.history) == config.generations
    for best, generation in zip(result.history, generations):
        assert best.key == generation["ordered"][0]["key"]
        assert (best.score if np.isfinite(best.score) else None) == generation["ordered"][0][
            "score"
        ]
    best = max(
        (g["ordered"][0] for g in generations),
        key=lambda row: row["score"] if row["score"] is not None else -np.inf,
    )
    assert result.best.key == best["key"] and result.best.score == best["score"]
    json.dumps(stats, allow_nan=False)


def assert_real_fitness_and_elites(stats, config):
    for generation in stats["generations"]:
        raw = np.asarray(generation["metric_stats"])
        for row in generation["ordered"]:
            totals = raw[row["root_index"], :, Metric.RANK_IC]
            if np.all(totals[:, 1] > 0):
                expected = (
                    np.mean(totals[:, 0] / totals[:, 1])
                    - config.complexity_penalty * row["complexity"]
                )
                assert row["score"] == pytest.approx(expected, rel=1e-14, abs=1e-14)
            else:
                assert row["score"] is None
    for previous, current in zip(stats["generations"], stats["generations"][1:]):
        assert {row["key"] for row in previous["ordered"][: config.elite_size]} <= {
            row["key"] for row in current["ordered"]
        }


def test_real_gpu_multitarget_p8_g2_h9_evolves_and_repeats(gpu_backend):
    """实际 CUDA 验收，需要显式设置 FACTORENGINE_TEST_GPU=1。"""
    source = search_source()
    engine = FactorEngine(plan_backend=gpu_backend)
    config = small_search_config()
    result = engine.search_multitarget(source, ["close", "volume"], config)
    stats = engine.last_search_stats
    assert stats is not None
    assert_search_diagnostics(result, stats, config, source)
    assert_real_fitness_and_elites(stats, config)
    operations = {row["operation"] for g in stats["generations"][1:] for row in g["ordered"]}
    assert "crossover" in operations
    assert operations.intersection({"subtree_mutation", "point_mutation", "hoist_mutation"})
    changed = [
        row
        for g in stats["generations"][1:]
        for row in g["ordered"]
        if row["parents"] and row["key"] not in row["parents"]
    ]
    assert any(row["operation"] == "crossover" for row in changed)
    assert any(row["operation"].endswith("mutation") for row in changed)
    repeated = engine.search_multitarget(source, ["close", "volume"], config)
    assert result == repeated
    assert semantic(stats) == semantic(engine.last_search_stats)
    assert gpu_backend.last_metric_details is None  # 只取紧凑汇总，不导出完整因子。


def test_real_gpu_multitarget_p16_g1_h3_full_and_tail_batches(gpu_backend):
    source = search_source((5, 30, 300), lanes=48)
    config = small_search_config(population_size=16, generations=1)
    engine = FactorEngine(plan_backend=gpu_backend)
    result = engine.search_multitarget(source, ["close", "volume"], config)
    stats = engine.last_search_stats
    assert stats is not None
    assert_search_diagnostics(result, stats, config, source)
    assert_real_fitness_and_elites(stats, config)
    resources = gpu_backend.multitarget_resources()
    assert resources["batch_index"] == "first_lane"
    assert set(resources["batches"]) == {0, 32}
    assert [r["capacity_lanes"] for r in resources["batches"].values()] == [32, 16]
    assert resources["algorithm"] == "fused"
    assert resources["owned_device_bytes"] <= gpu_backend.memory_budget_bytes


def test_real_gpu_multitarget_h1_matches_single_target_search(gpu_backend):
    source = search_source((30,))
    config = small_search_config(generations=1)
    engine = FactorEngine(plan_backend=gpu_backend)
    multi = engine.search_multitarget(source, ["close", "volume"], config)
    multi_stats = engine.last_search_stats
    assert multi_stats is not None
    single = engine.search_indexed(
        source.source, ["close", "volume"], config, capture_generations=True
    )
    assert multi == single
    single_stats = engine.last_search_stats
    assert single_stats is not None
    for multi_generation, single_generation in zip(
        multi_stats["generations"], single_stats["generations"], strict=True
    ):
        raw = np.asarray(multi_generation["metric_stats"])
        for multi_row, single_row in zip(
            multi_generation["ordered"], single_generation["ordered"], strict=True
        ):
            assert multi_row["key"] == single_row["key"]
            assert multi_row["score"] == single_row["score"]
            rank_sum, count = raw[multi_row["root_index"], 0, Metric.RANK_IC]
            assert rank_sum == single_row["rank_ic_sum"]
            assert count == single_row["finite_lanes"]
    gpu_backend.close()
    assert gpu_backend.closed and not gpu_backend._metric_buffers
    with pytest.raises(RuntimeError, match="closed"):
        engine.search_multitarget(source, ["close"], config)
    assert engine.last_search_stats is None


def test_example_rejects_unmatured_longest_horizon_before_opening_backend(capsys):
    from examples.multitarget_search import main

    with pytest.raises(SystemExit) as error:
        main(["--time", "902"])
    assert error.value.code == 2
    assert "largest horizon" in capsys.readouterr().err


def semantic(stats):
    return [
        {key: value for key, value in generation.items() if key != "cache"}
        for generation in stats["generations"]
    ]
