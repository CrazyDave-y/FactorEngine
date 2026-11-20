import numpy as np
import pytest

from factor_engine.dsl import ValueType
from factor_engine.engine import FactorEngine, SearchConfig
from factor_engine.expression import terminal
from factor_engine.gp import GPConfig, GeneticProgram


def test_gp_generation_and_genetic_operators_preserve_root_type():
    cfg = GPConfig(population_size=12, max_depth=3, max_nodes=48, seed=7)
    gp = GeneticProgram(["x", "y"], cfg)
    pop = gp.initial_population()
    assert len(pop) == 12
    assert all(c.expr.value_type is ValueType.SERIES for c in pop)
    rng = np.random.default_rng(99)
    a, b = pop[0].expr, pop[1].expr
    children = [
        gp.subtree_mutate(a, rng),
        gp.point_mutate(a, rng),
        gp.hoist_mutate(a, rng),
        gp.crossover(a, b, rng)[0],
    ]
    assert all(x.value_type is ValueType.SERIES for x in children)


def test_initial_population_exhausts_random_proposals_without_stopping(monkeypatch):
    gp = GeneticProgram(["x", "y"], GPConfig(population_size=3, max_depth=1, max_nodes=2))
    monkeypatch.setattr(gp, "generate", lambda *args: terminal("x"))
    population = gp.initial_population()
    assert len({c.key for c in population}) == 3
    assert all(gp._within_limits(c.expr) for c in population)


def test_population_space_exhaustion_is_reported():
    gp = GeneticProgram(["x", "y"], GPConfig(population_size=3, max_depth=1, max_nodes=1))
    with pytest.raises(ValueError, match="expression space exhausted"):
        gp.initial_population()


def test_search_with_unit_window_only_does_not_generate_invalid_pair_windows():
    gp = GeneticProgram(["x", "y"], GPConfig(population_size=20, max_depth=2, windows=(1,), seed=0))
    population = gp.initial_population()
    assert len(population) == 20
    assert len({candidate.key for candidate in population}) == 20


def test_pair_operators_skip_unit_window_when_other_windows_exist():
    gp = GeneticProgram(
        ["x", "y"], GPConfig(population_size=100, max_depth=2, windows=(1, 2), seed=2)
    )
    population = gp.initial_population()

    def pair_nodes(expr):
        if expr.op in {"cov", "corr"}:
            yield expr
        for child in expr.args:
            yield from pair_nodes(child)

    pairs = [node for candidate in population for node in pair_nodes(candidate.expr)]
    assert pairs and all(node.param == 2 for node in pairs)


@pytest.mark.parametrize(
    "options",
    [
        {"tournament_size": 0},
        {"terminal_probability": float("nan")},
        {"scalar_leaf_probability": 1.1},
        {"crossover_probability": -0.1, "reproduction_probability": 0.65},
    ],
)
def test_invalid_gp_probabilities_and_tournament_size_are_rejected(options):
    with pytest.raises(ValueError):
        GPConfig(**options)


def test_negative_generation_count_is_rejected():
    with pytest.raises(ValueError, match="generations"):
        SearchConfig(generations=-1)


def test_next_generation_fills_unique_population_when_proposals_collide(monkeypatch):
    config = SearchConfig(population_size=3, elite_size=1, max_depth=1, max_nodes=2)
    gp = GeneticProgram(["x", "y"], GPConfig(population_size=3, max_depth=1, max_nodes=2))
    population = gp.initial_population()
    monkeypatch.setattr(gp, "tournament", lambda ranked, rng: ranked[0][1])
    monkeypatch.setattr(gp, "crossover_one", lambda a, b, rng: a)
    monkeypatch.setattr(gp, "subtree_mutate", lambda a, rng: a)
    monkeypatch.setattr(gp, "point_mutate", lambda a, rng: a)
    monkeypatch.setattr(gp, "hoist_mutate", lambda a, rng: a)
    ranked = [(float(3 - i), c) for i, c in enumerate(population)]
    engine = FactorEngine(plan_backend=None)
    result = engine._next_population(gp, ranked, config, np.random.default_rng(1))
    assert len(result) == 3
    assert len({engine._dedup_key(c.expr, config) for c in result}) == 3
    assert result[0] is population[0]
