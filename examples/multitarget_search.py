"""原生 CPU/CUDA 多期限 GP 示例；公开指标为 PearsonIC 和 RankIC。

使用合成数据验证执行流程，默认以期限加权 RankIC 作为 fitness。

PYTHONPATH=python .venv/bin/python examples/multitarget_search.py --output /tmp/multi-gp.json
显式调节 --population-size、--generations、--horizons、--lanes、--time 和批宽。
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

from factor_engine.engine import FactorEngine, SearchConfig
from factor_engine.runtime import ROOT, open_backend
from factor_engine.session_source import ArraySource
from factor_engine.synthetic import make_synthetic_panel
from factor_engine.target_bank import MultiTargetSource, targets_from_close


H9 = (2, 5, 8, 15, 30, 60, 120, 300, 900)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-study")
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--horizons", type=int, nargs="+", default=H9)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--time", type=int, default=1025)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-lanes", type=int, default=32)
    parser.add_argument("--memory-budget-mib", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.workers)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    horizons = tuple(args.horizons)
    if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        parser.error("horizons must be unique positive integers")
    if args.time < max(horizons) + 3:
        parser.error("time must exceed the largest horizon by at least three observations")
    if args.lanes < 1 or not 1 <= args.batch_lanes <= 32 or args.memory_budget_mib < 1:
        parser.error("positive lanes/budget and batch-lanes in 1..32 required")
    if not 0 < args.elite_size <= args.population_size or args.generations < 0:
        parser.error("require 0 < elite-size <= population-size and generations >= 0")

    config = SearchConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        max_depth=3,
        max_nodes=32,
        seed=args.seed,
    )
    panel, _ = make_synthetic_panel(
        sessions=1, securities=args.lanes, time=args.time, seed=args.seed
    )
    targets = targets_from_close(panel.fields["close"], horizons, panel.valid)
    base = ArraySource(
        panel,
        targets.values[0][None],
        source_identity=f"synthetic-multi-gp-{args.seed}-{args.lanes}-{args.time}",
    )
    source = MultiTargetSource(base, targets)

    with open_backend(
        args.backend,
        workers=args.workers,
        build_dir=args.build_dir,
        batch_lanes=args.batch_lanes,
        memory_budget_bytes=args.memory_budget_mib * 1024**2,
    ) as backend:
        engine = FactorEngine(plan_backend=backend)
        result = engine.search_multitarget(
            source,
            ["return_1", "volume_change"],
            config,
        )
        history = [asdict(row) for row in result.history]
        for row in history:
            if not math.isfinite(row["score"]):
                row["score"] = None
        report = {
            "status": "MULTITARGET_GP_PASS",
            "qualification": "SYNTHETIC_ENGINEERING_NOT_OOS",
            "configuration": asdict(config),
            "winner": asdict(result.best),
            "history": history,
            "search_stats": engine.last_search_stats,
        }

    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"{report['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
