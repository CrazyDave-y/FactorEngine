"""用模拟数据运行原生因子搜索；run_search 也供 marketlike 入口调用。"""

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


def run_search(
    kind,
    source,
    config,
    *,
    build_dir,
    batch_lanes,
    workers,
    terminals=("return_1", "volume_change"),
    memory_budget_bytes=None,
):
    with open_backend(
        kind,
        build_dir=build_dir,
        batch_lanes=batch_lanes,
        workers=workers,
        memory_budget_bytes=memory_budget_bytes,
    ) as backend:
        engine = FactorEngine(plan_backend=backend)
        result = engine.search_indexed(source, list(terminals), config, capture_generations=True)
        stats = engine.last_search_stats
        if stats is None:
            raise RuntimeError("search completed without execution statistics")

        history = []
        for row in result.history:
            record = asdict(row)
            # 初期可能整代都无效；保留这段历史，用 JSON 的 null 表示未定义分数。
            if not math.isfinite(row.score):
                record["score"] = None
            history.append(record)
        report = {
            "winner": asdict(result.best),
            "history": history,
            "generations": stats["generations"],
            "logical_scoring_calls": stats["logical_scoring_calls"],
            "scored_roots": stats["scored_roots"],
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-study")
    parser.add_argument("--batch-lanes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch_lanes < 1:
        parser.error("batch-lanes must be positive")
    if args.workers < 1:
        parser.error("workers must be positive")

    os.environ["OMP_NUM_THREADS"] = str(args.workers)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    panel, target = make_synthetic_panel(sessions=2, securities=3, time=96, seed=7)
    source = ArraySource(panel, target, source_identity="gp-example-seed7-2x3x96")
    config = SearchConfig(
        population_size=24, generations=3, elite_size=4, max_depth=3, max_nodes=48, seed=11
    )
    reports = {
        args.backend: run_search(
            args.backend,
            source,
            config,
            build_dir=args.build_dir,
            batch_lanes=args.batch_lanes,
            workers=args.workers,
        )
    }
    receipt = {
        "status": "NATIVE_SEARCH_PASS",
        "qualification": "SYNTHETIC_ENGINEERING_NOT_OOS",
        "configuration": asdict(config),
    }
    receipt.update(reports)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)

    for kind, report in reports.items():
        winner = report["winner"]
        print(
            f"{kind}: {winner['expression']}; fitness={winner['score']:.9f}; "
            f"scoring_calls={report['logical_scoring_calls']}"
        )
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
