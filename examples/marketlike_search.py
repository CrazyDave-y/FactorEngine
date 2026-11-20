"""生成或扩展 12 字段模拟数据，或在已有数据上运行因子搜索。"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

from examples.gp_search import run_search
from factor_engine.market_source import (
    atomic_json,
    open_market_source,
    prepare_marketlike,
    prepare_tail,
)
from factor_engine.engine import SearchConfig
from factor_engine.marketlike import MarketLikeSource, TERMINALS
from factor_engine.runtime import ROOT
from factor_engine.session_source import SessionSource


def search_config():
    return SearchConfig(
        population_size=256,
        generations=10,
        elite_size=16,
        max_depth=4,
        max_nodes=96,
        seed=42,
        complexity_penalty=0.001,
        canonicalize_population=True,
        cse=True,
        windows=(5, 15, 30, 60, 120, 300, 900),
        lags=(1, 2, 5, 15, 30, 60),
    )


def prepare(args):
    generator = MarketLikeSource(
        seed=args.seed, sessions=args.sessions, securities=args.securities, time=args.time
    )
    if args.reuse and args.source.exists():
        with SessionSource(args.source) as source:
            if (
                source.shape != generator.shape
                or source.field_names != TERMINALS
                or dict(source.provenance) != generator.provenance
            ):
                raise ValueError("existing source configuration differs")
            identity = source.source_identity
    else:
        prepared = prepare_marketlike(args.source, generator, batch_lanes=args.batch_lanes)
        identity = prepared["source_identity"]
    print(f"source={args.source}; shape={generator.shape}; identity={identity}")


def extend(args):
    # 延长数据时沿用原数据的随机种子、标的数和每个 session 的长度。
    with SessionSource(args.head) as head:
        securities = head.securities
        time_length = head.time
        seed = head.provenance["seed"]
    report = prepare_tail(
        args.head,
        args.source,
        sessions=args.sessions,
        batch_lanes=args.batch_lanes,
        securities=securities,
        time=time_length,
        seed=seed,
    )
    print(
        f"source={args.source}; shape={report['shape']}; new_lanes_from={report['tail_global_first_lane']}"
    )


def search(args):
    config = search_config()
    population_size = config.population_size
    elite_size = config.elite_size
    generations = config.generations
    if args.population is not None:
        # 缩小种群时至少留一个位置给新生成的个体。
        population_size = args.population
        elite_size = min(config.elite_size, population_size - 1)
    if args.generations is not None:
        generations = args.generations
    config = replace(
        config, population_size=population_size, elite_size=elite_size, generations=generations
    )
    if args.backend == "both":
        kinds = ("cpu", "gpu")
    else:
        kinds = (args.backend,)
    with open_market_source(args.source) as source:
        if source.field_names != TERMINALS:
            raise ValueError("marketlike search requires all twelve input fields")
        reports = {}
        for kind in kinds:
            reports[kind] = run_search(
                kind,
                source,
                config,
                build_dir=args.build_dir,
                batch_lanes=args.batch_lanes,
                workers=args.workers,
                terminals=TERMINALS,
                memory_budget_bytes=args.memory_mib * 1024**2,
            )
        comparison = "NOT_RUN"
        if len(kinds) == 2:
            cpu_report = dict(reports["cpu"])
            gpu_report = dict(reports["gpu"])
            cpu_json = json.dumps(cpu_report, sort_keys=True, allow_nan=False)
            gpu_json = json.dumps(gpu_report, sort_keys=True, allow_nan=False)
            if cpu_json != gpu_json:
                raise RuntimeError("CPU/GPU complete search results differ")
            comparison = "BITWISE_EQUAL"
        receipt = {
            "status": "NATIVE_SEARCH_PASS",
            "qualification": "SYNTHETIC_ENGINEERING_NOT_OOS",
            "comparison": comparison,
            "source_identity": source.source_identity,
            "shape": list(source.shape),
            "fields": list(source.field_names),
            "configuration": asdict(config),
        }
        receipt.update(reports)
    output = args.output
    if output is None:
        output = ROOT / "build-study/marketlike-runs" / f"{time.time_ns()}.json"
    atomic_json(output, receipt)
    for kind, report in reports.items():
        winner = report["winner"]
        print(
            f"{kind}: {winner['expression']}; fitness={winner['score']:.9f}; scoring_calls={report['logical_scoring_calls']}"
        )
    print(f"{receipt['status']}; comparison={comparison}; output={output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="generate and persist once")
    grow = commands.add_parser("extend", help="keep the prefix and persist new sessions only")
    run = commands.add_parser(
        "search", help="read persisted data and run the current native engine"
    )
    for command in (build, grow, run):
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--batch-lanes", type=int, default=4)
    # 生成器保留小批次；搜索用更多完整 lane 摊薄每批调度和上传开销。
    run.set_defaults(batch_lanes=32)
    build.add_argument("--sessions", type=int, default=10)
    build.add_argument("--securities", type=int, default=400)
    build.add_argument("--time", type=int, default=23400, help="observations in each session")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument(
        "--reuse", action="store_true", help="verify and reuse an identical existing source"
    )
    grow.add_argument("--head", type=Path, required=True)
    grow.add_argument(
        "--sessions", type=int, required=True, help="total sessions including the head"
    )
    run.add_argument("--backend", choices=("cpu", "gpu", "both"), default="cpu")
    run.add_argument("--population", type=int)
    run.add_argument("--generations", type=int)
    run.add_argument("--build-dir", type=Path, default=ROOT / "build-study")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument(
        "--memory-mib", type=int, default=8192, help="existing native execution memory budget"
    )
    run.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_lanes <= 32:
        parser.error("batch-lanes must be 1..32")
    if args.command == "search":
        if args.population is not None and args.population < 2:
            parser.error("population must be >= 2 to leave room for offspring")
        if args.workers < 1 or args.memory_mib <= 0:
            parser.error("workers and memory-mib must be positive")
        os.environ["OMP_NUM_THREADS"] = str(args.workers)
        os.environ["OMP_DYNAMIC"] = "FALSE"
    if args.command == "prepare":
        prepare(args)
    elif args.command == "extend":
        extend(args)
    elif args.command == "search":
        search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
