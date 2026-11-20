"""构建工程并运行现有检查。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", action="store_true", help="build CUDA and run real-device pytest cases"
    )
    parser.add_argument("--jobs", type=int, default=2, help="bounded build parallelism")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running or writing"
    )
    parser.add_argument(
        "--skip-lint", action="store_true", help="omit Ruff/Pyright, not numerical tests"
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("jobs must be positive")

    build = ROOT / "build-study"
    selected_env = {
        "FACTORENGINE_TEST_GPU": "1" if args.gpu else "0",
        "FACTORENGINE_NATIVE_PLAN_LIB": str(build / "libfactor_engine_plan.so"),
        "OMP_NUM_THREADS": "2",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    commands = [
        [
            "cmake",
            "-S",
            ".",
            "-B",
            "build-study",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DFACTORENGINE_ENABLE_CUDA={'ON' if args.gpu else 'OFF'}",
            "-DFACTORENGINE_BUILD_TESTS=ON",
        ],
        ["cmake", "--build", "build-study", "--parallel", str(args.jobs)],
        [sys.executable, "-m", "pip", "check"],
    ]
    if not args.skip_lint:
        commands.extend(
            [
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "format",
                    "--check",
                    "python",
                    "examples",
                    "scripts",
                    "tests",
                ],
                [sys.executable, "-m", "ruff", "check", "python", "examples", "scripts", "tests"],
                [sys.executable, "-m", "pyright"],
            ]
        )
    commands.extend(
        [
            ["ctest", "--test-dir", "build-study", "--output-on-failure"],
            [sys.executable, "-m", "pytest", "--junitxml=build-study/validation/tests.xml"],
        ]
    )
    if args.dry_run:
        print(json.dumps({"commands": commands, "environment": selected_env}, indent=2))
        return 0

    env = dict(os.environ)
    env.update(selected_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    (build / "validation").mkdir(parents=True, exist_ok=True)
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
