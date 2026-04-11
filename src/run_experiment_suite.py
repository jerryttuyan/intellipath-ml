"""Run a multi-experiment suite definition in sequence."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
else:
    repo_root = Path(__file__).resolve().parents[1]

import sys

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.run_baseline_experiment import main as run_baseline_main
from src.suites import load_suite


OVERRIDE_TO_FLAG = {
    "data_path": "--data-path",
    "horizon": "--horizon",
    "target_node": "--target-node",
    "target_node_index": "--target-node-index",
    "num_target_nodes": "--num-target-nodes",
    "train_ratio": "--train-ratio",
    "val_ratio": "--val-ratio",
    "rf_n_estimators": "--rf-n-estimators",
    "rf_random_state": "--rf-random-state",
    "rf_n_jobs": "--rf-n-jobs",
    "results_path": "--results-path",
    "summary_results_path": "--summary-results-path",
}

ROUTING_OVERRIDE_TO_FLAG = {
    "data_path": "--data-path",
    "adj_path": "--adj-path",
    "horizon": "--horizon",
    "num_nodes": "--num-nodes",
    "start_index": "--start-index",
    "num_timestamps": "--num-timestamps",
    "num_od_pairs": "--num-od-pairs",
    "seed": "--seed",
    "prediction_model": "--prediction-model",
    "results_path": "--results-path",
    "summary_path": "--summary-path",
}


def _build_cli_args(base_preset: str | None, overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if base_preset:
        args.extend(["--preset", base_preset])

    for key, value in overrides.items():
        if key not in OVERRIDE_TO_FLAG:
            raise ValueError(f"Unsupported override key in suite: {key}")
        if value is None:
            continue
        args.extend([OVERRIDE_TO_FLAG[key], str(value)])

    args.append("--show-config")
    return args


def _build_routing_cli_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        if key not in ROUTING_OVERRIDE_TO_FLAG:
            raise ValueError(f"Unsupported routing override key in suite: {key}")
        if value is None:
            continue
        args.extend([ROUTING_OVERRIDE_TO_FLAG[key], str(value)])
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predefined experiment suites.")
    parser.add_argument("--suite", default="required_suite", help="Suite name from config/suites/*.json")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit for debugging (run only first N entries).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    suite = load_suite(repo_root, args.suite)
    description = suite.get("description", "")
    base_preset = suite.get("base_preset")
    runs = suite["runs"]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    print(f"Running suite: {args.suite}")
    if description:
        print(f"Description: {description}")
    print(f"Total runs: {len(runs)}")

    suite_start = time.perf_counter()
    for idx, run in enumerate(runs, start=1):
        run_name = run.get("name", f"run_{idx}")
        run_type = run.get("type", "baseline")
        overrides = run.get("overrides", {})
        if run_type == "routing":
            run_args = _build_routing_cli_args(overrides)
            runner = "src/run_routing_experiment.py"
        else:
            run_args = _build_cli_args(base_preset, overrides)
            runner = "src/run_baseline_experiment.py"

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(runs)}] Running ({run_type}): {run_name}")
        print(f"Runner: {runner}")
        print(f"CLI args: {' '.join(run_args)}")
        run_start = time.perf_counter()
        if run_type == "routing":
            command = [sys.executable, "src/run_routing_experiment.py", *run_args]
            result = subprocess.run(command, cwd=repo_root, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Routing run '{run_name}' failed with code {result.returncode}")
        else:
            run_baseline_main(run_args)
        elapsed = time.perf_counter() - run_start
        print(f"Completed run '{run_name}' in {elapsed:.2f}s")

    suite_elapsed = time.perf_counter() - suite_start
    print("\n" + "=" * 80)
    print(f"Suite '{args.suite}' completed in {suite_elapsed:.2f}s")


if __name__ == "__main__":
    main()
