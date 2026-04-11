"""
Simple experiment runner for IntelliPath traffic forecasting baselines.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.config import CONFIG
from src.data.gla_loader import chronological_split, load_traffic_data
from src.evaluation.metrics import mae, rmse
from src.features.baseline_features import create_features
from src.models.linear_regression_baseline import LinearRegressionBaseline
from src.models.persistence import PersistenceBaseline
from src.models.random_forest_baseline import RandomForestBaseline
from src.presets import list_presets, load_preset, save_preset


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for optional experiment configuration overrides."""
    parser = argparse.ArgumentParser(
        description="Run traffic forecasting baselines with optional config overrides."
    )
    parser.add_argument("--data-path", type=str, help="Path to the traffic history file.")
    parser.add_argument(
        "--preset",
        type=str,
        help="Load shared experiment settings from config/presets/<name>.json.",
    )
    parser.add_argument(
        "--save-preset",
        type=str,
        help="Save current non-device settings to config/presets/<name>.json and exit.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available shared presets and exit.",
    )
    parser.add_argument("--horizon", type=int, help="Forecast horizon in minutes.")
    parser.add_argument(
        "--target-node",
        type=str,
        help="Single sensor ID to evaluate (overrides range-based node selection).",
    )
    parser.add_argument(
        "--target-node-index",
        type=int,
        help="Start index in the sensor column list when sweeping multiple nodes.",
    )
    parser.add_argument(
        "--num-target-nodes",
        type=int,
        help="Number of consecutive sensor IDs to evaluate when --target-node is not set.",
    )
    parser.add_argument("--train-ratio", type=float, help="Training split ratio (0 to 1).")
    parser.add_argument("--val-ratio", type=float, help="Validation split ratio (0 to 1).")
    parser.add_argument("--rf-n-estimators", type=int, help="Random Forest tree count.")
    parser.add_argument("--rf-random-state", type=int, help="Random Forest random seed.")
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        help="Random Forest CPU workers (1 single core, -1 all cores, >1 capped cores).",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        help="Output path for per-node model metrics CSV.",
    )
    parser.add_argument(
        "--summary-results-path",
        type=str,
        help="Output path for aggregated summary CSV.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved configuration values before running.",
    )
    parser.add_argument(
        "--no-save-history",
        action="store_true",
        help="Skip writing timestamped run artifacts and run_history.csv entry.",
    )
    return parser


def resolve_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge optional CLI overrides into the default repository config."""
    merged: dict[str, Any] = dict(CONFIG)
    if args.preset:
        merged.update(load_preset(repo_root, args.preset))

    override_map = {
        "data_path": args.data_path,
        "horizon": args.horizon,
        "target_node": args.target_node,
        "target_node_index": args.target_node_index,
        "num_target_nodes": args.num_target_nodes,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_random_state": args.rf_random_state,
        "rf_n_jobs": args.rf_n_jobs,
        "results_path": args.results_path,
        "summary_results_path": args.summary_results_path,
    }
    for key, value in override_map.items():
        if value is not None:
            merged[key] = value
    return merged


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate core runtime parameters before loading data."""
    if int(config["horizon"]) <= 0:
        raise ValueError("horizon must be positive.")
    if int(config["target_node_index"]) < 0:
        raise ValueError("target_node_index must be non-negative.")
    if int(config["num_target_nodes"]) <= 0:
        raise ValueError("num_target_nodes must be positive.")
    if float(config["rf_n_estimators"]) <= 0:
        raise ValueError("rf_n_estimators must be positive.")
    if int(config["rf_n_jobs"]) == 0:
        raise ValueError("rf_n_jobs cannot be 0. Use -1, 1, or a positive integer.")

    train_ratio = float(config["train_ratio"])
    val_ratio = float(config["val_ratio"])
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")


def save_run_artifacts(
    *,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    experiment_config: dict[str, Any],
    selected_preset: str | None,
    elapsed_seconds: float,
) -> tuple[Path, Path, Path]:
    """Persist timestamped run outputs and append a row to run_history.csv."""
    results_root = repo_root / "results"
    runs_dir = results_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_comparison_path = runs_dir / f"{run_id}_comparison.csv"
    run_summary_path = runs_dir / f"{run_id}_summary.csv"
    history_path = results_root / "run_history.csv"

    results_df.to_csv(run_comparison_path, index=False)
    summary_df.to_csv(run_summary_path, index=False)

    model_rows = {
        row["model"]: row
        for _, row in summary_df.iterrows()
    }
    rf_row = model_rows.get("Random Forest", {})
    linear_row = model_rows.get("Linear Regression", {})
    persistence_row = model_rows.get("Persistence", {})

    history_row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "preset": selected_preset or "",
        "horizon": int(experiment_config["horizon"]),
        "num_target_nodes": int(experiment_config["num_target_nodes"]),
        "target_node_index": int(experiment_config["target_node_index"]),
        "target_node": "" if experiment_config.get("target_node") is None else str(experiment_config["target_node"]),
        "train_ratio": float(experiment_config["train_ratio"]),
        "val_ratio": float(experiment_config["val_ratio"]),
        "rf_n_estimators": int(experiment_config["rf_n_estimators"]),
        "rf_random_state": int(experiment_config["rf_random_state"]),
        "rf_n_jobs": int(experiment_config["rf_n_jobs"]),
        "rf_mean_mae": rf_row.get("mean_mae"),
        "rf_mean_rmse": rf_row.get("mean_rmse"),
        "linear_mean_mae": linear_row.get("mean_mae"),
        "linear_mean_rmse": linear_row.get("mean_rmse"),
        "persistence_mean_mae": persistence_row.get("mean_mae"),
        "persistence_mean_rmse": persistence_row.get("mean_rmse"),
        "elapsed_seconds": elapsed_seconds,
        "comparison_csv": str(run_comparison_path.relative_to(repo_root)),
        "summary_csv": str(run_summary_path.relative_to(repo_root)),
    }

    history_row_df = pd.DataFrame([history_row])
    if history_path.exists():
        existing_df = pd.read_csv(history_path)
        combined_df = pd.concat([existing_df, history_row_df], ignore_index=True)
        combined_df.to_csv(history_path, index=False)
    else:
        history_row_df.to_csv(history_path, index=False)

    return run_comparison_path, run_summary_path, history_path


def select_target_nodes(df: pd.DataFrame, experiment_config: dict[str, Any]) -> list[str]:
    """Choose which sensor IDs to evaluate from the loaded DataFrame."""
    target_node = experiment_config.get("target_node")
    if target_node is not None:
        target_node = str(target_node)
        if target_node in df.columns:
            return [target_node]
        # Friendly fallback: if a numeric target_node is not a sensor ID, treat it as start index.
        if target_node.isdigit():
            fallback_index = int(target_node)
            experiment_config["target_node"] = None
            experiment_config["target_node_index"] = fallback_index
            print(
                "Note: target_node was numeric but not a sensor ID; "
                f"using target_node_index={fallback_index} instead."
            )
        else:
            raise ValueError(
                f"Configured target_node '{target_node}' not found in data. "
                "Use a real sensor ID (e.g., 767494) or leave target_node blank "
                "and set target_node_index/num_target_nodes."
            )

    start_index = int(experiment_config.get("target_node_index", 0))
    num_target_nodes = int(experiment_config.get("num_target_nodes", 10))
    if start_index < 0:
        raise ValueError("target_node_index must be non-negative")
    if num_target_nodes <= 0:
        raise ValueError("num_target_nodes must be positive")

    selected = list(df.columns[start_index:start_index + num_target_nodes])
    if not selected:
        raise ValueError("No target nodes selected from the configured index range")
    return selected


def infer_horizon_steps(df: pd.DataFrame, horizon_minutes: int) -> int:
    """Convert a horizon in minutes into whole time steps for the given DataFrame."""
    time_deltas = df.index.to_series().diff().dropna()
    if time_deltas.empty:
        raise ValueError("Cannot infer sampling interval from an empty or 1-row DataFrame")

    sampling_interval = time_deltas.mode().iloc[0]
    sampling_minutes = int(sampling_interval.total_seconds() // 60)
    if sampling_minutes <= 0:
        raise ValueError(f"Invalid sampling interval: {sampling_interval}")
    if horizon_minutes % sampling_minutes != 0:
        raise ValueError(
            f"Horizon {horizon_minutes} minutes is not divisible by sampling interval {sampling_minutes} minutes"
        )
    return horizon_minutes // sampling_minutes


def evaluate_target_node(
    df: pd.DataFrame,
    target_node: str,
    horizon_steps: int,
    horizon_minutes: int,
    train_ratio: float,
    val_ratio: float,
    rf_n_estimators: int,
    rf_random_state: int,
    rf_n_jobs: int | None,
) -> list[dict[str, float | str | int]]:
    """Train and evaluate the baselines for one target node."""
    node_df = df[[target_node]]
    train_df, val_df, test_df = chronological_split(node_df, train_ratio=train_ratio, val_ratio=val_ratio)

    X_train, y_train = create_features(train_df, target_node, horizon=horizon_steps)
    X_test, y_test = create_features(test_df, target_node, horizon=horizon_steps)

    persistence_model = PersistenceBaseline()
    persistence_model.fit(X_train, y_train)
    y_pred_persistence = X_test["current_value"].astype(float).rename(y_test.name)
    mae_persistence = mae(y_test, y_pred_persistence)
    rmse_persistence = rmse(y_test, y_pred_persistence)

    rf_model = RandomForestBaseline(
        n_estimators=rf_n_estimators,
        random_state=rf_random_state,
        n_jobs=rf_n_jobs,
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mae(y_test, y_pred_rf)
    rmse_rf = rmse(y_test, y_pred_rf)

    linear_model = LinearRegressionBaseline()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mae_linear = mae(y_test, y_pred_linear)
    rmse_linear = rmse(y_test, y_pred_linear)

    return [
        {
            "model": "Persistence",
            "target_node": target_node,
            "horizon_minutes": horizon_minutes,
            "mae": mae_persistence,
            "rmse": rmse_persistence,
        },
        {
            "model": "Random Forest",
            "target_node": target_node,
            "horizon_minutes": horizon_minutes,
            "mae": mae_rf,
            "rmse": rmse_rf,
        },
        {
            "model": "Linear Regression",
            "target_node": target_node,
            "horizon_minutes": horizon_minutes,
            "mae": mae_linear,
            "rmse": rmse_linear,
        },
    ]


def main(argv: list[str] | None = None):
    """Run the baseline comparison experiment."""
    args = build_arg_parser().parse_args(argv)
    if args.list_presets:
        presets = list_presets(repo_root)
        if presets:
            print("Available presets:")
            for preset in presets:
                print(f"  - {preset}")
        else:
            print("No presets found in config/presets.")
        return

    experiment_config = resolve_experiment_config(args)

    if args.save_preset:
        path = save_preset(repo_root, args.save_preset, experiment_config)
        print(f"Saved preset: {path.relative_to(repo_root)}")
        return

    run_start = time.perf_counter()
    validate_experiment_config(experiment_config)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    data_path = experiment_config.get("data_path", "data/raw/LargeST/gla/gla_his_2019.h5")
    horizon_minutes = int(experiment_config.get("horizon", 15))
    train_ratio = float(experiment_config.get("train_ratio", 0.7))
    val_ratio = float(experiment_config.get("val_ratio", 0.15))
    rf_n_estimators = int(experiment_config.get("rf_n_estimators", 100))
    rf_random_state = int(experiment_config.get("rf_random_state", 42))
    raw_rf_n_jobs = experiment_config.get("rf_n_jobs", 1)
    rf_n_jobs = None if raw_rf_n_jobs is None else int(raw_rf_n_jobs)
    results_path = experiment_config.get("results_path", "results/baseline_comparison.csv")
    summary_results_path = experiment_config.get("summary_results_path", "results/baseline_summary.csv")

    if args.show_config:
        print("Resolved configuration:")
        for key in sorted(experiment_config):
            print(f"  {key}: {experiment_config[key]}")

    # Load data
    print("Loading data...")
    df = load_traffic_data(data_path)
    horizon_steps = infer_horizon_steps(df, horizon_minutes)
    target_nodes = select_target_nodes(df, experiment_config)
    print(f"Evaluating {len(target_nodes)} target node(s): {target_nodes[:10]}")
    print(f"Forecast horizon: {horizon_minutes} minutes ({horizon_steps} step(s))")
    print(
        f"Global data shape: {df.shape} | "
        f"train/val/test ratios: {train_ratio:.2f}/{val_ratio:.2f}/{1 - train_ratio - val_ratio:.2f}"
    )
    print(
        f"Random Forest params: n_estimators={rf_n_estimators}, "
        f"random_state={rf_random_state}, n_jobs={rf_n_jobs}"
    )

    all_results: list[dict[str, float | str | int]] = []
    for idx, target_node in enumerate(target_nodes, start=1):
        print(f"\n[{idx}/{len(target_nodes)}] Evaluating target node {target_node}...")
        node_results = evaluate_target_node(
            df,
            target_node=target_node,
            horizon_steps=horizon_steps,
            horizon_minutes=horizon_minutes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            rf_n_estimators=rf_n_estimators,
            rf_random_state=rf_random_state,
            rf_n_jobs=rf_n_jobs,
        )
        all_results.extend(node_results)
        for result in node_results:
            print(
                f"  {result['model']}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    summary_df = (
        results_df.groupby("model", as_index=False)
        .agg(
            num_nodes=("target_node", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            median_mae=("mae", "median"),
            median_rmse=("rmse", "median"),
        )
        .sort_values("mean_mae")
    )

    print("\nAverage results across evaluated nodes:")
    print(summary_df.to_string(index=False))

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_results_path, index=False)
    print(f"Results saved to {results_path}")
    print(f"Summary saved to {summary_results_path}")

    elapsed_seconds = time.perf_counter() - run_start
    print(f"Elapsed time: {elapsed_seconds:.2f}s")
    if not args.no_save_history:
        run_comparison_path, run_summary_path, history_path = save_run_artifacts(
            results_df=results_df,
            summary_df=summary_df,
            experiment_config=experiment_config,
            selected_preset=args.preset,
            elapsed_seconds=elapsed_seconds,
        )
        print(f"Run comparison snapshot saved to {run_comparison_path.relative_to(repo_root)}")
        print(f"Run summary snapshot saved to {run_summary_path.relative_to(repo_root)}")
        print(f"Run history updated at {history_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
