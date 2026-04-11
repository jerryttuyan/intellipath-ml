"""Repeatable routing experiment using current vs predicted traffic weights."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
else:
    repo_root = Path(__file__).resolve().parents[1]

import sys

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import CONFIG
from src.data.gla_loader import chronological_split, load_traffic_data
from src.features.baseline_features import create_features
from src.models.linear_regression_baseline import LinearRegressionBaseline
from src.models.random_forest_baseline import RandomForestBaseline
from src.routing.a_star import a_star_routing
from src.routing.graph_builder import build_graph
from src.run_baseline_experiment import infer_horizon_steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run routing comparison experiments.")
    parser.add_argument("--data-path", default=CONFIG["data_path"])
    parser.add_argument("--adj-path", default="data/raw/LargeST/gla/gla_rn_adj.npy")
    parser.add_argument("--horizon", type=int, default=int(CONFIG["horizon"]))
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-timestamps", type=int, default=5)
    parser.add_argument("--num-od-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-model", choices=["random_forest", "linear_regression"], default="random_forest")
    parser.add_argument("--results-path", default="results/routing_comparison.csv")
    parser.add_argument("--summary-path", default="results/routing_summary.csv")
    parser.add_argument("--no-save-history", action="store_true")
    return parser.parse_args(argv)


def _clamp_speed(value: float, min_speed: float = 1.0) -> float:
    return max(float(value), min_speed)


def _path_realized_cost(graph: Any, path: list[int] | None) -> float | None:
    if not path or len(path) < 2:
        return None
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = graph.get_edge_data(u, v)
        if edge is None or "travel_time_min" not in edge:
            return None
        total += float(edge["travel_time_min"])
    return total


def _append_history_row(history_path: Path, row: dict[str, Any]) -> None:
    row_df = pd.DataFrame([row])
    if history_path.exists():
        existing = pd.read_csv(history_path)
        pd.concat([existing, row_df], ignore_index=True).to_csv(history_path, index=False)
    else:
        row_df.to_csv(history_path, index=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_nodes <= 1:
        raise ValueError("num-nodes must be > 1")
    if args.num_timestamps <= 0 or args.num_od_pairs <= 0:
        raise ValueError("num-timestamps and num-od-pairs must be positive")

    run_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    df = load_traffic_data(args.data_path)
    horizon_steps = infer_horizon_steps(df, int(args.horizon))

    end_index = args.start_index + args.num_nodes
    selected_columns = list(df.columns[args.start_index:end_index])
    if len(selected_columns) < 2:
        raise ValueError("Selected node slice is too small for routing.")

    df_subset = df[selected_columns]
    adj_matrix = np.load(args.adj_path)
    adj_subset = adj_matrix[args.start_index:end_index, args.start_index:end_index]
    if adj_subset.shape[0] != len(selected_columns):
        raise ValueError("Adjacency subset shape does not match selected node count.")

    model_by_node: dict[str, Any] = {}
    x_test_by_node: dict[str, pd.DataFrame] = {}
    y_test_by_node: dict[str, pd.Series] = {}

    for node in selected_columns:
        node_df = df_subset[[node]]
        train_df, _, test_df = chronological_split(
            node_df,
            train_ratio=float(CONFIG["train_ratio"]),
            val_ratio=float(CONFIG["val_ratio"]),
        )
        x_train, y_train = create_features(train_df, node, horizon=horizon_steps)
        x_test, y_test = create_features(test_df, node, horizon=horizon_steps)
        if x_train.empty or x_test.empty:
            continue

        if args.prediction_model == "linear_regression":
            model = LinearRegressionBaseline()
        else:
            model = RandomForestBaseline(
                n_estimators=int(CONFIG["rf_n_estimators"]),
                random_state=int(CONFIG["rf_random_state"]),
                n_jobs=int(CONFIG["rf_n_jobs"]),
            )
        model.fit(x_train, y_train)
        model_by_node[node] = model
        x_test_by_node[node] = x_test
        y_test_by_node[node] = y_test

    if len(model_by_node) < 2:
        raise ValueError("Not enough node models were trained for routing evaluation.")
    if len(model_by_node) != len(selected_columns):
        missing = sorted(set(selected_columns) - set(model_by_node))
        raise ValueError(
            "Routing evaluation expects trained models for all selected nodes; "
            f"failed to train {len(missing)} node(s): {missing[:10]}"
        )

    common_times = None
    for node in model_by_node:
        node_times = set(x_test_by_node[node].index)
        common_times = node_times if common_times is None else common_times.intersection(node_times)
    if not common_times:
        raise ValueError("No common timestamps available across trained nodes.")

    sorted_times = sorted(common_times)
    sample_count = min(args.num_timestamps, len(sorted_times))
    chosen_indices = rng.choice(len(sorted_times), size=sample_count, replace=False)
    chosen_times = [sorted_times[idx] for idx in sorted(chosen_indices)]

    rows: list[dict[str, Any]] = []
    for ts in chosen_times:
        current_speeds: dict[int, float] = {}
        predicted_speeds: dict[int, float] = {}
        future_speeds: dict[int, float] = {}

        trained_nodes = list(model_by_node.keys())
        for local_idx, node in enumerate(trained_nodes):
            current_value = df_subset.loc[ts, node]
            predicted_value = model_by_node[node].predict(x_test_by_node[node].loc[[ts]]).iloc[0]
            future_value = y_test_by_node[node].loc[ts]
            current_speeds[local_idx] = _clamp_speed(float(current_value))
            predicted_speeds[local_idx] = _clamp_speed(float(predicted_value))
            future_speeds[local_idx] = _clamp_speed(float(future_value))

        graph_distance = build_graph(adj_subset, default_speed=50.0)
        graph_current = build_graph(adj_subset, current_speeds=current_speeds, default_speed=50.0)
        graph_predicted = build_graph(adj_subset, predicted_speeds=predicted_speeds, default_speed=50.0)
        graph_eval = build_graph(adj_subset, current_speeds=future_speeds, default_speed=50.0)

        node_count = len(trained_nodes)
        od_generated = 0
        attempts = 0
        max_attempts = args.num_od_pairs * 20
        while od_generated < args.num_od_pairs and attempts < max_attempts:
            attempts += 1
            start = int(rng.integers(0, node_count))
            goal = int(rng.integers(0, node_count))
            if start == goal:
                continue

            path_dist, planned_dist = a_star_routing(graph_distance, start, goal, weight="distance")
            path_curr, planned_curr = a_star_routing(graph_current, start, goal, weight="travel_time_min")
            path_pred, planned_pred = a_star_routing(graph_predicted, start, goal, weight="travel_time_min")
            if not path_dist or not path_curr or not path_pred:
                continue

            realized_dist = _path_realized_cost(graph_eval, path_dist)
            realized_curr = _path_realized_cost(graph_eval, path_curr)
            realized_pred = _path_realized_cost(graph_eval, path_pred)
            if realized_dist is None or realized_curr is None or realized_pred is None:
                continue

            od_key = f"{start}->{goal}"
            rows.extend(
                [
                    {
                        "timestamp": ts,
                        "od_pair": od_key,
                        "method": "Shortest Distance",
                        "planned_cost": planned_dist,
                        "realized_cost": realized_dist,
                    },
                    {
                        "timestamp": ts,
                        "od_pair": od_key,
                        "method": "Current Traffic",
                        "planned_cost": planned_curr,
                        "realized_cost": realized_curr,
                    },
                    {
                        "timestamp": ts,
                        "od_pair": od_key,
                        "method": "Predicted Traffic",
                        "planned_cost": planned_pred,
                        "realized_cost": realized_pred,
                    },
                ]
            )
            od_generated += 1

    if not rows:
        raise ValueError("No valid OD route comparisons were generated.")

    results_df = pd.DataFrame(rows)
    best_realized = (
        results_df.groupby(["timestamp", "od_pair"], as_index=False)["realized_cost"]
        .min()
        .rename(columns={"realized_cost": "best_realized_cost"})
    )
    results_df = results_df.merge(best_realized, on=["timestamp", "od_pair"], how="left")
    results_df["regret"] = results_df["realized_cost"] - results_df["best_realized_cost"]

    summary_df = (
        results_df.groupby("method", as_index=False)
        .agg(
            num_routes=("od_pair", "count"),
            mean_realized_cost=("realized_cost", "mean"),
            median_realized_cost=("realized_cost", "median"),
            p90_realized_cost=("realized_cost", lambda s: float(np.percentile(s, 90))),
            mean_regret=("regret", "mean"),
        )
        .sort_values("mean_realized_cost")
    )

    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_path, index=False)
    summary_df.to_csv(args.summary_path, index=False)
    print(f"Routing results saved to {args.results_path}")
    print(f"Routing summary saved to {args.summary_path}")
    print(summary_df.to_string(index=False))

    elapsed = time.perf_counter() - run_start
    print(f"Elapsed time: {elapsed:.2f}s")

    if args.no_save_history:
        return

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = repo_root / "results" / "routing_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_results_path = runs_dir / f"{run_id}_comparison.csv"
    run_summary_path = runs_dir / f"{run_id}_summary.csv"
    results_df.to_csv(run_results_path, index=False)
    summary_df.to_csv(run_summary_path, index=False)

    history_row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "prediction_model": args.prediction_model,
        "horizon_minutes": args.horizon,
        "start_index": args.start_index,
        "num_nodes": args.num_nodes,
        "num_timestamps": args.num_timestamps,
        "num_od_pairs": args.num_od_pairs,
        "seed": args.seed,
        "best_method": summary_df.iloc[0]["method"],
        "best_mean_realized_cost": float(summary_df.iloc[0]["mean_realized_cost"]),
        "elapsed_seconds": elapsed,
        "comparison_csv": str(run_results_path.relative_to(repo_root)),
        "summary_csv": str(run_summary_path.relative_to(repo_root)),
    }
    history_path = repo_root / "results" / "routing_history.csv"
    _append_history_row(history_path, history_row)
    print(f"Routing run snapshot saved to {run_results_path.relative_to(repo_root)}")
    print(f"Routing summary snapshot saved to {run_summary_path.relative_to(repo_root)}")
    print(f"Routing history updated at {history_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
