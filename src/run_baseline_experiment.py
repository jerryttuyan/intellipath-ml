"""
Simple experiment runner for IntelliPath traffic forecasting baselines.
"""

import os
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.config import CONFIG
from src.data.gla_loader import chronological_split, load_traffic_data
from src.evaluation.metrics import mae, rmse
from src.features.baseline_features import create_features
from src.models.persistence import PersistenceBaseline
from src.models.random_forest_baseline import RandomForestBaseline


def select_target_nodes(df: pd.DataFrame) -> list[str]:
    """Choose which sensor IDs to evaluate from the loaded DataFrame."""
    target_node = CONFIG.get("target_node")
    if target_node is not None:
        target_node = str(target_node)
        if target_node not in df.columns:
            raise ValueError(f"Configured target_node '{target_node}' not found in data")
        return [target_node]

    start_index = int(CONFIG.get("target_node_index", 0))
    num_target_nodes = int(CONFIG.get("num_target_nodes", 10))
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

    rf_model = RandomForestBaseline()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mae(y_test, y_pred_rf)
    rmse_rf = rmse(y_test, y_pred_rf)

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
    ]


def main():
    """Run the baseline comparison experiment."""
    # Create results directory
    os.makedirs("results", exist_ok=True)

    data_path = CONFIG.get("data_path", "data/raw/LargeST/gla/gla_his_2019.h5")
    horizon_minutes = int(CONFIG.get("horizon", 15))
    train_ratio = float(CONFIG.get("train_ratio", 0.7))
    val_ratio = float(CONFIG.get("val_ratio", 0.15))
    results_path = CONFIG.get("results_path", "results/baseline_comparison.csv")
    summary_results_path = CONFIG.get("summary_results_path", "results/baseline_summary.csv")

    # Load data
    print("Loading data...")
    df = load_traffic_data(data_path)
    horizon_steps = infer_horizon_steps(df, horizon_minutes)
    target_nodes = select_target_nodes(df)
    print(f"Evaluating {len(target_nodes)} target node(s): {target_nodes[:10]}")
    print(f"Forecast horizon: {horizon_minutes} minutes ({horizon_steps} step(s))")
    print(
        f"Global data shape: {df.shape} | "
        f"train/val/test ratios: {train_ratio:.2f}/{val_ratio:.2f}/{1 - train_ratio - val_ratio:.2f}"
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


if __name__ == "__main__":
    main()
