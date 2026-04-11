"""
Simple experiment runner for IntelliPath traffic forecasting baselines.
"""

import os
import pandas as pd
from src.data.gla_loader import load_traffic_data, chronological_split
from src.features.baseline_features import create_features
from src.models.persistence import PersistenceBaseline
from src.models.random_forest_baseline import RandomForestBaseline
from src.evaluation.metrics import mae, rmse


def main():
    """Run the baseline comparison experiment."""
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Load data for one node
    print("Loading data...")
    df = load_traffic_data("data/raw/LargeST/gla/gla_his_2019.h5", node_ids=[0])
    target_node = '0'

    # Split chronologically
    train_df, val_df, test_df = chronological_split(df)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    # Create features for Random Forest
    print("Creating features...")
    X_train, y_train = create_features(train_df, target_node)
    X_test, y_test = create_features(test_df, target_node)

    # Persistence baseline
    print("Evaluating persistence baseline...")
    persistence_model = PersistenceBaseline()
    persistence_model.fit(X_train, y_train)
    persisted_value = persistence_model.predict(y_train)
    y_pred_persistence = pd.Series([persisted_value] * len(y_test), index=y_test.index)
    mae_persistence = mae(y_test, y_pred_persistence)
    rmse_persistence = rmse(y_test, y_pred_persistence)

    # Random Forest baseline
    print("Training and evaluating Random Forest baseline...")
    rf_model = RandomForestBaseline()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mae(y_test, y_pred_rf)
    rmse_rf = rmse(y_test, y_pred_rf)

    # Print results
    print("\nResults:")
    print(f"Persistence - MAE: {mae_persistence:.4f}, RMSE: {rmse_persistence:.4f}")
    print(f"Random Forest - MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        'model': ['Persistence', 'Random Forest'],
        'mae': [mae_persistence, mae_rf],
        'rmse': [rmse_persistence, rmse_rf]
    })
    results_df.to_csv("results/baseline_comparison.csv", index=False)
    print("Results saved to results/baseline_comparison.csv")


if __name__ == "__main__":
    main()