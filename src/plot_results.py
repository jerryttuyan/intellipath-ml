"""
Plotting script for IntelliPath baseline comparison results.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from src.data.gla_loader import load_traffic_data, chronological_split
from src.features.baseline_features import create_features
from src.models.persistence import PersistenceBaseline
from src.models.random_forest_baseline import RandomForestBaseline
from src.evaluation.metrics import mae, rmse


def main():
    """Generate plots for the baseline comparison."""
    # Create figures directory
    os.makedirs("figures", exist_ok=True)

    # Load data and prepare predictions (similar to experiment_runner)
    df = load_traffic_data("data/raw/LargeST/gla/gla_his_2019.h5", node_ids=[0])
    target_node = '0'
    train_df, val_df, test_df = chronological_split(df)

    X_train, y_train = create_features(train_df, target_node)
    X_test, y_test = create_features(test_df, target_node)

    # Persistence predictions
    y_pred_persistence = pd.Series([train_df[target_node].iloc[-1]] * len(y_test), index=y_test.index)

    # Random Forest predictions
    rf_model = RandomForestBaseline()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Calculate metrics
    mae_persistence = mae(y_test, y_pred_persistence)
    rmse_persistence = rmse(y_test, y_pred_persistence)
    mae_rf = mae(y_test, y_pred_rf)
    rmse_rf = rmse(y_test, y_pred_rf)

    # Plot 1: Actual vs Predicted over time (sample first 100 points for readability)
    plt.figure(figsize=(12, 6))
    sample_size = 100
    idx = y_test.index[:sample_size]
    plt.plot(idx, y_test[:sample_size], label='Actual', linewidth=2)
    plt.plot(idx, y_pred_persistence[:sample_size], label='Persistence', linestyle='--')
    plt.plot(idx, y_pred_rf[:sample_size], label='Random Forest', linestyle=':')
    plt.xlabel('Time')
    plt.ylabel('Traffic Speed')
    plt.title('Traffic Speed Forecasting: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/actual_vs_predicted.png', dpi=150)
    plt.close()

    # Plot 2: Bar chart comparing MAE and RMSE
    models = ['Persistence', 'Random Forest']
    mae_values = [mae_persistence, mae_rf]
    rmse_values = [rmse_persistence, rmse_rf]

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar([i - width/2 for i in x], mae_values, width, label='MAE', alpha=0.8)
    plt.bar([i + width/2 for i in x], rmse_values, width, label='RMSE', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title('Model Comparison: MAE and RMSE')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=150)
    plt.close()

    print("Plots saved to figures/ folder")


if __name__ == "__main__":
    main()