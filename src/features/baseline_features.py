"""Baseline feature extraction."""

import pandas as pd
from typing import Tuple


def create_features(df: pd.DataFrame, target_node: str, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create supervised learning features for traffic forecasting.

    Args:
        df: DataFrame with timestamp index and node columns.
        target_node: Name of the target node column to forecast.
        horizon: Number of time steps ahead to forecast (default 1).

    Returns:
        Tuple of (X, y) where X is feature DataFrame and y is target Series.

    Raises:
        ValueError: If target_node not in df columns or horizon <= 0.
    """
    if target_node not in df.columns:
        raise ValueError(f"Target node '{target_node}' not found in DataFrame columns")
    if horizon <= 0:
        raise ValueError("Horizon must be positive")

    # Create lag features
    lags = [1, 2, 3, 6]
    lag_features = {}
    for lag in lags:
        lag_features[f'lag_{lag}'] = df[target_node].shift(lag)

    # Create time-based features
    time_features = {
        'hour': df.index.hour,
        'day_of_week': df.index.dayofweek
    }

    # Combine all features
    X = pd.DataFrame({**lag_features, **time_features})

    # Create target
    y = df[target_node].shift(-horizon)

    # Drop rows with missing values (from lags and future target)
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    X = combined.drop(columns=[target_node])  # y column name is target_node
    y = combined[target_node]

    return X, y