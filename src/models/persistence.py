"""Persistence baseline model."""

import pandas as pd
from typing import Dict, Union


class PersistenceBaseline:
    """
    A simple persistence baseline model for time series forecasting.

    The model predicts that the future value will be the same as the most recent observed value.
    """

    def __init__(self):
        """Initialize the persistence model."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model. For persistence, this does nothing.

        Args:
            X: Feature DataFrame (ignored).
            y: Target Series (ignored).
        """
        pass  # No fitting required for persistence

    def predict(self, history: pd.Series) -> float:
        """
        Predict the next value using persistence.

        Args:
            history: Series of historical values.

        Returns:
            The most recent value as the prediction.
        """
        if len(history) == 0:
            raise ValueError("History cannot be empty")
        return history.iloc[-1]


def evaluate_persistence(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Evaluate persistence predictions against true values.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE and RMSE metrics.
    """
    mae = (y_true - y_pred).abs().mean()
    rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
    return {"mae": mae, "rmse": rmse}