"""Persistence baseline model."""

import pandas as pd
from typing import Dict, Optional, Union


class PersistenceBaseline:
    """
    A simple persistence baseline model for time series forecasting.

    The model predicts that the future value will be the same as the most recent observed value.
    """

    def __init__(self) -> None:
        """Initialize the persistence model."""
        self.last_value: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Fit the model by storing the most recent observed target value.

        Args:
            X: Feature DataFrame (ignored).
            y: Target Series or DataFrame containing history.
        """
        if isinstance(y, pd.DataFrame):
            y_series = y.iloc[:, -1]
        else:
            y_series = y

        if len(y_series) == 0:
            raise ValueError("Target history cannot be empty")

        self.last_value = float(y_series.iloc[-1])

    def predict(self, history: Union[pd.Series, pd.DataFrame]) -> float:
        """
        Predict the next value using persistence.

        Args:
            history: Historical values as a Series or DataFrame.

        Returns:
            The most recent value as the prediction.
        """
        if isinstance(history, pd.DataFrame):
            history = history.iloc[:, -1]

        if len(history) == 0:
            raise ValueError("History cannot be empty")

        return float(history.iloc[-1])


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