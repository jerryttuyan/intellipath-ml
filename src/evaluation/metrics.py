"""Traffic prediction metrics."""

import numpy as np
import pandas as pd
from typing import Union


def mae(y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
        y_pred: Union[np.ndarray, pd.Series, pd.DataFrame]) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE value.

    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> mae(y_true, y_pred)
        0.1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Input lengths do not match: {len(y_true)} vs {len(y_pred)}")

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) == 0:
        raise ValueError("No valid values to compare after dropping NaNs")
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def rmse(y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
         y_pred: Union[np.ndarray, pd.Series, pd.DataFrame]) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RMSE value.

    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> rmse(y_true, y_pred)
        0.1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Input lengths do not match: {len(y_true)} vs {len(y_pred)}")

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) == 0:
        raise ValueError("No valid values to compare after dropping NaNs")
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def mape(y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
         y_pred: Union[np.ndarray, pd.Series, pd.DataFrame]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAPE value as percentage.

    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> mape(y_true, y_pred)
        4.44...
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Input lengths do not match: {len(y_true)} vs {len(y_pred)}")

    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | (y_true == 0))
    if np.sum(mask) == 0:
        raise ValueError("No valid non-zero true values for MAPE calculation")

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
