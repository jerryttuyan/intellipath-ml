"""Loader for GLA dataset."""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple


def load_traffic_data(file_path: str, node_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load traffic flow data from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file containing traffic data.
        node_ids: Optional list of node IDs to select. If None, loads all nodes.

    Returns:
        DataFrame with timestamp index and node IDs as columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the timestamp index is not monotonic.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load data from HDF5
    df = pd.read_hdf(file_path, key="t")

    # Convert index to datetime and sort by timestamp
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Ensure columns are string labels so downstream code can refer to node IDs consistently
    df.columns = df.columns.astype(str)

    # Validate monotonic index after sorting
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamp index is not monotonic increasing")

    # Select subset of nodes if specified
    if node_ids is not None:
        node_ids_str = [str(node_id) for node_id in node_ids]
        missing_nodes = set(node_ids_str) - set(df.columns)
        if missing_nodes:
            raise ValueError(f"Node IDs not found in data: {missing_nodes}")
        df = df[node_ids_str]

    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train, validation, and test sets.

    Args:
        df: DataFrame with timestamp index, assumed to be sorted.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1, and their sum less than 1")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    file_path = "data/raw/LargeST/gla/gla_his_2019.h5"
    try:
        df = load_traffic_data(file_path, node_ids=[0, 1, 2])  # Load first 3 nodes
        print(f"Loaded data shape: {df.shape}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")

        train, val, test = chronological_split(df)
        print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")
    except Exception as e:
        print(f"Error: {e}")