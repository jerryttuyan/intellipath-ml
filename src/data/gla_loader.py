"""Loader for GLA dataset."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd


NodeSelector = Union[int, str]


def load_traffic_data(file_path: str, node_ids: Optional[List[NodeSelector]] = None) -> pd.DataFrame:
    """
    Load traffic flow data from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file containing traffic data.
        node_ids: Optional list of node selectors to select. String values are treated
            as sensor IDs. Integer values are first treated as sensor IDs, then as
            positional column indices if no matching sensor ID exists.

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
        selected_columns: List[str] = []
        missing_nodes: List[str] = []

        for node_id in node_ids:
            node_label = str(node_id)
            if node_label in df.columns:
                selected_columns.append(node_label)
                continue

            if isinstance(node_id, int) and 0 <= node_id < len(df.columns):
                selected_columns.append(str(df.columns[node_id]))
                continue

            missing_nodes.append(node_label)

        if missing_nodes:
            raise ValueError(f"Node IDs not found in data: {missing_nodes}")
        df = df[selected_columns]

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
        df = load_traffic_data(file_path, node_ids=[0, 1, 2])  # Load first 3 nodes by position
        print(f"Loaded data shape: {df.shape}")
        print(f"Selected node labels: {df.columns.tolist()}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")

        train, val, test = chronological_split(df)
        print(f"Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")
    except Exception as e:
        print(f"Error: {e}")
