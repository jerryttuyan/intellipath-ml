"""Configuration settings."""

from typing import Dict, Any


CONFIG: Dict[str, Any] = {
    "dataset": "GLA",
    "data_path": "data/raw/LargeST/gla/gla_his_2019.h5",
    "horizon": 15,  # minutes
    "target_node": None,  # set to a sensor ID string like "767494" to pin a single node
    "target_node_index": 0,  # used as the starting position when sweeping multiple nodes
    "num_target_nodes": 10,  # evaluate this many consecutive nodes when target_node is None
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "results_path": "results/baseline_comparison.csv",
    "summary_results_path": "results/baseline_summary.csv",
}
