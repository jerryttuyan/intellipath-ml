"""
Shared experiment preset helpers for CLI and Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CONFIG

PRESET_KEYS = (
    "horizon",
    "target_node",
    "target_node_index",
    "num_target_nodes",
    "train_ratio",
    "val_ratio",
    "rf_n_estimators",
    "rf_random_state",
)

PRESET_DIR = Path("config/presets")


def ensure_preset_dir(repo_root: Path) -> Path:
    preset_dir = repo_root / PRESET_DIR
    preset_dir.mkdir(parents=True, exist_ok=True)
    return preset_dir


def list_presets(repo_root: Path) -> list[str]:
    preset_dir = ensure_preset_dir(repo_root)
    return sorted(path.stem for path in preset_dir.glob("*.json"))


def load_preset(repo_root: Path, preset_name: str) -> dict[str, Any]:
    preset_dir = ensure_preset_dir(repo_root)
    preset_path = preset_dir / f"{preset_name}.json"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_name}")

    with preset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Preset '{preset_name}' must contain a JSON object.")

    invalid_keys = sorted(set(payload) - set(PRESET_KEYS))
    if invalid_keys:
        raise ValueError(
            f"Preset '{preset_name}' has unsupported keys: {', '.join(invalid_keys)}"
        )
    return payload


def save_preset(repo_root: Path, preset_name: str, config: dict[str, Any]) -> Path:
    cleaned_name = preset_name.strip()
    if not cleaned_name:
        raise ValueError("Preset name cannot be empty.")
    if any(ch in cleaned_name for ch in ("/", "\\", "..", " ")):
        raise ValueError("Preset name must not include spaces or path characters.")

    preset_dir = ensure_preset_dir(repo_root)
    preset_path = preset_dir / f"{cleaned_name}.json"
    payload = {key: config.get(key, CONFIG.get(key)) for key in PRESET_KEYS}

    with preset_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return preset_path
