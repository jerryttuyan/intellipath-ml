"""Helpers for experiment suite definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUITE_DIR = Path("config/suites")


def ensure_suite_dir(repo_root: Path) -> Path:
    suite_dir = repo_root / SUITE_DIR
    suite_dir.mkdir(parents=True, exist_ok=True)
    return suite_dir


def list_suites(repo_root: Path) -> list[str]:
    suite_dir = ensure_suite_dir(repo_root)
    return sorted(path.stem for path in suite_dir.glob("*.json"))


def load_suite(repo_root: Path, suite_name: str) -> dict[str, Any]:
    suite_dir = ensure_suite_dir(repo_root)
    suite_path = suite_dir / f"{suite_name}.json"
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite not found: {suite_name}")

    with suite_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Suite '{suite_name}' must contain a JSON object.")
    if "runs" not in payload or not isinstance(payload["runs"], list):
        raise ValueError(f"Suite '{suite_name}' must define a list field named 'runs'.")
    return payload
