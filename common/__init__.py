"""Shared setup for both pairs: config, data, model and inference for WILDS iWildCam."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load the shared YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
