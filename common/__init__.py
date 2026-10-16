"""Shared setup for both pairs: config, data, model and inference for WILDS iWildCam.

Attributes:
    ROOT: Repository root; relative paths in the config are resolved against it.
    CONFIG_PATH: Default config file, `common/config.yaml`.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load the shared YAML config.

    Args:
        path: Config file to read. Defaults to `common/config.yaml`.

    Returns:
        The config as nested dicts, e.g. `cfg["paths"]["data_dir"]`.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root.

    Args:
        path: Absolute path, or path relative to the repository root.

    Returns:
        The absolute path; absolute inputs are returned unchanged.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
