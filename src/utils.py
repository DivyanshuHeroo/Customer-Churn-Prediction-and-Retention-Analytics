"""
Shared utilities: configuration loading, path handling, logging, and seeding.

Keeping these helpers in one place means every module behaves consistently and
the project stays reproducible (single source of truth for the random seed).
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Project root = two levels up from this file (src/utils.py -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the central YAML configuration as a dictionary."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def resolve_path(relative_path: str | os.PathLike) -> Path:
    """Resolve a path from config relative to the project root."""
    p = Path(relative_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if it does not exist; return it."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and the hash seed for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "churn") -> logging.Logger:
    """Return a configured, non-duplicating logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
