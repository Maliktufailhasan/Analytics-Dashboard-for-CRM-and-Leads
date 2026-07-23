"""
Small helpers for saving and loading trained models with joblib.

Every model module in this folder uses the same on-disk format so it's
easy to add new models later.
"""

from pathlib import Path
from typing import Any

import joblib

from config import settings


def model_path(name: str) -> Path:
    """Return the on-disk path for a model file, e.g. `lead_conversion.joblib`."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return settings.MODELS_DIR / f"{name}.joblib"


def save_bundle(name: str, bundle: dict[str, Any]) -> Path:
    """Persist a dict bundle (model + preprocessor + metadata) to disk."""
    path = model_path(name)
    joblib.dump(bundle, path)
    return path


def load_bundle(name: str) -> dict[str, Any] | None:
    """Load a bundle previously saved with `save_bundle`, or None if missing."""
    path = model_path(name)
    if not path.exists():
        return None
    return joblib.load(path)
