import json
import os
import pickle
import shutil
import sys
from datetime import datetime

import pandas as pd

from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging

MODEL_REGISTRY_PATH = os.path.join("artifact", "model_registry.json")


def get_next_model_version(model_type: str = "plot") -> int:
    """Return the next version number for a model type (auto-increments from registry)."""
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return 1
    with open(MODEL_REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    latest = registry.get(model_type, {}).get("latest", 0)
    return latest + 1


def register_model_version(
    model_type: str,
    version: int,
    model_path: str,
    metrics: dict,
) -> None:
    """Record a trained model version with timestamp and metrics in the registry."""
    os.makedirs(os.path.dirname(MODEL_REGISTRY_PATH), exist_ok=True)
    registry = {}
    if os.path.exists(MODEL_REGISTRY_PATH):
        with open(MODEL_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    if model_type not in registry:
        registry[model_type] = {"latest": 0, "versions": {}}
    registry[model_type]["latest"] = version
    registry[model_type]["versions"][str(version)] = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_path": model_path,
        "metrics": metrics,
    }
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    logging.info(f"Model registry updated: {model_type} v{version} → {model_path}")


def save_object(file_path: str, obj: object) -> None:
    """Pickle-save any Python object to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved → {file_path}")
    except Exception as e:
        raise RealEstateException(e, sys)


def load_object(file_path: str) -> object:
    """Load a pickled object from disk."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RealEstateException(e, sys)


def load_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV into a DataFrame with logging."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded CSV: {file_path}  shape={df.shape}")
        return df
    except Exception as e:
        raise RealEstateException(e, sys)
