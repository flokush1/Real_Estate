import os
import sys
import pickle
import pandas as pd

from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging


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
