"""Data preprocessing utilities for intrusion-detection inference."""

from pathlib import Path

import pandas as pd
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALER_PATH = PROJECT_ROOT / "scaler.joblib"


def load_csv_instance(filepath):
    """Load one network-flow CSV instance as a pandas DataFrame."""
    return pd.read_csv(filepath)


def scale_instance(filepath, scaler_path=SCALER_PATH):
    """Scale uploaded network-flow features using the saved training scaler."""
    instance_df = load_csv_instance(filepath)
    scaler = load(scaler_path)
    scaled_instance = scaler.transform(instance_df)
    return scaled_instance[0]


def flatten_csv_features(filepath):
    """Read numeric feature values from a CSV and return them as a flat list."""
    instance_df = pd.read_csv(filepath, header=None)
    return instance_df.astype(float).values.flatten().tolist()
