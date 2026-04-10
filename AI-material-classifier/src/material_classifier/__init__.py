"""Beginner-friendly utilities for the AI material classifier project."""

from .dataset import (
    CLASS_LABELS,
    DATASET_CONFIGS,
    DEFAULT_DATASET_NAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_dataset_config,
    load_dataset,
)
from .training import load_saved_bundle, predict_from_dataframe, train_and_save_model

__all__ = [
    "CLASS_LABELS",
    "DATASET_CONFIGS",
    "DEFAULT_DATASET_NAME",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "get_dataset_config",
    "load_dataset",
    "load_saved_bundle",
    "predict_from_dataframe",
    "train_and_save_model",
]
