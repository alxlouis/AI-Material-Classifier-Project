from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "glass.csv"

ID_COLUMN = "Id"
TARGET_COLUMN = "Type"
FEATURE_COLUMNS = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]

CLASS_LABELS = {
    1: "building_windows_float_processed",
    2: "building_windows_non_float_processed",
    3: "vehicle_windows_float_processed",
    4: "vehicle_windows_non_float_processed",
    5: "containers",
    6: "tableware",
    7: "headlamps",
}


def load_dataset(csv_path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load the glass dataset and return features, labels, feature names, and class names."""
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    features = dataframe[FEATURE_COLUMNS].copy()
    labels = dataframe[TARGET_COLUMN].astype(int).copy()
    class_names = [CLASS_LABELS.get(class_id, f"type_{class_id}") for class_id in sorted(labels.unique())]

    return features, labels, FEATURE_COLUMNS.copy(), class_names

