from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "glass.csv"
DEFAULT_MATERIALS_DATA_PATH = PROJECT_ROOT / "data" / "materials_dataset.csv"

DEFAULT_DATASET_NAME = "glass"

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


@dataclass(frozen=True)
class DatasetConfig:
    """Simple configuration for each dataset option."""

    name: str
    description: str
    default_path: Path
    target_column: str
    feature_columns: list[str] | None = None
    drop_columns: tuple[str, ...] = ()
    class_labels: dict[Any, str] | None = None
    infer_numeric_features: bool = False


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "glass": DatasetConfig(
        name="glass",
        description="Starter UCI Glass Identification dataset used in the beginner-friendly workflow.",
        default_path=DEFAULT_DATA_PATH,
        target_column=TARGET_COLUMN,
        feature_columns=FEATURE_COLUMNS.copy(),
        drop_columns=(ID_COLUMN,),
        class_labels=CLASS_LABELS,
    ),
    "materials_csv": DatasetConfig(
        name="materials_csv",
        description=(
            "Path for a richer materials dataset stored as CSV with a `material_class` target column "
            "and numeric descriptor columns."
        ),
        default_path=DEFAULT_MATERIALS_DATA_PATH,
        target_column="material_class",
        drop_columns=("material_id", "formula"),
        infer_numeric_features=True,
    ),
}


def get_dataset_config(dataset_name: str = DEFAULT_DATASET_NAME) -> DatasetConfig:
    """Return the configuration for a named dataset."""
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset name: {dataset_name}. Available datasets: {available}")
    return DATASET_CONFIGS[dataset_name]


def resolve_dataset_path(
    dataset_name: str = DEFAULT_DATASET_NAME,
    csv_path: str | Path | None = None,
) -> Path:
    """Resolve the CSV path for the selected dataset."""
    config = get_dataset_config(dataset_name)
    return Path(csv_path) if csv_path is not None else config.default_path


def _resolve_dataset_request(
    dataset_name: str | Path = DEFAULT_DATASET_NAME,
    csv_path: str | Path | None = None,
) -> tuple[DatasetConfig, Path]:
    """Keep backward compatibility with older calls that passed only a CSV path."""
    if isinstance(dataset_name, Path):
        return get_dataset_config(DEFAULT_DATASET_NAME), dataset_name

    if dataset_name in DATASET_CONFIGS:
        return get_dataset_config(dataset_name), resolve_dataset_path(dataset_name, csv_path)

    if csv_path is None:
        return get_dataset_config(DEFAULT_DATASET_NAME), Path(dataset_name)

    raise ValueError(f"Unknown dataset name: {dataset_name}")


def _infer_feature_columns(dataframe: pd.DataFrame, config: DatasetConfig) -> list[str]:
    """Infer numeric feature columns for flexible materials CSV datasets."""
    excluded_columns = {config.target_column, *config.drop_columns}
    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def _get_class_names(labels: pd.Series, class_labels: dict[Any, str] | None) -> list[str]:
    """Build display names for the classes found in the dataset."""
    unique_labels = sorted(labels.unique(), key=lambda value: str(value))
    if class_labels is None:
        return [str(label) for label in unique_labels]
    return [class_labels.get(label, str(label)) for label in unique_labels]


def load_dataset(
    dataset_name: str | Path = DEFAULT_DATASET_NAME,
    csv_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load a selected dataset and return features, labels, feature names, and class names."""
    config, dataset_path = _resolve_dataset_request(dataset_name, csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {dataset_path}. "
            f"For the `{config.name}` dataset, place the CSV at this path or pass `--data`."
        )

    dataframe = pd.read_csv(dataset_path)
    if config.target_column not in dataframe.columns:
        raise ValueError(
            f"Dataset is missing the target column `{config.target_column}` for `{config.name}`."
        )

    feature_columns = config.feature_columns or _infer_feature_columns(dataframe, config)
    if not feature_columns:
        raise ValueError(
            f"No usable numeric feature columns were found for `{config.name}`. "
            f"Add numeric descriptor columns and keep the target column as `{config.target_column}`."
        )

    missing_columns = [column for column in feature_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    features = dataframe[feature_columns].copy()
    labels = dataframe[config.target_column].copy()
    class_names = _get_class_names(labels, config.class_labels)

    return features, labels, feature_columns.copy(), class_names
