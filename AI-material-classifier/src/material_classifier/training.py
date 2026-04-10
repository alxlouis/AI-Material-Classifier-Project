from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from joblib import dump, load
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MODELS_DIR = PROJECT_ROOT / "models"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import DEFAULT_DATASET_NAME, DEFAULT_DATA_PATH, get_dataset_config, load_dataset, resolve_dataset_path

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
DEFAULT_METRICS_PATH = MODELS_DIR / "metrics.json"
DEFAULT_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
DEFAULT_COMPARISON_PLOT_PATH = MODELS_DIR / "model_comparison.png"
DEFAULT_CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
DEFAULT_FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.png"


def build_candidate_models() -> dict[str, Pipeline]:
    """Create the beginner-friendly model pipelines used in this project."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                    ),
                )
            ]
        ),
    }


def _to_serializable(value: Any) -> Any:
    """Convert numpy and sklearn output into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_candidate_models(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Run 5-fold cross-validation for each candidate model."""
    candidate_models = build_candidate_models()
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: list[dict[str, float | str]] = []

    for model_name, model in candidate_models.items():
        scores = cross_validate(
            model,
            features,
            labels,
            cv=cross_validation,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            n_jobs=None,
        )
        rows.append(
            {
                "model": model_name,
                "accuracy": float(scores["test_accuracy"].mean()),
                "macro_f1": float(scores["test_f1_macro"].mean()),
            }
        )

    comparison = pd.DataFrame(rows)
    return comparison.sort_values(
        by=["macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)


def save_model_comparison_plot(comparison: pd.DataFrame, output_path: str | Path) -> None:
    """Save a bar chart comparing model accuracy and macro F1."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_frame = comparison.copy()
    positions = range(len(plot_frame))
    bar_width = 0.35

    # Plot both metrics side by side so the models are easy to compare.
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        [position - bar_width / 2 for position in positions],
        plot_frame["accuracy"],
        width=bar_width,
        label="Accuracy",
        color="#4C72B0",
    )
    axis.bar(
        [position + bar_width / 2 for position in positions],
        plot_frame["macro_f1"],
        width=bar_width,
        label="Macro F1",
        color="#55A868",
    )

    # Format the axis labels so the chart stays beginner-friendly and readable.
    axis.set_title("Model Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(plot_frame["model"])
    axis.set_ylim(0, 1.0)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_feature_importance_plot(
    random_forest_pipeline: Pipeline,
    feature_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a simple feature importance chart for the trained Random Forest model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random_forest_model = random_forest_pipeline.named_steps["model"]
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": random_forest_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # Sorting the bars makes it easier for beginners to see the most important features first.
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(importance_frame["feature"], importance_frame["importance"], color="#4C72B0")
    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Importance")
    axis.tick_params(axis="x", rotation=45)

    # Tight layout keeps labels readable when feature names are rotated.
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def train_and_save_model(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    comparison_plot_path: str | Path = DEFAULT_COMPARISON_PLOT_PATH,
    confusion_matrix_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
    feature_importance_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
) -> dict[str, Any]:
    """Train candidate models, save the best one, and return a short summary."""
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    comparison_path = Path(comparison_path)
    comparison_plot_path = Path(comparison_plot_path)
    confusion_matrix_path = Path(confusion_matrix_path)
    feature_importance_path = Path(feature_importance_path)
    resolved_dataset_path = resolve_dataset_path(dataset_name, dataset_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_plot_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = get_dataset_config(dataset_name)
    features, labels, feature_names, _class_names = load_dataset(dataset_name=dataset_name, csv_path=resolved_dataset_path)
    class_ids = sorted(labels.unique(), key=lambda value: str(value))
    if dataset_config.class_labels is None:
        class_labels = {class_id: str(class_id) for class_id in class_ids}
    else:
        class_labels = {class_id: dataset_config.class_labels.get(class_id, str(class_id)) for class_id in class_ids}

    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    comparison = evaluate_candidate_models(features_train, labels_train)
    # Save the comparison table so the scores can be reused outside the code.
    comparison.to_csv(comparison_path, index=False)
    save_model_comparison_plot(comparison, comparison_plot_path)
    best_model_name = str(comparison.iloc[0]["model"])

    candidate_models = build_candidate_models()
    best_model = candidate_models[best_model_name]
    best_model.fit(features_train, labels_train)

    # The feature importance chart always comes from the trained Random Forest model.
    random_forest_model = best_model
    if best_model_name != "random_forest":
        random_forest_model = candidate_models["random_forest"]
        random_forest_model.fit(features_train, labels_train)
    save_feature_importance_plot(random_forest_model, feature_names, feature_importance_path)

    predictions = best_model.predict(features_test)
    accuracy = float(accuracy_score(labels_test, predictions))
    macro_f1 = float(f1_score(labels_test, predictions, average="macro"))
    report = classification_report(labels_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels_test, predictions, labels=class_ids)

    figure, axis = plt.subplots(figsize=(10, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[str(class_id) for class_id in class_ids])
    display.plot(ax=axis, colorbar=False)
    axis.set_title(f"Confusion Matrix: {best_model_name}")
    figure.tight_layout()
    figure.savefig(confusion_matrix_path, dpi=200)
    plt.close(figure)

    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": feature_names,
        "class_labels": class_labels,
        "dataset_name": dataset_name,
        "dataset_path": str(resolved_dataset_path),
        "test_metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
    }
    dump(bundle, model_path)

    metrics = {
        "best_model_name": best_model_name,
        "dataset_name": dataset_name,
        "test_metrics": {"accuracy": accuracy, "macro_f1": macro_f1},
        "classification_report": _to_serializable(report),
        "class_labels": class_labels,
        "artifacts": {
            "model_path": str(model_path),
            "comparison_path": str(comparison_path),
            "comparison_plot_path": str(comparison_plot_path),
            "confusion_matrix_path": str(confusion_matrix_path),
            "feature_importance_path": str(feature_importance_path),
        },
    }
    metrics_path.write_text(json.dumps(_to_serializable(metrics), indent=2), encoding="utf-8")

    return {
        "best_model_name": best_model_name,
        "dataset_name": dataset_name,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "comparison_path": str(comparison_path),
        "comparison_plot_path": str(comparison_plot_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "feature_importance_path": str(feature_importance_path),
    }


def load_saved_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load the serialized model bundle created during training."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model was not found at {path}. Run `python train.py` first."
        )
    return load(path)


def predict_from_dataframe(input_frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Validate input columns, run predictions, and return a beginner-friendly result table."""
    expected_columns = list(bundle["feature_names"])
    missing_columns = [column for column in expected_columns if column not in input_frame.columns]
    extra_columns = [column for column in input_frame.columns if column not in expected_columns]

    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Unexpected input columns: {extra_columns}")

    ordered_features = input_frame[expected_columns].copy()
    model = bundle["model"]
    class_labels = bundle["class_labels"]
    predicted_ids = model.predict(ordered_features)

    predictions = pd.DataFrame(
        {
            "predicted_type_id": predicted_ids,
            "predicted_type_name": [class_labels.get(class_id, str(class_id)) for class_id in predicted_ids],
        }
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(ordered_features)
        predictions["confidence"] = probabilities.max(axis=1)

    return predictions
