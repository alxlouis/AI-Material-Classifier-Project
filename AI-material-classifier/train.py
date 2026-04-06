from __future__ import annotations

import argparse
from pathlib import Path

from src.material_classifier.training import train_and_save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the beginner-friendly AI material classifier.")
    parser.add_argument("--data", default="data/glass.csv", help="Path to the training dataset CSV.")
    parser.add_argument(
        "--model-output",
        default="models/best_model.joblib",
        help="Path for the saved trained model bundle.",
    )
    parser.add_argument(
        "--metrics-output",
        default="models/metrics.json",
        help="Path for the saved metrics JSON file.",
    )
    parser.add_argument(
        "--comparison-output",
        default="models/model_comparison.csv",
        help="Path for the saved cross-validation comparison CSV.",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        default="models/confusion_matrix.png",
        help="Path for the saved confusion matrix image.",
    )
    parser.add_argument(
        "--feature-importance-output",
        default="models/feature_importance.png",
        help="Path for the saved Random Forest feature importance plot.",
    )
    args = parser.parse_args()

    summary = train_and_save_model(
        dataset_path=Path(args.data),
        model_path=Path(args.model_output),
        metrics_path=Path(args.metrics_output),
        comparison_path=Path(args.comparison_output),
        confusion_matrix_path=Path(args.confusion_matrix_output),
        feature_importance_path=Path(args.feature_importance_output),
    )

    print("Training complete.")
    print(f"Best model: {summary['best_model_name']}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Test macro F1: {summary['test_macro_f1']:.4f}")
    print(f"Saved model: {summary['model_path']}")
    print(f"Saved metrics: {summary['metrics_path']}")
    print(f"Saved comparison table: {summary['comparison_path']}")
    print(f"Saved confusion matrix: {summary['confusion_matrix_path']}")
    print(f"Saved feature importance plot: {summary['feature_importance_path']}")


if __name__ == "__main__":
    main()
