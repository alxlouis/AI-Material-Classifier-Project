from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.material_classifier.training import load_saved_bundle, predict_from_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions with the saved AI material classifier.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV file containing the feature columns RI, Na, Mg, Al, Si, K, Ca, Ba, Fe.",
    )
    parser.add_argument(
        "--model",
        default="models/best_model.joblib",
        help="Path to the saved trained model bundle.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the prediction results as CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    bundle = load_saved_bundle(Path(args.model))
    input_frame = pd.read_csv(input_path)
    predictions = predict_from_dataframe(input_frame, bundle)

    for index, prediction in predictions.iterrows():
        # Print each prediction as a small summary instead of a raw table.
        if len(predictions) > 1:
            print(f"Prediction {index + 1}:")

        print(f"Predicted Material Type: {prediction['predicted_type_name']}")

        confidence = prediction.get("confidence")
        if pd.notna(confidence):
            print(f"Confidence: {confidence * 100:.2f}%")
        else:
            print("Confidence: N/A")

        if index < len(predictions) - 1:
            print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
