from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.material_classifier.training import load_saved_bundle, predict_from_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


@st.cache_resource
def get_model_bundle() -> dict:
    """Load the saved model once so the app stays responsive."""
    return load_saved_bundle(MODEL_PATH)


@st.cache_data
def get_default_feature_values(bundle: dict) -> dict[str, float]:
    """Use dataset averages when available, and fall back to 0.0 for other datasets."""
    default_values = {feature: 0.0 for feature in bundle["feature_names"]}
    dataset_path = Path(bundle.get("dataset_path", ""))
    if not dataset_path.exists():
        return default_values

    dataframe = pd.read_csv(dataset_path)
    for feature in bundle["feature_names"]:
        if feature in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature]):
            default_values[feature] = float(dataframe[feature].mean())

    return default_values


st.set_page_config(page_title="Material Classifier")
st.title("Material Classifier")
st.write("Enter the feature values below and click Predict to estimate the material type.")

try:
    bundle = get_model_bundle()
except FileNotFoundError:
    st.error("Saved model not found. Run `python train.py` first to create `models/best_model.joblib`.")
    st.stop()

feature_names = list(bundle["feature_names"])
default_values = get_default_feature_values(bundle)
feature_values: dict[str, float] = {}

with st.form("prediction_form"):
    # Split the inputs into two columns so the form stays compact and readable.
    left_column, right_column = st.columns(2)

    for index, feature in enumerate(feature_names):
        target_column = left_column if index % 2 == 0 else right_column
        with target_column:
            feature_values[feature] = st.number_input(
                feature,
                value=default_values[feature],
                format="%.4f",
            )

    predict_button = st.form_submit_button("Predict")

if predict_button:
    # Build a one-row table so we can reuse the same prediction logic as the CLI.
    input_frame = pd.DataFrame([feature_values])
    prediction = predict_from_dataframe(input_frame, bundle).iloc[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted Material Type: {prediction['predicted_type_name']}")

    if "confidence" in prediction and pd.notna(prediction["confidence"]):
        st.write(f"Confidence: {prediction['confidence'] * 100:.2f}%")
