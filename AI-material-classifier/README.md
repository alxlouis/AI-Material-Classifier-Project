# Machine Learning for Material Classification

## Motivation

This project reflects my interest in the intersection of artificial intelligence and materials science.
I wanted to build a beginner-friendly machine learning system that shows how data-driven methods can be used to classify materials from measured properties.

Although this project uses a small starter dataset, the broader motivation is connected to future work in materials informatics and nanotechnology, where machine learning can support the discovery, screening, and analysis of advanced materials.

## Project Overview

This repository contains a simple classification pipeline built with `scikit-learn`.
By default, the model is trained on the UCI Glass Identification dataset, which includes composition-related numeric features such as refractive index and oxide measurements.
The project now also supports a second dataset path for a richer materials CSV, making it easier to move from a beginner dataset toward more realistic materials science workflows.
The goal is to predict the material class from these input features.

The project was designed to demonstrate a complete and clear machine learning workflow:

- load and prepare a structured dataset
- compare two classical machine learning models
- evaluate performance using standard classification metrics
- visualize confusion matrix and feature importance
- save the trained model for reuse
- run predictions on new input data

This project is intentionally kept simple and interpretable.
It is meant as a foundation for more advanced materials-focused machine learning work.
This project represents an initial step toward applying machine learning techniques in materials science, with potential future applications in nanotechnology and advanced material analysis.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook
- Streamlit

## Dataset

Starter dataset:

- Source: UCI Machine Learning Repository, Glass Identification
- Number of samples: 214
- Input features:
  - `RI`
  - `Na`
  - `Mg`
  - `Al`
  - `Si`
  - `K`
  - `Ca`
  - `Ba`
  - `Fe`
- Target: glass/material type

More realistic dataset path:

- Dataset name: `materials_csv`
- Default path: `data/materials_dataset.csv`
- Expected format:
  - one target column named `material_class`
  - numeric descriptor columns for the material features
  - optional identifier columns such as `material_id` or `formula`

This keeps the original beginner-friendly workflow intact while adding a clean path toward more realistic materials science datasets.
Instead of being limited to the Glass dataset, the project can now be pointed at a richer materials CSV with descriptors derived from composition, processing, or structure.
That makes the codebase more relevant for future materials informatics and nanotechnology-related applications.

## Model Approach

Two beginner-friendly models are trained and compared:

- Logistic Regression
- Random Forest Classifier

The final saved model is selected using cross-validation performance, with macro F1 score as the main comparison metric.

The training workflow also generates:

- classification metrics
- a model comparison graph
- a confusion matrix
- a Random Forest feature importance plot

## How To Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Train with the starter dataset explicitly:

```bash
python train.py --dataset glass
```

Train with a more realistic materials CSV:

```bash
python train.py --dataset materials_csv --data data/materials_dataset.csv
```

Run a sample prediction:

```bash
python predict.py --input data/sample_input.csv
```

Optionally save prediction results:

```bash
python predict.py --input data/sample_input.csv --output models/sample_predictions.csv
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The Streamlit app loads the saved model from `models/best_model.joblib`, so run `python train.py` first if the model file does not exist yet.
Because the saved model stores its own feature list, the CLI and Streamlit app continue to work with either dataset option after training.

## Example Output

Training:

```text
Training complete.
Dataset: glass
Best model: random_forest
Test accuracy: 0.8140
Test macro F1: 0.8264
Saved model: models\best_model.joblib
Saved metrics: models\metrics.json
Saved comparison table: models\model_comparison.csv
Saved comparison chart: models\model_comparison.png
Saved confusion matrix: models\confusion_matrix.png
Saved feature importance plot: models\feature_importance.png
```

Prediction:

```text
Predicted Material Type: building_windows_float_processed
Confidence: 81.67%
```

Streamlit UI:

- enter the nine feature values
- click `Predict`
- view the predicted material type and confidence percentage

## Results

Using the current dataset split and random seed in this repository:

- Best model: `Random Forest`
- Test accuracy: `0.8140`
- Test macro F1 score: `0.8264`

Cross-validation comparison:

- Random Forest accuracy: `0.7420`
- Random Forest macro F1: `0.6692`
- Logistic Regression accuracy: `0.6370`
- Logistic Regression macro F1: `0.4917`

These results suggest that the Random Forest model captures the structure of the dataset more effectively than Logistic Regression for this task.

## Repository Structure

```text
AI-material-classifier/
|-- data/
|-- models/
|-- notebooks/
|-- src/
|   `-- material_classifier/
|-- app.py
|-- train.py
|-- predict.py
|-- requirements.txt
`-- README.md
```

- `data/`: dataset files and sample prediction input
- `models/`: generated outputs after training
- `notebooks/`: walkthrough notebook
- `src/material_classifier/`: reusable dataset and training code

## Future Improvements

- Replace the starter dataset with a more realistic materials science dataset
- Extend the project toward nanotechnology-related material analysis tasks
- Improve the Streamlit interface with clearer result explanations
