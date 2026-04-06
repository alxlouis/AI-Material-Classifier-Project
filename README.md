# Machine Learning for Material Classification

## Motivation

This project reflects my interest in the intersection of artificial intelligence and materials science.
I wanted to build a beginner-friendly machine learning system that shows how data-driven methods can be used to classify materials from measured properties.

Although this project uses a small starter dataset, the broader motivation is connected to future work in materials informatics and nanotechnology, where machine learning can support the discovery, screening, and analysis of advanced materials.

## Project Overview

This repository contains a simple classification pipeline built with `scikit-learn`.
The model is trained on the UCI Glass Identification dataset, which includes composition-related numeric features such as refractive index and oxide measurements.
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

## Dataset

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

For this version, the dataset is used as a materials-related classification example.
It is not presented as a full nanotechnology dataset, but as a starting point for future materials informatics work.

## Model Approach

Two beginner-friendly models are trained and compared:

- Logistic Regression
- Random Forest Classifier

The final saved model is selected using cross-validation performance, with macro F1 score as the main comparison metric.

The training workflow also generates:

- classification metrics
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

Run a sample prediction:

```bash
python predict.py --input data/sample_input.csv
```

Optionally save prediction results:

```bash
python predict.py --input data/sample_input.csv --output models/sample_predictions.csv
```

## Example Output

Training:

```text
Training complete.
Best model: random_forest
Test accuracy: 0.8140
Test macro F1: 0.8264
Saved model: models\best_model.joblib
Saved metrics: models\metrics.json
Saved comparison table: models\model_comparison.csv
Saved confusion matrix: models\confusion_matrix.png
Saved feature importance plot: models\feature_importance.png
```

Prediction:

```text
Predicted Material Type: building_windows_float_processed
Confidence: 81.67%
```

## Results

Using the current dataset split and random seed in this repository:

- Best model: `Random Forest`
- Test accuracy: `0.8140`
- Test macro F1 score: `0.8264`

Cross-validation comparison:

- Random Forest macro F1: `0.6692`
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
- Add a simple web interface for interactive predictions
