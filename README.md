# 📌 End-to-End Customer Churn Prediction System

## Project Overview
Customer churn is a critical problem for subscription-based businesses. This project implements a **production-ready end-to-end machine learning pipeline** to predict customer churn based on historical usage and behavioral data.

## Features
- **Modular Codebase**: Clean and structured Python code for data processing, feature engineering, and model training.
- **Experiment Tracking**: Integrated with **MLflow** for logging metrics, parameters, and model versions.
- **Multiple Models**: Trains and compares Logistic Regression, Random Forest, and XGBoost.
- **Visualizations**: Automatically generates confusion matrices, ROC curves, and feature importance plots.
- **Reproducibility**: Standardized data processing and environment configuration.

## Project Structure
```text
churn_predictor/
├── data/
│   ├── raw/            # Original data (Telco Churn)
│   └── processed/      # Cleaned and split data
├── models/             # Saved model artifacts (joblib)
├── notebooks/          # Jupyter notebooks for EDA (optional)
├── reports/
│   └── figures/        # Generated plots
├── src/
│   ├── data/           # Data ingestion and cleaning scripts
│   ├── features/       # Feature engineering scripts
│   ├── models/         # Model training and prediction scripts
│   └── visualization/  # Plotting scripts
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd churn_predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
Clean and preprocess the raw data:
```bash
python src/data/make_dataset.py
python src/features/build_features.py
```

### 2. Model Training
Train multiple models and track experiments with MLflow:
```bash
python src/models/train_model.py
```

### 3. Visualization
Generate performance reports:
```bash
python src/visualization/visualize.py
```

### 4. View MLflow UI
To see the experiment results and comparison:
```bash
mlflow ui
```

## Results
The best model (XGBoost) achieved an **F1-score of ~0.77** and **ROC-AUC of ~0.90**. Key churn drivers identified include:
- Customer Service Calls
- Day Minutes/Charge
- International Plan status

