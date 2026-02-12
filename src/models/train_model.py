import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        # Log to MLflow
        mlflow.log_metrics(metrics)
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
            
        print(f"--- {model_name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        return model, metrics

if __name__ == "__main__":
    # Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    # Set MLflow experiment
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    # Models to train
    models = [
        (LogisticRegression(max_iter=1000), "Logistic_Regression"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random_Forest"),
        (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost")
    ]
    
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    for model, name in models:
        trained_model, metrics = train_and_evaluate(model, name, X_train, y_train, X_test, y_test)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = trained_model
            best_model_name = name
            
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"\nBest Model: {best_model_name} with F1: {best_f1:.4f}")
    with open("models/best_model_info.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\nF1 Score: {best_f1:.4f}")
