"""
model_training.py
------------------------------------------------
Purpose:
    Train machine learning models (Logistic Regression and XGBoost) for credit risk scoring.

Workflow:
    1. Load processed feature data.
    2. Split data into train/test sets.
    3. Train Logistic Regression and XGBoost models.
    4. Evaluate models using multiple metrics.
    5. Save the best performing model as model.pkl.

Functions:
    - load_features: Load processed feature dataset.
    - split_data: Split dataset into training and test sets.
    - train_logistic_regression: Train a Logistic Regression classifier.
    - train_xgboost: Train an XGBoost classifier.
    - evaluate_model: Evaluate a model on test data.
    - save_model: Save the best model to disk.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import xgboost as xgb
import joblib


# ---------------------------
# 1. Load Data
# ---------------------------

def load_features(filepath: str) -> pd.DataFrame:
    """
    Load the processed feature dataset.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath)


# ---------------------------
# 2. Split Data
# ---------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column name.
        test_size (float): Proportion of test data.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------
# 3. Train Models
# ---------------------------

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.
    """
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------
# 4. Evaluation
# ---------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a model using common classification metrics.

    Args:
        model: Trained model (LogisticRegression or XGBClassifier).
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.

    Returns:
        Dict[str, float]: Metrics including accuracy, precision, recall, F1-score, and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    }
    return metrics


# ---------------------------
# 5. Save Model
# ---------------------------

def save_model(model: Union[LogisticRegression, xgb.XGBClassifier], filepath: str) -> None:
    """
    Save trained model to disk.

    Args:
        model (Union[LogisticRegression, xgb.XGBClassifier]): Model to save.
        filepath (str): File path for saving.
    """
    joblib.dump(model, filepath)
    print(f"Model saved at {filepath}")


# ---------------------------
# 6. Main Execution
# ---------------------------

if __name__ == "__main__":
    feature_data_path = "../data/processed/credit_data_features.csv"
    model_output_path = "../app/model.pkl"
    TARGET_COL = "loan_status"  # Adjust based on dataset

    print("Loading feature dataset...")
    df = load_features(feature_data_path)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(df, target_col=TARGET_COL)

    print("Training Logistic Regression...")
    log_reg_model = train_logistic_regression(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    print("\nEvaluating models...")
    log_reg_metrics = evaluate_model(log_reg_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    print("\nLogistic Regression Metrics:", log_reg_metrics)
    print("XGBoost Metrics:", xgb_metrics)

    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, log_reg_model.predict(X_test)))

    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, xgb_model.predict(X_test)))

    # Choose best model
    best_model = xgb_model if xgb_metrics["roc_auc"] > log_reg_metrics["roc_auc"] else log_reg_model
    print(f"\nBest model: {'XGBoost' if best_model == xgb_model else 'Logistic Regression'}")

    # Save best model
    save_model(best_model, model_output_path)
