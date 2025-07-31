"""
test_credit_model.py
------------------------------------------------
Purpose:
    Unit tests for the Credit Risk Scoring Model pipeline.
    Tests data loading, model training, and prediction functions.

Framework:
    - pytest

Run:
    pytest tests/test_credit_model.py
"""

from __future__ import annotations
import os
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.model_training import (
    load_features,
    split_data,
    train_logistic_regression,
    train_xgboost,
    evaluate_model
)
from src.predict import predict_credit_risk

# ---------------------------
# Configuration
# ---------------------------
FEATURE_DATA_PATH = "../data/processed/credit_data_features.csv"
MODEL_PATH = "../app/model.pkl"
TARGET_COL = "loan_status"


# ---------------------------
# 1. Data Loading Tests
# ---------------------------
def test_load_features():
    """Test that feature dataset loads correctly."""
    df = load_features(FEATURE_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert TARGET_COL in df.columns


# ---------------------------
# 2. Model Training Tests
# ---------------------------
def test_train_models():
    """Test Logistic Regression and XGBoost training."""
    df = load_features(FEATURE_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, target_col=TARGET_COL)

    # Train Logistic Regression
    log_model = train_logistic_regression(X_train, y_train)
    assert isinstance(log_model, LogisticRegression)
    assert hasattr(log_model, "predict")

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    assert isinstance(xgb_model, XGBClassifier)
    assert hasattr(xgb_model, "predict")


# ---------------------------
# 3. Model Evaluation Tests
# ---------------------------
def test_evaluate_model():
    """Test evaluation metrics are returned as a dictionary."""
    df = load_features(FEATURE_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, target_col=TARGET_COL)

    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"])


# ---------------------------
# 4. Prediction Tests
# ---------------------------
def test_predict_credit_risk():
    """Test credit risk prediction for a sample applicant."""
    sample_input = {
        "annual_income": 50000,
        "total_debt": 20000,
        "current_balance": 1500,
        "total_credit_limit": 10000,
        "age": 35,
        "employment_status": 1,
        "credit_utilization": 0.15,
        "debt_to_income": 0.4
    }

    preds, probs = predict_credit_risk(
        input_data=sample_input,
        model_path=MODEL_PATH,
        feature_data_path=FEATURE_DATA_PATH,
        target_col=TARGET_COL
    )

    assert isinstance(preds, list)
    assert isinstance(probs, list)
    assert len(preds) == len(probs) == 1
    assert preds[0] in [0, 1]
