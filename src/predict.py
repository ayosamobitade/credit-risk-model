"""
predict.py
------------------------------------------------
Purpose:
    Provide a prediction interface for the Credit Risk Scoring Model.
    Loads the trained model (model.pkl) and predicts credit risk for new applicants.

Functions:
    - load_model: Load the trained model.
    - prepare_input_data: Convert raw input (dict/DataFrame) to proper format.
    - predict_credit_risk: Predict credit risk (Good/Bad) and probabilities.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Union, Any


# ---------------------------
# 1. Load Model
# ---------------------------

def load_model(model_path: str) -> Any:
    """
    Load a trained model from a pickle file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        Any: Loaded model.
    """
    return joblib.load(model_path)


# ---------------------------
# 2. Prepare Input Data
# ---------------------------

def prepare_input_data(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Prepare input data for model prediction.
    Ensures that all required features are present.

    Args:
        input_data (Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]): Raw input data.
        feature_columns (List[str]): Ordered list of features used by the model.

    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame.")

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing column with default value

    # Reorder columns to match the model
    df = df[feature_columns]
    return df


# ---------------------------
# 3. Predict Credit Risk
# ---------------------------

def predict_credit_risk(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model_path: str = "../app/model.pkl",
    feature_data_path: str = "../data/processed/credit_data_features.csv",
    target_col: str = "loan_status"
) -> Tuple[List[int], List[float]]:
    """
    Predict credit risk (Good/Bad) for new applicants.

    Args:
        input_data (Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]): New applicant data.
        model_path (str): Path to the trained model file.
        feature_data_path (str): Path to feature dataset (used to extract column order).
        target_col (str): Target column in the dataset.

    Returns:
        Tuple[List[int], List[float]]: Predicted classes (0/1) and probabilities.
    """
    # Load model
    model = load_model(model_path)

    # Load feature columns
    feature_df = pd.read_csv(feature_data_path)
    feature_columns = [col for col in feature_df.columns if col != target_col]

    # Prepare data
    processed_df = prepare_input_data(input_data, feature_columns)

    # Make predictions
    predictions = model.predict(processed_df)
    probabilities = (
        model.predict_proba(processed_df)[:, 1].tolist()
        if hasattr(model, "predict_proba") else [None] * len(predictions)
    )

    return predictions.tolist(), probabilities


# ---------------------------
# 4. Example Usage
# ---------------------------

if __name__ == "__main__":
    # Example new applicant data
    new_applicant = {
        "annual_income": 50000,
        "total_debt": 20000,
        "current_balance": 1500,
        "total_credit_limit": 10000,
        "age": 35,
        "employment_status": 1,  # Example encoded feature
        "credit_utilization": 0.15,
        "debt_to_income": 0.4
    }

    print("Loading model and making prediction...")
    preds, probs = predict_credit_risk(new_applicant)

    print(f"Predicted Class: {preds[0]} (0=Bad, 1=Good)")
    print(f"Probability of Good Credit Risk: {probs[0]:.2f}")
