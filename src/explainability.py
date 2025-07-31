"""
explainability.py
------------------------------------------------
Purpose:
    Provides tools for model interpretability using SHAP and LIME.

Workflow:
    1. Load the trained model and dataset.
    2. Compute global feature importance with SHAP.
    3. Generate local explanations for individual predictions using LIME.
    4. Provide visualization functions for both methods.

Functions:
    - load_model: Load a trained model (model.pkl).
    - shap_global_explanation: Visualize SHAP global feature importance.
    - shap_local_explanation: Explain individual predictions with SHAP.
    - lime_local_explanation: Explain individual predictions with LIME.
"""

from __future__ import annotations
import pandas as pd
import joblib
from typing import Any, List

import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------
# 1. Load Model
# ---------------------------

def load_model(model_path: str) -> Any:
    """
    Load the trained model from a .pkl file.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        Any: Loaded model object.
    """
    return joblib.load(model_path)


# ---------------------------
# 2. SHAP Explainability
# ---------------------------

def shap_global_explanation(model: Any, X: pd.DataFrame, max_display: int = 10) -> None:
    """
    Generate SHAP global feature importance plot.

    Args:
        model (Any): Trained model.
        X (pd.DataFrame): Feature dataset.
        max_display (int): Max number of features to display.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, max_display=max_display, plot_type="bar")
    plt.show()


def shap_local_explanation(model: Any, X: pd.DataFrame, sample_index: int = 0) -> None:
    """
    Generate SHAP explanation for a single prediction.

    Args:
        model (Any): Trained model.
        X (pd.DataFrame): Feature dataset.
        sample_index (int): Row index to explain.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[sample_index])
    plt.show()


# ---------------------------
# 3. LIME Explainability
# ---------------------------

def lime_local_explanation(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    sample_index: int = 0,
    class_names: List[str] | None = None
) -> None:
    """
    Generate a LIME explanation for a single prediction.

    Args:
        model (Any): Trained model.
        X_train (pd.DataFrame): Training dataset (for fitting LIME).
        X_test (pd.DataFrame): Test dataset (for explaining prediction).
        feature_names (List[str]): Names of features.
        sample_index (int): Index of the test sample to explain.
        class_names (List[str] | None): Class labels.
    """
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names if class_names else ["Bad", "Good"],
        mode='classification'
    )

    sample = X_test.iloc[sample_index].values
    exp = lime_explainer.explain_instance(sample, model.predict_proba)
    exp.show_in_notebook(show_table=True)
    exp.as_pyplot_figure()
    plt.show()


# ---------------------------
# 4. Main Example Usage
# ---------------------------

if __name__ == "__main__":
    model_path = "../app/model.pkl"
    feature_data_path = "../data/processed/credit_data_features.csv"
    TARGET_COL = "loan_status"

    print("Loading model and dataset...")
    model = load_model(model_path)
    df = pd.read_csv(feature_data_path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print("Generating SHAP global feature importance...")
    shap_global_explanation(model, X, max_display=10)

    print("Explaining single prediction using SHAP...")
    shap_local_explanation(model, X, sample_index=0)

    print("Explaining single prediction using LIME...")
    lime_local_explanation(model, X, X, feature_names=X.columns.tolist(), sample_index=0)
