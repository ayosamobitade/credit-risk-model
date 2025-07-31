"""
feature_engineering.py
--------------------------------
Purpose:
    Provides functions for feature engineering on the credit risk dataset:
    1. Create domain-specific financial ratios (credit utilization, debt-to-income).
    2. Encode categorical variables.
    3. Scale numerical features for modeling.

Functions:
    - create_domain_features: Add derived financial features.
    - encode_categorical_features: Apply one-hot encoding to categorical columns.
    - scale_features: Scale numerical columns using StandardScaler.
    - feature_engineering_pipeline: Combine all steps into a single pipeline.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features for credit risk modeling:
        - Debt-to-income ratio
        - Credit utilization rate

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new domain-specific features added.
    """
    # Debt-to-income ratio
    if {'total_debt', 'annual_income'}.issubset(df.columns):
        df['debt_to_income'] = df['total_debt'] / (df['annual_income'] + 1e-5)

    # Credit utilization rate
    if {'current_balance', 'total_credit_limit'}.issubset(df.columns):
        df['credit_utilization'] = df['current_balance'] / (df['total_credit_limit'] + 1e-5)

    return df


def encode_categorical_features(
    df: pd.DataFrame, 
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Encode categorical variables using OneHotEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (List[str]): List of categorical columns to encode.

    Returns:
        Tuple[pd.DataFrame, OneHotEncoder]: Transformed DataFrame and fitted OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded ones
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df, encoder


def scale_features(
    df: pd.DataFrame, 
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (List[str]): List of numerical columns to scale.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled DataFrame and fitted StandardScaler.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def feature_engineering_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder, StandardScaler]:
    """
    Complete feature engineering pipeline:
      1. Create domain features (credit utilization, debt-to-income).
      2. Encode categorical variables.
      3. Scale numerical features.

    Args:
        df (pd.DataFrame): Cleaned input DataFrame.

    Returns:
        Tuple[pd.DataFrame, OneHotEncoder, StandardScaler]:
            - Transformed DataFrame
            - Fitted OneHotEncoder
            - Fitted StandardScaler
    """
    # Step 1: Create domain-specific features
    df = create_domain_features(df)

    # Step 2: Identify columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Step 3: Encode categorical features
    if categorical_cols:
        df, encoder = encode_categorical_features(df, categorical_cols)
    else:
        encoder = None

    # Step 4: Scale numerical features
    df, scaler = scale_features(df, numeric_cols)

    return df, encoder, scaler


if __name__ == "__main__":
    # Example usage
    cleaned_data_path = "../data/processed/credit_data_cleaned.csv"
    feature_data_path = "../data/processed/credit_data_features.csv"

    print("Loading cleaned data...")
    df_cleaned = pd.read_csv(cleaned_data_path)

    print("Performing feature engineering...")
    df_features, ohe_encoder, scaler = feature_engineering_pipeline(df_cleaned)

    print(f"Saving processed feature data to {feature_data_path}...")
    df_features.to_csv(feature_data_path, index=False)
    print("Feature engineering complete!")
