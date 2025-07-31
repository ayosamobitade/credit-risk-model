"""
data_cleaning.py
--------------------------------
Purpose:
    Contains functions for cleaning the credit risk dataset by:
    1. Handling missing values.
    2. Removing or adjusting outliers.

Functions:
    - handle_missing_values: Fill missing values with median/mode.
    - remove_outliers_iqr: Remove outliers using IQR method.
    - clean_data: Perform complete data cleaning pipeline.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the dataset.
      - Numeric columns: filled with median.
      - Categorical columns: filled with mode.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from numerical columns using the IQR (Interquartile Range) method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of numerical columns to check for outliers.
        factor (float): IQR multiplier for defining outlier thresholds (default = 1.5).

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    for col in columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full data cleaning pipeline:
    1. Handle missing values.
    2. Remove outliers for numeric columns.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = handle_missing_values(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = remove_outliers_iqr(df, numeric_cols)
    return df


if __name__ == "__main__":
    # Example usage (for testing this module)
    raw_data_path = "../data/raw/loan_data.csv"
    cleaned_data_path = "../data/processed/credit_data_cleaned.csv"

    print("Loading raw data...")
    data = pd.read_csv(raw_data_path)

    print("Cleaning data...")
    data_cleaned = clean_data(data)

    print(f"Saving cleaned data to {cleaned_data_path}...")
    data_cleaned.to_csv(cleaned_data_path, index=False)
    print("Data cleaning complete!")
