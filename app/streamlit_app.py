"""
streamlit_app.py
------------------------------------------------
Purpose:
    Provide a user-friendly web interface to interact with the Credit Risk Scoring Model.
    Users can input applicant details, and the app predicts whether the applicant is
    a Good or Bad credit risk.

Dependencies:
    - streamlit
    - pandas
    - predict.py (custom module)
"""

import streamlit as st
import pandas as pd
from predict import predict_credit_risk

# ---------------------------
# 1. App Configuration
# ---------------------------
st.set_page_config(
    page_title="Credit Risk Scoring Model",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Scoring App")
st.write("Enter applicant details below to predict credit risk.")


# ---------------------------
# 2. Input Form
# ---------------------------
with st.form("credit_form"):
    st.subheader("Applicant Information")

    annual_income = st.number_input("Annual Income (USD)", min_value=0, value=50000, step=1000)
    total_debt = st.number_input("Total Debt (USD)", min_value=0, value=20000, step=1000)
    current_balance = st.number_input("Current Balance (USD)", min_value=0, value=1500, step=100)
    total_credit_limit = st.number_input("Total Credit Limit (USD)", min_value=1, value=10000, step=500)
    age = st.number_input("Age (years)", min_value=18, value=35, step=1)
    employment_status = st.selectbox("Employment Status", options=["Employed", "Unemployed", "Self-Employed"])
    
    # Derived features
    credit_utilization = current_balance / total_credit_limit if total_credit_limit > 0 else 0
    debt_to_income = total_debt / (annual_income + 1e-5)

    submitted = st.form_submit_button("Predict Credit Risk")

# ---------------------------
# 3. Prediction Logic
# ---------------------------
if submitted:
    st.write("Processing prediction...")

    # Encode employment_status (Example encoding)
    employment_status_map = {"Employed": 1, "Unemployed": 0, "Self-Employed": 2}

    new_applicant = {
        "annual_income": annual_income,
        "total_debt": total_debt,
        "current_balance": current_balance,
        "total_credit_limit": total_credit_limit,
        "age": age,
        "employment_status": employment_status_map.get(employment_status, 0),
        "credit_utilization": credit_utilization,
        "debt_to_income": debt_to_income
    }

    # Make prediction
    preds, probs = predict_credit_risk(new_applicant)
    prediction = "Good Credit Risk" if preds[0] == 1 else "Bad Credit Risk"
    probability = probs[0] * 100 if probs[0] is not None else None

    # Display results
    st.success(f"Prediction: **{prediction}**")
    if probability is not None:
        st.info(f"Probability of Good Credit Risk: **{probability:.2f}%**")


# ---------------------------
# 4. Footer
# ---------------------------
st.markdown("---")
st.caption("Powered by Logistic Regression & XGBoost | Built with Streamlit")
