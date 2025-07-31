# 💳 Credit Risk Scoring Model

## 📌 Project Overview
This project aims to build a **Credit Risk Scoring Model** that predicts whether a loan applicant is a **Good** or **Bad** credit risk.  
The system uses **machine learning models (Logistic Regression and XGBoost)**, feature engineering (credit utilization, debt-to-income ratio), and model explainability tools (SHAP and LIME).

A **Streamlit web application** is included for real-time predictions.

---

## 🚀 Features
- **Data Cleaning:** Handle missing values and outliers.
- **Feature Engineering:** Create domain-specific financial ratios.
- **Model Training:** Logistic Regression & XGBoost with evaluation metrics.
- **Model Explainability:** SHAP and LIME for local & global interpretations.
- **Web Interface:** Streamlit-based UI for credit risk prediction.

---

## 📂 Project Structure
```
credit-risk-model/
│
├── data/
│ ├── raw/
│ │ └── loan_data.csv
│ └── processed/
│ ├── credit_data_cleaned.csv
│ └── credit_data_features.csv
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_evaluation.ipynb
│
├── src/
│ ├── init.py
│ ├── data_cleaning.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── explainability.py
│ └── predict.py
│
├── app/
│ ├── streamlit_app.py
│ ├── model.pkl
│ └── requirements.txt
│
├── tests/
│ └── test_credit_model.py
│
├── README.md
├── LICENSE
├── .gitignore
└── setup.py
```

---

## 🔧 Setup Instructions

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/credit-risk-model.git
cd credit-risk-model
```
### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r app/requirements.txt
```
