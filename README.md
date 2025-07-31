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
---

## 🏋️ Training the Model

    Ensure your raw dataset is in data/raw/loan_data.csv.

    Clean data:

python src/data_cleaning.py

Perform feature engineering:

python src/feature_engineering.py

Train models (Logistic Regression & XGBoost) and save the best model:

    python src/model_training.py

## 🔍 Model Explainability
To visualize SHAP and LIME explanations:
```bash
python src/explainability.py
```
---

## 🌐 Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ✅ Running Tests
```bash
pytest tests/test_credit_model.py
```
---

## 📊 Example Prediction

To test predictions using `predict.py`:
```bash
python src/predict.py
```
---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ✨ Future Improvements
- Add deep learning models for better performance.
- Integrate with a real-time credit application API.
- Expand feature engineering with alternative credit scoring features.