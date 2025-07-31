"""
setup.py
------------------------------------------------
Setup script for the Credit Risk Scoring Model project.
Allows installation as a Python package using `pip install .`
"""

from setuptools import setup, find_packages

setup(
    name="credit-risk-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning-based credit risk scoring model with explainability and a Streamlit web interface.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/credit-risk-model",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scikit-learn>=1.5.1",
        "xgboost>=2.1.1",
        "shap>=0.45.1",
        "lime>=0.2.0.1",
        "streamlit>=1.37.0",
        "matplotlib>=3.9.1",
        "joblib>=1.4.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "credit-risk-app=app.streamlit_app:main",
        ],
    },
)
