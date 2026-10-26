# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave the company (churn).

## Project Overview

This project uses the Telco Customer Churn dataset to explore customer behavior, train classification models, and predict customer churn.

The project also provides a REST API built with FastAPI to make predictions using a saved machine learning model.

## Features

- Data cleaning and exploratory data analysis
- Feature engineering and preprocessing
- Logistic Regression, Random Forest, and Gradient Boosting models
- Model evaluation and comparison
- Hyperparameter tuning using GridSearchCV
- Saved machine learning model
- FastAPI prediction endpoint

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── raw/
├── models/
│   └── logistic_regression_churn.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
├── app.py
├── requirements.txt
└── README.md
```

## Model

The final model is a Logistic Regression pipeline that includes preprocessing and classification.

The model predicts:

- `0` — Customer is predicted not to churn
- `1` — Customer is predicted to churn

## Run the API Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn app:app --reload
```

### 5. Open the API documentation

Visit:

http://127.0.0.1:8000/docs

Use the `POST /predict` endpoint to submit customer information and receive a churn prediction.

## Learning Outcomes

This project helped me practice:

- Building an end-to-end machine learning workflow
- Data preprocessing and feature engineering
- Model evaluation and hyperparameter tuning
- Saving and loading trained models
- Building and testing a REST API with FastAPI

## Dataset

Telco Customer Churn dataset.

The dataset contains customer information, service details, account information, and churn labels.
