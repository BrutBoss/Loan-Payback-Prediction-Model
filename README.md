# 📘 Loan Payback Prediction Model

*A machine learning model for predicting loan repayment probability.*

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.0-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-blue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-black?logo=numpy)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)
![Render](https://img.shields.io/badge/Render-Cloud%20Deployment-46E3B7?logo=render)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![ML](https://img.shields.io/badge/model-XGBoost-orange)

## 🚀 Overview

This project builds a machine-learning pipeline that predicts a
customer's likelihood of successfully repaying a loan. It includes:

-   EDA & feature analysis\
-   Preprocessing pipeline\
-   Hyperparameter tuning\
-   Model evaluation\
-   Interactive data/report dashboard\
-   Flask API for predictions\
-   Docker container\
-   Cloud deployment on Render

## 🧩 1. Problem Description

Financial institutions face risk when issuing loans.\
**Goal:** Predict whether a loan will be repaid using historical
customer and loan data.

### Use Cases

-   Reduce default risk\
-   Support automated loan approvals\
-   Improve credit scoring models\
-   Prioritize low-risk loan applicants

## 📊 2. Dataset Overview

The dataset includes financial indicators, loan details, demographic
attributes, and repayment outcomes.

**Target Variable**

  Variable        Meaning
  --------------- ---------------------------
  `loan_status`   1 = repaid, 0 = defaulted

**Data Source**

The dataset for this project was sourced from the Kaggle Playground Series - Season 5, Episode 11 competition.

-   Kaggle Competition: https://www.kaggle.com/competitions/playground-series-s5e11/data

-   Project Repository (Datasets Folder): https://github.com/BrutBoss/Loan-Payback-Prediction-Model/tree/main/Datasets


## 📈 3. Exploratory Data Analysis (EDA)

### Key Findings

  --------------------------------------------------------------------------
  Finding            Details                  Implication
  ------------------ ------------------------ ------------------------------
  **Class            \~80% repaid vs \~20%    ROC AUC selected as primary
  imbalance**        default                  metric

  **Missing values** `annual_income`,         Managed with imputation
                     `debt_to_income_ratio`   
                     had minor missingness    

  **Correlations**   `credit_score` ↑         Critical predictors
                     repayment;               
                     `debt_to_income_ratio` ↓ 
                     repayment                

  **Categorical      `grade_subgrade`,        One-hot encoded
  variables**        `loan_purpose` show      
                     class separation         
  --------------------------------------------------------------------------

## 🛠️ 4. Preprocessing Pipeline

A **scikit-learn Pipeline** with **ColumnTransformer** ensures
reproducible preprocessing.

  ------------------------------------------------------------------------
  Feature Type                 Processing               Purpose
  ---------------------------- ------------------------ ------------------
  Categorical                  One-Hot Encoding         Converts
                                                        categories into
                                                        ML-ready vectors

  Numerical                    Imputation +             Handles missing
                               StandardScaler           values &
                                                        normalizes scales
  ------------------------------------------------------------------------

Final full pipeline is saved as **`xgb_tuned_model.pkl`**.

## 🤖 5. Model Training & Tuning

Models were evaluated using **ROC AUC**.

### Model Comparison

  ---------------------------------------------------------------------------
  Model        Baseline AUC       Best Tuned AUC        Tuning Method
  ------------ ------------------ --------------------- ---------------------
  Logistic     0.911              0.912                 GridSearchCV
  Regression                                            

  Random       0.880              0.907                 Iterative tuning
  Forest                                                

  **XGBoost    **0.922**          **0.926**             RandomizedSearchCV
  (Final                                                
  Model)**                                              
  ---------------------------------------------------------------------------

### Final XGBoost Hyperparameters

  Parameter            Value
  -------------------- -------
  `n_estimators`       411
  `max_depth`          3
  `learning_rate`      0.204
  `subsample`          0.969
  `colsample_bytree`   0.835
  `min_child_weight`   5
  `gamma`              0.036

## 📐 6. Model Evaluation

  Metric               Score       Meaning
  -------------------- ----------- ------------------------
  **ROC AUC**          **0.926**   Excellent predictor
  Accuracy             85.2%       Good despite imbalance
  Precision (Repaid)   \~0.89      Fewer false approvals
  Recall (Default)     \~0.70      Captures most defaults
  F1 Score             0.76        Balanced performance

## 📊 7. Interactive Report

The project includes a browser-based dashboard:

**`/workspaces/Loan-Payback-Prediction-Model/loan_report.html`**

Features: - Interactive EDA\
- Model comparison charts\
- Loan-simulator widget

## 📁 8. Project Structure

    .
    ├── Datasets/
    ├── train.py
    ├── predict.py
    ├── app.py
    ├── xgb_tuned_model.pkl
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    ├── loan_report_spa.html        <-- Interactive Report
    └── Loan_Payback_Prediction.ipynb

## 🔁 9. Training & Reproducibility

### Train the model

``` bash
python train.py
```

### Predict from CLI

``` bash
python predict.py '{"loan_amount": 15000.0, ... }'
```

## 🌐 10. Deployment (Flask API)

Start API locally:

``` bash
python app.py
```

### Example Request

``` json
{
  "loan_amount": 15000.0,
  "credit_score": 720,
  "employment_duration": 5,
  "annual_income": 75000.0,
  "interest_rate": 10.5,
  "debt_to_income_ratio": 0.25,
  "has_house": "Y",
  "gender": "male",
  "marital_status": "single",
  "education_level": "bachelors",
  "employment_status": "employed",
  "loan_purpose": "auto",
  "grade_subgrade": "B3"
}
```

## 🐳 11. Docker Deployment

Build image:

``` bash
docker build -t loan-payback-service .
```

Run container:

``` bash
docker run -dp 9696:9696 loan-payback-service
```

## ☁️ 12. Cloud Deployment (Render)

Example URL:

    https://loan-payback-prediction-model.onrender.com/

## 📦 13. Dependencies

Install:

``` bash
pip install -r requirements.txt
```

Includes:\
- Flask\
- Gunicorn\
- XGBoost\
- scikit-learn\
- pandas / numpy

## 📝 License

MIT License
