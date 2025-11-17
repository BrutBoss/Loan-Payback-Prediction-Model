📘 Loan Payback Prediction Model

This project builds a machine-learning system that predicts the probability that a customer will successfully repay a loan. The solution includes EDA, model training, hyperparameter tuning, scripts for reproducibility, containerization, and cloud deployment using Render.

1. Problem Descriptionof 

Financial institutions face significant risk when issuing loans. The goal this project is to develop a predictive model that estimates the likelihood of successful loan repayment based on historical loan and customer features.

Use cases include:

Reduce default risk

Support automated loan approval systems

Improve credit scoring

Prioritize low-risk customers

2. Dataset Overview

The dataset includes features such as loan details, customer financial indicators, demographic information, and the loan repayment outcome.

Target variable:

loan_status: 1 = repaid, 0 = default.

3. Exploratory Data Analysis (EDA)

The EDA focused on understanding the underlying data distribution, quality, and potential predictive power of features.

EDA Finding

Detail

Implication

Class Imbalance

The target variable (loan_status) was imbalanced, with approximately 80% of loans being repaid (class 1) and 20% defaulting (class 0).

Dictated the use of ROC AUC as the primary evaluation metric over raw accuracy.

Missing Values

Key financial variables like annual_income and debt_to_income_ratio contained a small percentage of missing values.

Handled via imputation (see Preprocessing Pipeline).

Feature Correlation

Strong positive correlation observed between credit_score and repayment status. Debt-to-income ratio showed a strong negative correlation with repayment.

These features were identified as critical drivers for the final model.

Categorical Variables

Variables like grade_subgrade and loan_purpose showed clear distinctions in repayment rates, suggesting they are highly predictive after One-Hot Encoding.

Used get_dummies for transformation within the pipeline.

4. Preprocessing Pipeline

To ensure the model receives clean, machine-readable data, a scikit-learn Pipeline incorporating a ColumnTransformer is used. This pipeline handles all data transformations automatically during both training and prediction.

Feature Type

Processing Step

Purpose

Categorical

One-Hot Encoding

Converts nominal categories (e.g., gender, loan_purpose) into binary features for the model.

Numerical

Imputation & Scaling (StandardScaler)

Fills any missing numerical values (if present) and scales all numerical inputs (e.g., loan_amount, annual_income) to a standard distribution (mean 0, variance 1).

The entire preprocessing logic is encapsulated within the saved xgb_tuned_model.pkl artifact.

5. Model Training and Hyperparameter Tuning

Multiple models were tested and evaluated on their ROC AUC score.

Comparative Model Performance

All models were evaluated based on the Area Under the ROC Curve (AUC) on the validation set.

Model

Baseline AUC

Best Tuned AUC

Tuning Method

Logistic Regression

0.911

0.912

GridSearchCV

Random Forest Classifier

0.880 (10 estimators)

0.907

Iterative Parameter Search

XGBoost Classifier

0.922

0.926

RandomizedSearchCV

Logistic Regression Tuning: Tuning using GridSearchCV (C, penalty, solver, class_weight) yielded a final AUC of 0.912 (up from 0.911 baseline).

Random Forest Tuning: The Random Forest model was significantly improved from its baseline (0.880) to 0.907 by adjusting n_estimators, max_depth, and min_samples_leaf using an iterative search strategy that stopped when the improvement threshold was not met.

XGBoost Tuning: The final model was tuned using RandomizedSearchCV across 30 iterations to efficiently optimize the key parameters, achieving the highest performance of 0.926 AUC.

Model Saved: The final, trained, and preprocessed pipeline is saved as xgb_tuned_model.pkl.

Final XGBoost Parameters:

The final model pipeline uses the following key hyperparameters found during the Randomized Search:

Parameter

Value

Description

n_estimators

411

Number of boosting rounds.

max_depth

3

Maximum depth of the individual trees.

learning_rate

0.204

Step size shrinkage to prevent overfitting.

subsample

0.969

Subsample ratio of the training instances.

colsample_bytree

0.835

Subsample ratio of columns per tree.

min_child_weight

5

Minimum sum of instance weight in a child.

gamma

0.036

Minimum loss reduction required to make a further partition.

6. Model Evaluation and Performance

The final XGBoost model was tuned to maximize the Area Under the ROC Curve (AUC) on the validation set.

Metric

Score

Interpretation

ROC AUC

0.926

Primary Metric. Indicates a 92.6% chance that the model will rank a randomly chosen positive case (repaid loan) higher than a randomly chosen negative case (defaulted loan). Excellent discriminator.

Accuracy

85.2%

Overall correct predictions. Given the 80/20 class imbalance, this is a strong score, but less reliable than AUC/F1.

Precision (Repaid)

~0.89

Of all loans predicted to be 'Repaid', 89% were actually repaid. Minimizes False Approvals (approving a loan that defaults).

Recall (Default)

~0.70

Of all actual 'Default' loans, 70% were correctly identified. Minimizes False Rejections (rejecting a safe loan).

F1 Score

0.76

Harmonic mean of precision and recall for the positive class. Represents a solid balance between minimizing false positives and false negatives.

7. Interactive Report

For a dynamic and visual exploration of the project's data insights, model comparisons, and core logic, open the single-page application:

➡️ View Interactive Report: loan_report.html

This interactive dashboard allows you to:

See model performance visualized in bar charts.

Interact with a Model Simulator to test hypothetical loan applications.

Explore the key findings of the EDA.

8. Project Structure

.
├── Datasets/
├── train.py
├── predict.py
├── app.py
├── xgb_tuned_model.pkl
├── requirements.txt
├── Dockerfile
├── README.md
├── loan_report_spa.html  <-- Interactive Report
└── Loan_Payback_Predictiont.ipynb



9. Training & Reproducibility

To train the model and generate the final xgb_tuned_model.pkl file, run the training script:

python train.py



Generate predictions for a single record via the command line (CLI example):

python predict.py '{"loan_amount": 15000.0, "credit_score": 720, "employment_duration": 5, "annual_income": 75000.0, "interest_rate": 10.5, "debt_to_income_ratio": 0.25, "has_house": "Y", "gender": "male", "marital_status": "single", "education_level": "bachelors", "employment_status": "employed", "loan_purpose": "auto", "grade_subgrade": "B3"}'



10. Model Deployment (Flask API)

The service exposes the /predict endpoint, which accepts both single JSON records and lists of records (for batch prediction).

Start locally:

python app.py



Required Input Schema

The prediction endpoint requires 13 features with their corresponding types in the JSON body.

Example Request (Single Record):

POST /predict
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



11. Docker Deployment

The application is containerized for consistent deployment.

Build the Docker image:

docker build -t loan-payback-service .



Run the container locally on port 9696:

docker run -dp 9696:9696 loan-payback-service



12. Cloud Deployment (Render)

The model is deployed on Render via the Docker image.

Add your service URL here:

[https://loan-payback-prediction-model.onrender.com/](https://loan-payback-prediction-model.onrender.com/)



13. Dependencies

Install requirements:

pip install -r requirements.txt



Includes:

flask: Web server framework.

gunicorn: Production WSGI HTTP server.

xgboost: Final classification algorithm.

scikit-learn==1.7.0: Core machine learning library (for pipeline and preprocessing).

pandas / numpy: Data handling libraries.