import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

 
# 1. Load dataset


train = pd.read_csv('/workspaces/Loan-Payback-Prediction-Model/Datasets/train.csv')


# Data Preparation

X = train.drop(['loan_paid_back', 'id'], axis=1, errors='ignore')
y = train['loan_paid_back']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
 
# Drop rows with missing values in key predictor and target columns
X = X.dropna(subset=[
    'annual_income', 'debt_to_income_ratio', 'credit_score',
    'loan_amount', 'interest_rate', 'gender', 'marital_status',
    'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade'
])
y = y.dropna()

# Split data into training and validation sets (80/20) with stratified sampling
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=19
)

# Define preprocessing pipelines for numeric and categorical features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Combine both pipelines into a single preprocessor using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

 
# 4. Model Training
 
#  Define the specific best parameters found in your search
best_params = {
    'n_estimators': 411,
    'max_depth': 3,
    'learning_rate': 0.2037791625223493,
    'subsample': 0.9693757124071966,
    'colsample_bytree': 0.8349427857182792,
    'min_child_weight': 5,
    'gamma': 0.0358888351658756,
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'random_state': 19,
    'n_jobs': -1
}

# Instantiate the final model with these parameters
xgb_final = XGBClassifier(**best_params)

# Recreate the pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_final)
])

# Train the model on the full training data
print("Training final model...")
final_pipeline.fit(X_train, y_train)
print("Training complete.")

# 5. Save the trained pipeline to a file using pickle
filename = 'xgb_tuned_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(final_pipeline, file)

print(f"✅ Model successfully saved to {filename}")


