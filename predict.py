import pickle
import pandas as pd
import sys
import json

  
# Load trained pipeline

try:
    with open("xgb_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: The model file 'xgb_tuned_model.pkl' was not found.")
    print("Please ensure you have run 'train_save_model.py' successfully.")
    sys.exit(1)


  
# Predict function
  
def predict(record: dict):
    # Convert the input dictionary (single record) into a pandas DataFrame
    # This maintains the structure required by the pipeline's preprocessor
    df = pd.DataFrame([record])
    
    # Predict the probability of the positive class (column 1)
    prob = model.predict_proba(df)[0][1]
    
    # Return the probability as a standard float
    return float(prob)

  
# CLI usage
  

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        # Updated the example to show the structure of the required feature set
        print("  python predict.py '{\"loan_amount\": 15000.0, \"credit_score\": 720, \"employment_duration\": 5, \"has_house\": \"Y\"}'")
        sys.exit(1)

    try:
        record = json.loads(sys.argv[1])
        result = predict(record)

        print(f"Probability of loan payback: {result:.4f}")
    except json.JSONDecodeError:
        print("Error: Could not parse input argument. Ensure the JSON string is properly formatted.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Note: If this error occurs, it usually means a feature name or type in the input JSON
        # does not match the data the preprocessor was trained on.
        sys.exit(1)