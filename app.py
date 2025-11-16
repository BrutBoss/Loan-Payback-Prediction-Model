from flask import Flask, request, jsonify
import pickle
import pandas as pd
import sys

# -----------------------------
# Load trained model pipeline
# -----------------------------
try:
    with open("xgb_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Use sys.stderr for runtime errors instead of raising a complex exception
    print("ERROR: 'xgb_tuned_model.pkl' not found. Run training script first.", file=sys.stderr)
    sys.exit(1)

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON → pandas DataFrame
        record = request.get_json()

        if record is None:
            return jsonify({"error": "Empty or invalid JSON"}), 400

        # Handle both single record (dict) or list of records
        if isinstance(record, dict):
            df = pd.DataFrame([record])
        elif isinstance(record, list):
            df = pd.DataFrame(record)
        else:
            return jsonify({"error": "Input must be a dictionary (single record) or a list of dictionaries (multiple records)."}), 400

        # Predict probability of positive class (column 1: Payback probability)
        probabilities = model.predict_proba(df)[:, 1].tolist()

        # Format the response
        if len(probabilities) == 1:
            return jsonify({
                "loan_payback_probability": probabilities[0]
            })
        else:
             return jsonify({
                "loan_payback_probabilities": probabilities
            })

    except Exception as e:
        # Catch errors from the model or preprocessor (e.g., missing features)
        print(f"Prediction Error: {e}", file=sys.stderr)
        return jsonify({"error": f"Prediction failed. Check input features: {e}"}), 500


# -----------------------------
# Health Check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)