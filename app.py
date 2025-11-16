from flask import Flask, request, jsonify, render_template
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
    print("ERROR: 'xgb_tuned_model.pkl' not found. Run training script first.", file=sys.stderr)
    sys.exit(1)

# -----------------------------
# Initialize Flask App
# -----------------------------
# Tell Flask to look for templates in the current directory's 'templates' folder
app = Flask(__name__, template_folder='templates')

# -----------------------------
# Route: Serve the UI
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Route: API Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        record = request.get_json()
        if record is None:
            return jsonify({"error": "Empty or invalid JSON"}), 400

        if isinstance(record, dict):
            df = pd.DataFrame([record])
        elif isinstance(record, list):
            df = pd.DataFrame(record)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        probabilities = model.predict_proba(df)[:, 1].tolist()

        if len(probabilities) == 1:
            return jsonify({"loan_payback_probability": probabilities[0]})
        else:
             return jsonify({"loan_payback_probabilities": probabilities})

    except Exception as e:
        print(f"Prediction Error: {e}", file=sys.stderr)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# -----------------------------
# Health Check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)