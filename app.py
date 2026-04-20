from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# ===== LOAD MODELS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "ml_services", "models")

pass_fail_model = joblib.load(os.path.join(model_path, "svm_pass_fail_model.pkl"))
cgpa_model = joblib.load(os.path.join(model_path, "cgpa_model.pkl"))
risk_model = joblib.load(os.path.join(model_path, "rf_risk_model.pkl"))
scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))

# ===== HOME ROUTE =====
@app.route('/')
def home():
    return "API Running Successfully"

# ===== PREDICT ROUTE =====
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data["attendance_percent"],
            data["internal_marks"],
            data["assignment_score"],
            data["quiz_score"],
            data["study_hours_per_day"],
            data["previous_cgpa"],
            data["backlogs"]
        ]
        scaled = scaler.transform([features])
        pass_fail = pass_fail_model.predict(scaled)[0]
        cgpa = cgpa_model.predict(scaled)[0]
        risk = risk_model.predict(scaled)[0]

        return jsonify({
            "pass_fail": str(pass_fail),
            "predicted_cgpa": float(cgpa),
            "risk_level": str(risk)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== RUN =====
if __name__ == '__main__':
    app.run(debug=True)
