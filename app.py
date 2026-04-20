from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ---------------- BASE PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ml_services", "models")

# ---------------- LOAD MODELS ----------------
pass_fail_model = joblib.load(os.path.join(model_path, "svm_pass_fail_model.pkl"))
scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
risk_model = joblib.load(os.path.join(model_path, "rf_risk_model.pkl"))
cgpa_model = joblib.load(os.path.join(model_path, "cgpa_model.pkl"))

# ---------------- ROOT (FIX FOR 404) ----------------
@app.route("/")
def home():
    return "ML API Running Successfully"

# ---------------- PREDICTION API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # -------- INPUT --------
        attendance = float(data["attendance_percent"])
        internal = float(data["internal_marks"])
        assignment = float(data["assignment_score"])
        quiz = float(data["quiz_score"])
        study_hours = float(data["study_hours_per_day"])
        prev_cgpa = float(data["previous_cgpa"])
        backlogs = int(data["backlogs"])

        # -------- ENCODING --------
        participation_map = {"Low": 0, "Medium": 1, "High": 2}
        submission_map = {"No": 0, "Yes": 1}

        participation = participation_map[data["class_participation"]]
        submission = submission_map[data["submission_regular"]]

        # -------- VALIDATION --------
        if not (0 <= prev_cgpa <= 10):
            return jsonify({"error": "CGPA must be between 0 and 10"}), 400

        if not (0 <= attendance <= 100):
            return jsonify({"error": "Attendance must be between 0 and 100"}), 400

        # -------- FEATURES --------
        features = np.array([[ 
            attendance, internal, assignment, quiz,
            study_hours, prev_cgpa, backlogs,
            participation, submission
        ]])

        # -------- CGPA --------
        ml_cgpa = cgpa_model.predict(features)[0]
        adjustment = 0

        if attendance < 50: adjustment -= 0.5
        if internal < 40: adjustment -= 0.5
        if study_hours < 2: adjustment -= 0.3
        if backlogs > 2: adjustment -= 0.7

        if attendance > 85: adjustment += 0.3
        if internal > 80: adjustment += 0.3
        if study_hours > 5: adjustment += 0.2

        predicted_cgpa = ml_cgpa + adjustment

        if attendance > 90 and internal > 90 and backlogs == 0:
            predicted_cgpa = max(predicted_cgpa, prev_cgpa)

        predicted_cgpa = round(max(0, min(10, predicted_cgpa)), 2)

        # -------- PASS FAIL --------
        pass_fail = "Pass" if predicted_cgpa >= 5 else "Fail"

        # -------- SVM --------
        features_scaled = scaler.transform(features)
        svm_pred = pass_fail_model.predict(features_scaled)[0]
        model_pass = "Pass" if svm_pred == 1 else "Fail"

        # -------- RISK --------
        if pass_fail == "Fail":
            risk_level = "High"
        elif attendance < 60:
            risk_level = "High"
        elif attendance < 80:
            risk_level = "Medium"
        else:
            risk_level = "Low" if internal >= 60 and backlogs <= 1 else "Medium"

        if backlogs >= 3:
            risk_level = "High"

        # -------- RESPONSE --------
        return jsonify({
            "pass_fail": pass_fail,
            "risk_level": risk_level,
            "predicted_cgpa": predicted_cgpa,
            "model_pass_prediction": model_pass
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
