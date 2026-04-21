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

# ---------------- ROOT ----------------
@app.route("/")
def home():
    return "ML API Running Successfully"

# ---------------- PREDICT ----------------
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

        predicted_cgpa = round(max(0, min(10, ml_cgpa + adjustment)), 2)

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

        # -------- REASON --------
        reasons = []
        suggestions = []

        if attendance < 50:
            reasons.append("very low attendance")
            suggestions.append("increase attendance above 75%")
        elif attendance < 75:
            reasons.append("moderate attendance")
            suggestions.append("attend classes regularly")
        elif attendance > 85:
            reasons.append("excellent attendance")

        if internal < 40:
            reasons.append("poor internal marks")
            suggestions.append("revise concepts daily")
        elif internal < 60:
            reasons.append("average internal marks")
            suggestions.append("practice more questions")
        elif internal > 80:
            reasons.append("strong academic performance")

        if backlogs >= 3:
            reasons.append("multiple backlogs")
            suggestions.append("focus on clearing backlogs first")

        if study_hours < 2:
            reasons.append("very low study hours")
            suggestions.append("study at least 3-4 hours daily")
        elif study_hours > 5:
            reasons.append("good study consistency")

        # -------- TEXT --------
        if pass_fail == "Fail":
            reason_text = "The student is likely to fail due to " + ", ".join(reasons)
        else:
            reason_text = "The student is expected to pass with " + ", ".join(reasons)

        suggestion_text = (
            "Recommended actions: " + ", ".join(suggestions)
            if suggestions else "Maintain current performance."
        )

        # -------- RESPONSE --------
        return jsonify({
            "pass_fail": pass_fail,
            "risk_level": risk_level,
            "predicted_cgpa": predicted_cgpa,
            "reason": reason_text,
            "suggestion": suggestion_text,
            "model_pass_prediction": model_pass
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RENDER FIX ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
