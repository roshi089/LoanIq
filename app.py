from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from bank_rules import BANK_RULES

app = Flask(__name__)
CORS(app)

print("Loading model artifacts...")

model          = joblib.load("best_model.pkl")
model_name     = joblib.load("best_model_name.pkl")
shap_explainer = joblib.load("shap_explainer.pkl")
scaler         = joblib.load("scaler.pkl")
encoders       = joblib.load("encoder.pkl")
feature_names  = joblib.load("feature_names.pkl")
X_train        = joblib.load("X_train.pkl")

print(f"✅ Model loaded: {model_name}")

print("Building LIME explainer...", end=" ", flush=True)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = np.array(X_train),
    feature_names = feature_names,
    class_names   = ["Rejected", "Approved"],
    mode          = "classification",
    random_state  = 42
)
print("done.")
print("✅ All artifacts ready.\n")

CATEGORICAL_COLS = [
    'employment_status', 'education_level',
    'residence_type', 'loan_purpose', 'marital_status'
]
def get_eligible_banks(data):
    eligible = []
    rejected = []

    credit_score = float(data.get("credit_score", 0))
    dti_ratio    = float(data.get("dti_ratio", 0))

    for bank, rules in BANK_RULES.items():
        if (credit_score >= rules["min_credit_score"] and
            dti_ratio    <= rules["max_dti"]):
            eligible.append({
                "bank":          bank,
                "interest_rate": rules["interest_rate"],
                "wait_period":   rules["wait_period"]
            })
        else:
            rejected.append(bank)

    return eligible, rejected

@app.route("/")
def index():
    return send_from_directory(".", "loan_approval_ui.html")

def preprocess(data):
    df = pd.DataFrame([data])
    for col in CATEGORICAL_COLS:
        le  = encoders[col]
        val = df[col].iloc[0]
        df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])
    df = df[feature_names]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    return df, df_scaled

def get_shap_values(df_raw, df_scaled):
    input_df  = df_raw if model_name == "Random Forest" else df_scaled
    shap_vals = shap_explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    values = np.array(shap_vals).flatten()
    return dict(zip(feature_names, [round(float(v), 4) for v in values]))

def get_lime_explanation(df_raw, df_scaled):
    input_arr = (df_raw if model_name == "Random Forest" else df_scaled).values[0]
    exp = lime_explainer.explain_instance(input_arr, model.predict_proba, num_features=6)
    result = []
    for feat, weight in exp.as_list():
        direction = "positively" if weight > 0 else "negatively"
        result.append(f"{feat} influenced the decision {direction} ({weight:+.3f})")
    return result

def get_counterfactuals(data):
    cfs = []
    if data["credit_score"] < 650:
        cfs.append(f"Increase Credit Score by <strong>{650 - data['credit_score']} points</strong> (current: {data['credit_score']}, target: 650+)")
    if data["previous_default"] == 1:
        cfs.append("Resolve your <strong>previous default record</strong> — maintain on-time payments for 12+ months")
    if data["credit_history"] == 0:
        cfs.append("Build a <strong>positive credit history</strong> — use a secured credit card and pay on time for 6+ months")
    if data["dti_ratio"] > 0.45:
        cfs.append(f"Reduce DTI Ratio from <strong>{data['dti_ratio']:.2f} → 0.45</strong> by paying down existing loans")
    total_income = data["applicant_income"] + data["coapplicant_income"]
    if total_income > 0 and data["loan_amount"] / (total_income * 12) > 8:
        safe = int(total_income * 12 * 8)
        cfs.append(f"Reduce Loan Amount by <strong>₹{int(data['loan_amount'] - safe):,}</strong> (safe limit: ₹{safe:,})")
    if data["savings_balance"] < data["applicant_income"] * 3:
        cfs.append(f"Increase Savings to at least <strong>₹{int(data['applicant_income'] * 3):,}</strong> (3× monthly income)")
    return cfs[:4]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON received"}), 400
        df_raw, df_scaled = preprocess(data)
        input_df    = df_raw if model_name == "Random Forest" else df_scaled
        probability = float(model.predict_proba(input_df)[0][1])
        prediction  = int(probability >= 0.5)
        eligible_banks,rejected_banks=get_eligible_banks(data)
        return jsonify({
            "prediction":       prediction,
            "probability":      round(probability, 4),
            "model":            model_name,
            "eligible_banks": eligible_banks,
            "rejected_banks": rejected_banks,
            "shap_values":      get_shap_values(df_raw, df_scaled),
            "lime_explanation": get_lime_explanation(df_raw, df_scaled),
            "counterfactuals":  get_counterfactuals(data) if prediction == 0 else []
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model_name, "features": len(feature_names)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
