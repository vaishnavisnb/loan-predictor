import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")


@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, columns


try:
    model, columns = load_model()
except FileNotFoundError:
    st.error("Model files not found. Run `python train.py` first.")
    st.stop()

st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant's details and click **Predict** to see the result.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3], format_func=lambda x: "3+" if x == 3 else str(x))
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=130, step=10)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (meets guidelines)" if x == 1.0 else "Bad (does not meet guidelines)",
    )

if st.button("Predict", type="primary"):
    raw = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": float(loan_term),
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }
        ]
    )

    encoded = pd.get_dummies(raw)
    encoded = encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(encoded)[0]
    probability = model.predict_proba(encoded)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Loan likely APPROVED  (approval probability: {probability:.0%})")
    else:
        st.error(f"❌ Loan likely REJECTED  (approval probability: {probability:.0%})")

    st.progress(float(probability))
    st.caption("This is a demo built on a small public dataset. It is not real financial advice.")