
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("🚀 Customer Churn Prediction")
st.write("Predict whether a customer will churn or not")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Encoding (same as training logic)
contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
internet_map = {"DSL":0, "Fiber optic":1, "No":2}

contract = contract_map[contract]
internet = internet_map[internet]

# Feature array (IMPORTANT: order matters)
features = np.array([[tenure, monthly_charges, total_charges, contract, internet]])

# Scale
features = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk ({probability:.2f})")
    else:
        st.success(f"✅ Low Churn Risk ({probability:.2f})")

    st.progress(int(probability * 100))

# Info
st.info("Customers with low tenure and high charges are more likely to churn.")
