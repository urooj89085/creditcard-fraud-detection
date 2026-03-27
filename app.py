# creditcard_fraud_app_fixed_final.py
import streamlit as st
import pandas as pd
import joblib

# ---------------------- Load Model & Scaler ----------------------
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details manually to check if it's fraudulent.")

# Load model & scaler (make sure these are uploaded in the app folder)
@st.cache_resource
def load_model():
    model = joblib.load('creditcard_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Hardcoded feature columns (order must match training)
feature_cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# ---------------------- Manual Transaction Form ----------------------
with st.form("transaction_form"):
    st.subheader("Enter Transaction Details:")

    cols = st.columns(4)
    V = {}
    for i in range(1, 29):
        col = cols[(i-1) % 4]
        V[f'V{i}'] = col.number_input(f"V{i}", value=0.0, format="%.6f")

    amount = st.number_input("Amount", value=0.0, format="%.2f")
    submitted = st.form_submit_button("Predict Fraud")

# ---------------------- Predict & Display ----------------------
if submitted:
    transaction = pd.DataFrame([{**V, 'Amount': amount}])

    # Ensure columns match training
    transaction = transaction[feature_cols]

    # Scale and predict
    transaction_scaled = scaler.transform(transaction)
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0,1]

    # Display results
    st.subheader("🔹 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud Probability: {probability:.2f}")

    st.subheader("🔹 Transaction Details")
    st.dataframe(transaction)

  
