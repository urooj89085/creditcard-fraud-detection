# creditcard_fraud_app_manual.py
import streamlit as st
import pandas as pd
import joblib

# ---------------------- Load Model & Scaler ----------------------
@st.cache_resource
def load_model():
    model = joblib.load('creditcard_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details manually to check if it's fraudulent.")

# ---------------------- Manual Transaction Form ----------------------
with st.form("transaction_form"):
    st.subheader("Enter Transaction Details:")
    
    # All 28 V features + Amount (Time removed)
    V = {}
    for i in range(1, 29):
        V[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.6f")
    amount = st.number_input("Amount", value=0.0, format="%.2f")

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    # Build DataFrame for single transaction
    transaction = pd.DataFrame([ {**V, 'Amount': amount} ])
    
    # Scale and predict
    transaction_scaled = scaler.transform(transaction)
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0,1]

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud Probability: {probability:.2f}")

    # Show transaction details
    st.subheader("Transaction Details")
    st.dataframe(transaction)
