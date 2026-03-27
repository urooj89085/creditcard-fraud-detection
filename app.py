# creditcard_fraud_app_fixed.py
import streamlit as st
import pandas as pd
import joblib

# ---------------------- Load Model, Scaler & Feature Columns ----------------------
@st.cache_resource
def load_resources():
    model = joblib.load('creditcard_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')  # List of columns in correct order
    return model, scaler, feature_cols

model, scaler, feature_cols = load_resources()

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details manually to check if it's fraudulent.")

# ---------------------- Manual Transaction Form ----------------------
with st.form("transaction_form"):
    st.subheader("Enter Transaction Details:")

    # Split V1–V28 into 4 columns
    cols = st.columns(4)
    V = {}
    for i in range(1, 29):
        col = cols[(i-1) % 4]
        V[f'V{i}'] = col.number_input(f"V{i}", value=0.0, format="%.6f")

    amount = st.number_input("Amount", value=0.0, format="%.2f")
    submitted = st.form_submit_button("Predict Fraud")

# ---------------------- Predict & Display ----------------------
if submitted:
    # Build transaction dataframe
    transaction = pd.DataFrame([{**V, 'Amount': amount}])

    # Ensure columns match training
    transaction = transaction[feature_cols]

    # Scale features
    transaction_scaled = scaler.transform(transaction)

    # Predict
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0,1]

    # Display prediction
    st.subheader("🔹 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud Probability: {probability:.2f}")

    # Display transaction details
    st.subheader("🔹 Transaction Details")
    st.dataframe(transaction)

    # Download CSV
    transaction['Predicted_Class'] = prediction
    transaction['Fraud_Probability'] = probability
    csv = transaction.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Transaction Prediction",
        data=csv,
        file_name='single_transaction_prediction.csv',
        mime='text/csv'
    )
