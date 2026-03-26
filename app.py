import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('creditcard_fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Card Fraud Detection")

# All columns in correct order
columns = ['Time', 'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
           'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
           'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# Create input form
input_data = {}
for col in columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Prediction button
if st.button("Predict Fraud"):
    input_df = pd.DataFrame([input_data], columns=columns)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {1-prob:.2f}")
