import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------- Load Model & Scaler -------------------
model = joblib.load('creditcard_fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------- App Title -------------------
st.title("💳 Credit Card Fraud Checker")

# ------------------- Columns -------------------
columns = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

# ------------------- Example Transaction -------------------
transaction = [0,-1.3598,-0.0728,2.5363,1.3781,-0.3383,0.4623,0.2395,
               0.0986,0.3637,0.0907,-0.5515,-0.6178,-0.9913,-0.3111,
               1.4681,-0.4704,0.2079,0.0257,0.4039,0.2514,-0.0183,
               0.2778,-0.1104,0.0669,0.1285,-0.1891,0.1335,-0.0210,149.62]

# ------------------- Threshold -------------------
threshold = 0.5  # default fixed threshold

# ------------------- Predict Button -------------------
if st.button("Predict Transaction"):
    df_input = pd.DataFrame([transaction], columns=columns)
    df_scaled = scaler.transform(df_input)
    prob = model.predict_proba(df_scaled)[0][1]

    if prob >= threshold:
        st.write(f"❌ Fraudulent Transaction! Probability: {prob:.2f}")
    else:
        st.write(f"✅ Legitimate Transaction. Probability: {1-prob:.2f}")
