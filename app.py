import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------- Load Model & Scaler -------------------
model = joblib.load('creditcard_fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------- App Title -------------------
st.title("💳 Credit Card Fraud Detection")

# ------------------- Instructions -------------------
st.markdown("""
**Instructions:**  
1. Enter the transaction details in **the same order** as below.  
2. Separate each value with a **comma** (`,`).  
3. Make sure you enter **all 30 values** including `Time`, `V1`-`V28`, and `Amount`.  
4. Click **Predict Fraud** to check if the transaction is legitimate or fraudulent.

**Column Order:**  
`Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount`
""")

# ------------------- Columns -------------------
columns = ['Time', 'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
           'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
           'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# ------------------- Normal Transaction Input -------------------
st.subheader("✅ Normal Transaction Input")
input_text = st.text_area("Paste full transaction row here (comma-separated):", 
                          "0,-1.3598,-0.0728,2.5363,1.3781,-0.3383,0.4623,0.2395,0.0986,0.3637,0.0907,-0.5515,-0.6178,-0.9913,-0.3111,1.4681,-0.4704,0.2079,0.0257,0.4039,0.2514,-0.0183,0.2778,-0.1104,0.0669,0.1285,-0.1891,0.1335,-0.0210,149.62")

if st.button("Predict Normal Transaction"):
    try:
        input_list = list(map(float, input_text.strip().split(',')))
        if len(input_list) != len(columns):
            st.error(f"❌ You must enter exactly {len(columns)} values!")
        else:
            input_df = pd.DataFrame([input_list], columns=columns)
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ Fraudulent Transaction! Probability: {prob:.2f}")
            else:
                st.success(f"✅ Legitimate Transaction. Probability: {1-prob:.2f}")
    except Exception as e:
        st.error(f"❌ Input format error: {e}")

# ------------------- Extreme Fraud Test -------------------
st.subheader("⚠️ Extreme Fraud Test")
if st.button("Run Extreme Fraud Test"):
    fraud_test = [0] + [5]*28 + [5000]  # synthetic extreme input
    input_df = pd.DataFrame([fraud_test], columns=columns)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f"⚠️ Extreme Fraud Detected! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Extreme Transaction detected as Legitimate. Probability: {1-prob:.2f}")

# ------------------- Optional Threshold Tweak -------------------
st.markdown("**Adjust Fraud Probability Threshold:**")
threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict With Custom Threshold"):
    try:
        input_list = list(map(float, input_text.strip().split(',')))
        if len(input_list) != len(columns):
            st.error(f"❌ You must enter exactly {len(columns)} values!")
        else:
            input_df = pd.DataFrame([input_list], columns=columns)
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]
            if prob >= threshold:
                st.error(f"⚠️ Fraudulent Transaction! Probability: {prob:.2f} (Threshold: {threshold})")
            else:
                st.success(f"✅ Legitimate Transaction. Probability: {prob:.2f} (Threshold: {threshold})")
    except Exception as e:
        st.error(f"❌ Input format error: {e}")
