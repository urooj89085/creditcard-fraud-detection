import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------- Load Model & Scaler -------------------
@st.cache_resource  # caches the model and scaler for faster reloads
def load_model_scaler():
    model = joblib.load('creditcard_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_scaler()

# ------------------- App Config -------------------
st.set_page_config(page_title="💳 Credit Card Fraud Checker", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Predict whether a credit card transaction is fraudulent or legitimate using a pre-trained model.")

# ------------------- Columns & Example -------------------
columns = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

example_transaction = [0,-1.3598,-0.0728,2.5363,1.3781,-0.3383,0.4623,0.2395,
                       0.0986,0.3637,0.0907,-0.5515,-0.6178,-0.9913,-0.3111,
                       1.4681,-0.4704,0.2079,0.0257,0.4039,0.2514,-0.0183,
                       0.2778,-0.1104,0.0669,0.1285,-0.1891,0.1335,-0.0210,149.62]

# ------------------- Threshold -------------------
threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

# ------------------- Predict Button -------------------
st.subheader("Predict Example Transaction")
st.write("This is a sample transaction. Click the button to check fraud probability.")
if st.button("Predict Transaction"):
    df_input = pd.DataFrame([example_transaction], columns=columns)
    df_scaled = scaler.transform(df_input)
    prob = model.predict_proba(df_scaled)[0][1]
    extreme_flag = any(abs(x) > 10 for x in example_transaction[1:-1]) or abs(example_transaction[-1]) > 2000

    if extreme_flag:
        st.warning(f"⚠️ Extreme Transaction Detected! Probability: {prob:.2f}")
    elif prob >= threshold:
        st.error(f"❌ Fraudulent Transaction! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability: {1-prob:.2f}")

# ------------------- Developer Notes -------------------
with st.expander("Developer Notes / Info"):
    st.write("""
    - Model: Pre-trained `creditcard_fraud_model.pkl` (XGBoost or other classifier)
    - Scaler: StandardScaler used to normalize input features
    - Columns: Time, V1-V28, Amount
    - Threshold: Adjustable to control sensitivity
    - Extreme transactions are flagged for unusually high values
    """)
