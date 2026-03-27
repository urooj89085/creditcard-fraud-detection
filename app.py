# creditcard_fraud_app_final_safe.py
import streamlit as st
import pandas as pd
import joblib

# ---------------------- Load Model & Scaler ----------------------
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details manually to check if it's fraudulent.")

@st.cache_resource
def load_model():
    model = joblib.load('creditcard_fraud_model.pkl')  # Upload this
    scaler = joblib.load('scaler.pkl')                  # Upload this
    return model, scaler

model, scaler = load_model()

# ---------------------- Manual Transaction Form ----------------------
with st.form("transaction_form"):
    st.subheader("Enter Transaction Details:")

    # 4-column layout
    cols = st.columns(4)
    V = {}
    for i in range(1, 29):
        col = cols[(i-1) % 4]  # Correct order for input collection
        V[f'V{i}'] = col.number_input(f"V{i}", value=0.0, format="%.6f")

    amount = st.number_input("Amount", value=0.0, format="%.2f")
    submitted = st.form_submit_button("Predict Fraud")

# ---------------------- Predict & Display ----------------------
if submitted:
    # Create DataFrame
    transaction = pd.DataFrame([{**V, 'Amount': amount}])

    # ---------------------- SAFETY: Match Scaler Columns ----------------------
    # This ensures the input columns exactly match what the scaler expects
    transaction = transaction.reindex(columns=scaler.feature_names_in_)

    # Scale and predict
    transaction_scaled = scaler.transform(transaction)
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0,1]

    # ---------------------- Display Results ----------------------
    st.subheader("🔹 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud Probability: {probability:.2f}")

    st.subheader("🔹 Transaction Details")
    st.dataframe(transaction)

    # Optional CSV download
    transaction['Predicted_Class'] = prediction
    transaction['Fraud_Probability'] = probability
    csv = transaction.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Transaction Prediction",
        data=csv,
        file_name='single_transaction_prediction.csv',
        mime='text/csv'
    )
