# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------------- Load Saved Model & Scaler ----------------------
model = joblib.load("creditcard_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent based on input features.")

# ---------------------- User Input ----------------------
st.sidebar.header("Enter Transaction Details:")

# Dynamically create inputs for all features except 'Class'
# Replace these feature names with your dataset's actual column names
feature_names = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                 'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                 'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

user_input = []
for feature in feature_names:
    val = st.sidebar.number_input(feature, value=0.0)
    user_input.append(val)

input_df = pd.DataFrame([user_input], columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_df)

# ---------------------- Prediction ----------------------
if st.button("Predict Fraud"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction is Safe. (Probability of Fraud: {prob:.2f})")

# ---------------------- Optional: Upload CSV for batch prediction ----------------------
st.header("Or upload a CSV for batch prediction")
uploaded_file = st.file_uploader("Choose CSV", type="csv")
if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    X_csv_scaled = scaler.transform(df_csv[feature_names])
    predictions = model.predict(X_csv_scaled)
    df_csv['Prediction'] = predictions
    st.write(df_csv.head())

    # Optional: Show confusion matrix if 'Class' column exists
    if 'Class' in df_csv.columns:
        cm = confusion_matrix(df_csv['Class'], df_csv['Prediction'])
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=[0,1]).plot(ax=ax)
        st.pyplot(fig)
