import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------- Load Model & Scaler -------------------
st.sidebar.title("💾 Load Model & Scaler")
model_file = st.sidebar.file_uploader("Upload trained XGBoost model (.pkl)", type=["pkl", "joblib"])
scaler_file = st.sidebar.file_uploader("Upload Scaler (.pkl)", type=["pkl", "joblib"])

if model_file and scaler_file:
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    st.success("✅ Model and scaler loaded successfully!")

    # ------------------- App Title -------------------
    st.title("💳 Credit Card Fraud Detection")
    
    st.markdown("""
    **Instructions:**  
    1. Enter the transaction details in **the same order** as shown below.  
    2. Separate each value with a **comma** (`,`).  
    3. Make sure you enter **all 30 values** including `Time`, `V1`-`V28`, and `Amount`.  
    4. Click **Predict Fraud** to check if the transaction is legitimate or fraudulent.
    """)

    # ------------------- Columns -------------------
    columns = [
        'Time', 'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
    ]
    st.markdown(f"**Column Order:** {', '.join(columns)}")

    # ------------------- Input Field -------------------
    input_text = st.text_area(
        "Paste full transaction row here (comma-separated):",
        "0,-1.3598,-0.0728,2.5363,1.3781,-0.3383,0.4623,0.2395,0.0986,0.3637,0.0907,-0.5515,-0.6178,-0.9913,-0.3111,1.4681,-0.4704,0.2079,0.0257,0.4039,0.2514,-0.0183,0.2778,-0.1104,0.0669,0.1285,-0.1891,0.1335,-0.0210,149.62"
    )

    # ------------------- Threshold Slider -------------------
    threshold = st.slider("Adjust Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

    # ------------------- Predict Button -------------------
    if st.button("Predict Fraud"):
        try:
            # Convert input string to list of floats
            input_list = list(map(float, input_text.strip().split(',')))

            if len(input_list) != len(columns):
                st.error(f"❌ You must enter exactly {len(columns)} values!")
            else:
                # Check for extreme values
                extreme_flag = False
                if abs(input_list[-1]) > 2000:  # Amount > 2000 considered extreme
                    extreme_flag = True
                if any(abs(x) > 10 for x in input_list[1:-1]):  # V1-V28 extreme
                    extreme_flag = True

                # Create DataFrame and scale input
                input_df = pd.DataFrame([input_list], columns=columns)
                input_scaled = scaler.transform(input_df)

                # Predict
                prob = model.predict_proba(input_scaled)[0][1]
                pred = model.predict(input_scaled)[0]

                # Display results
                if extreme_flag:
                    st.warning(f"⚠️ Extreme Transaction Detected! Probability: {prob:.2f}")
                elif prob >= threshold:
                    st.error(f"❌ Fraudulent Transaction! Probability: {prob:.2f}")
                else:
                    st.success(f"✅ Legitimate Transaction. Probability: {1-prob:.2f}")

        except ValueError:
            st.error("❌ Invalid input! Make sure all values are numeric and separated by commas.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

else:
    st.info("Please upload both the trained model and scaler to start prediction.")
