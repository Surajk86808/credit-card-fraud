import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("credit_card_model(decision tree).pkl")
scaler = joblib.load("scaler_credit_card_model(decision tree).pkl")
feature_columns = joblib.load("columns_credit_card_model(decision tree).pkl")


st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction data below to predict if it's **fraudulent or legitimate**.")


user_data = {}
for feature in feature_columns:
    user_data[feature] = st.number_input(f"Enter {feature}", value=0.0)


input_df = pd.DataFrame([user_data])


input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])


if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Fraudulent Transaction Detected!")
    else:
        st.success("✅ Prediction: Legitimate Transaction")


with st.expander("📄 Show Entered Data"):
    st.dataframe(input_df)
