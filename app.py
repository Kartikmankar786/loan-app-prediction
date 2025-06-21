import streamlit as st
import pickle
import numpy as np

# Load Scalers and Models properly
scaler_cibil = pickle.load(open('scaler_cibil.pkl', 'rb'))
model_cibil = pickle.load(open('model_cibil.pkl', 'rb'))
scaler_loan = pickle.load(open('scaler_loan.pkl', 'rb'))    # FIXED this line
model_loan = pickle.load(open('model_loan.pkl', 'rb'))

st.title('Loan Prediction App 🏦💰')

st.header('Enter Details Below:')

# User Inputs
Gender = st.selectbox('Gender:', ['Male', 'Female'])
Married = st.selectbox('Married:', ['Yes', 'No'])
Education = st.selectbox('Education Level:', ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox('Self Employed:', ['Yes', 'No'])
ApplicantIncome = st.number_input('Applicant Monthly Income:', min_value=0)
Property_Area = st.selectbox('Property Area:', ['Urban', 'Semi-Urban', 'Rural'])
Loan_Duration = st.number_input('Enter the Loan Duration in Years:', min_value=1)

# Manual Encoding (FIXED)
Gender = 1 if Gender == 'Male' else 0
Married = 1 if Married == 'Yes' else 0
Education = 0 if Education == 'Graduate' else 1
Self_Employed = 1 if Self_Employed == 'Yes' else 0  # FIXED
Property_Area = {'Urban': 2, 'Semi-Urban': 1, 'Rural': 0}[Property_Area]

# Combine all features
features_cibil = np.array([[Gender, Married, Education, Self_Employed,
                            ApplicantIncome, Property_Area, Loan_Duration]])

if st.button('Predict CIBIL Score'):
    scaled_cibil = scaler_cibil.transform(features_cibil)
    cibil_pred = model_cibil.predict(scaled_cibil)[0]
    st.success(f'✅ Predicted CIBIL Score: {cibil_pred:.2f}')

    st.header('Loan Amount Prediction 💳:')
    feature_loan = np.array([[ApplicantIncome, cibil_pred]])
    scaled_loan = scaler_loan.transform(feature_loan)
    loan_pred = model_loan.predict(scaled_loan)[0]
    st.success(f'💰 Eligible Loan Amount: ₹{loan_pred:.2f}')
