import streamlit as st
import pickle
import numpy as np

# Load Scalers and Models
scaler_cibil = pickle.load(open('scaler_cibil.pkl', 'rb'))
model_cibil = pickle.load(open('model_cibil.pkl', 'rb'))
scaler_loan = pickle.load(open('scaler_loan.pkl', 'rb'))
model_loan = pickle.load(open('model_loan.pkl', 'rb'))

st.title('🏦 Loan Prediction App with EMI & Interest Calculator 💳')

st.header('Enter Your Details:')

# User Inputs
Gender = st.selectbox('Gender:', ['Male', 'Female'])
Married = st.selectbox('Married:', ['Yes', 'No'])
Education = st.selectbox('Education Level:', ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox('Self Employed:', ['Yes', 'No'])
ApplicantIncome = st.number_input('Applicant Monthly Income:', min_value=0)
Property_Area = st.selectbox('Property Area:', ['Urban', 'Semi-Urban', 'Rural'])
Loan_Duration = st.number_input('Loan Duration (in Years):', min_value=1)

# New Input: Interest Rate
interest_rate = st.number_input('Enter Interest Rate (% per annum):', min_value=1.0, max_value=20.0, value=8.5)

# Manual Encoding
Gender = 1 if Gender == 'Male' else 0
Married = 1 if Married == 'Yes' else 0
Education = 0 if Education == 'Graduate' else 1
Self_Employed = 1 if Self_Employed == 'Yes' else 0
Property_Area = {'Urban': 2, 'Semi-Urban': 1, 'Rural': 0}[Property_Area]

# Features for CIBIL Prediction
features_cibil = np.array([[Gender, Married, Education, Self_Employed,
                            ApplicantIncome, Property_Area, Loan_Duration]])
min_income_required = 25000
if st.button('Predict CIBIL Score & Loan Details'):
    if ApplicantIncome == 0:
        st.warning('Please Enter the income')
    elif ApplicantIncome < min_income_required:
        st.error(f"⚠️ Sorry, minimum income of ₹{min_income_required} is required to process the loan application.")
    else:
    # Predict CIBIL Score
        scaled_cibil = scaler_cibil.transform(features_cibil)
        cibil_pred = model_cibil.predict(scaled_cibil)[0]
        st.success(f'✅ Predicted CIBIL Score: {cibil_pred:.2f}')

    # Predict Loan Amount
        feature_loan = np.array([[ApplicantIncome, cibil_pred]])
        scaled_loan = scaler_loan.transform(feature_loan)
        loan_pred = model_loan.predict(scaled_loan)[0]
        st.success(f'💰 Eligible Loan Amount: ₹{loan_pred:.2f}')

    # EMI Calculation
        P = loan_pred  # Principal
        R = interest_rate / (12 * 100)  # Monthly Interest Rate
        N = Loan_Duration * 12  # Total Loan Months

        try:
            emi = (P * R * (1 + R)**N) / ((1 + R)**N - 1)
        except ZeroDivisionError:
            emi = 0  # If interest rate = 0

        total_payment = emi * N
        total_interest = total_payment - P

    # Display Loan Details
        st.header('📋 Loan Repayment Details:')
        st.info(f'📌 Monthly EMI: ₹{emi:.2f}')
        st.info(f'📌 Total Payment (Principal + Interest): ₹{total_payment:.2f}')
        st.info(f'📌 Total Interest Payable: ₹{total_interest:.2f}')