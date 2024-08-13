import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Will they move to another service provider :iphone:")

# Load the model
model = joblib.load('model_saved')

# Define the columns based on the processed DataFrame
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
           'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Get user inputs
gender = st.select_slider('Enter Customer Gender', ['Female', 'Male'])
SeniorCitizen = st.select_slider('Is the customer a Senior Citizen?', ['Yes', 'No'])
Partner = st.select_slider('Does the customer have a partner?', ['Yes', 'No'])
Dependents = st.select_slider('Is the customer dependent?', ['Yes', 'No'])
tenure = st.number_input('Enter tenure')
PhoneService = st.select_slider('Do they have phone service?', ['No', 'Yes'])
MultipleLines = st.select_slider('Do they have multiple lines?', ['No Phone Service', 'No', 'Yes'])
InternetService = st.select_slider('Do they have internet service?', ['DSL', 'Fiber Optics', 'No'])
OnlineSecurity = st.select_slider('Do they have online security?', ['No', 'Yes', 'No internet service'])
OnlineBackup = st.select_slider('Do they have online backup?', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.select_slider('Do they have device protection?', ['No', 'Yes', 'No internet service'])
TechSupport = st.select_slider('Do they have tech support?', ['No', 'Yes', 'No internet service'])
StreamingTV = st.select_slider('Do they have streaming TV?', ['No', 'Yes', 'No internet service'])
StreamingMovies = st.select_slider('Do they have streaming movies?', ['No', 'Yes', 'No internet service'])
Contract = st.select_slider('Contract type', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.select_slider('Is it paperless billing?', ['Yes', 'No'])
PaymentMethod = st.select_slider('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input('Enter monthly charges')
TotalCharges = st.number_input('Enter total charges')

def predict():
    # Encode the user inputs to match the model's expected format
    label_encodings = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {'No': 0, 'Yes': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No Phone Service': 0, 'No': 1, 'Yes': 2},
        'InternetService': {'DSL': 0, 'Fiber Optics': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'OnlineBackup': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'DeviceProtection': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'TechSupport': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'StreamingTV': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'StreamingMovies': {'No': 0, 'Yes': 2, 'No internet service': 1},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1}
    }

    row = [
        label_encodings['gender'][gender],
        label_encodings['SeniorCitizen'][SeniorCitizen],
        label_encodings['Partner'][Partner],
        label_encodings['Dependents'][Dependents],
        tenure,
        label_encodings['PhoneService'][PhoneService],
        label_encodings['MultipleLines'][MultipleLines],
        label_encodings['InternetService'][InternetService],
        label_encodings['OnlineSecurity'][OnlineSecurity],
        label_encodings['OnlineBackup'][OnlineBackup],
        label_encodings['DeviceProtection'][DeviceProtection],
        label_encodings['TechSupport'][TechSupport],
        label_encodings['StreamingTV'][StreamingTV],
        label_encodings['StreamingMovies'][StreamingMovies],
        label_encodings['Contract'][Contract],
        label_encodings['PaperlessBilling'][PaperlessBilling],
        label_encodings['PaymentMethod'][PaymentMethod],
        MonthlyCharges,
        TotalCharges
    ]

    # Create a DataFrame
    X = pd.DataFrame([row], columns=columns)

    # Check if the feature names match
    if hasattr(model, 'feature_names_in_'):
        if set(model.feature_names_in_) != set(X.columns):
            st.error("Feature names do not match! Please check the input features.")
            st.write("Model expects:", model.feature_names_in_)
            st.write("Input features provided:", X.columns.tolist())
        else:
            # Make a prediction
            prediction = model.predict(X)[0]

            if prediction == 0:
                st.success('They are likely to stay :thumbsup:')
            else:
                st.error('They will leave :thumbsdown:')
    else:
        st.error("Model does not have feature_names_in_ attribute to check feature names.")

# Button to trigger prediction
st.button('Predict', on_click=predict)
