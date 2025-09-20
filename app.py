import streamlit as st

# file: app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open("predict_insurance.pkl", "rb") as f1:
    model = pickle.load(f1)

with open("insurance_accuracy.pkl", "rb") as f2:
    model_accuracy = pickle.load(f2)



st.markdown("<h2 style='text-align: center;'><b>Medical Insurance Charge Prediction App</b></h2>",
             unsafe_allow_html=True)

# defines cols
col1, col2, col3 = st.columns(3)

# input fields
with col1:
    sex = st.selectbox( label="Select Your Gender",
                             options=['Male','Female'], index=0)
    bmi = st.number_input("Enter Your BMI (Body Mass Index)", min_value=10, max_value=50, value=10)


with col2:
    age = st.number_input("Enter Your Age", min_value=18, max_value=100, value=18)
    smoker = st.selectbox( label="Are You Smoker",
                             options=['Yes','No'], index=1)

with col3:
    children = st.number_input("How Many Children Do You Have", min_value=0, max_value=15, value=0)
    region = st.selectbox( label="Select Your Location",
                             options=['southeast','southwest','northeast','northwest'], index=0)
              

if st.button("Predict Placement"):
      if isinstance(sex, str):
        if sex.lower() == 'male':
            sex = 1
        elif sex.lower() == 'female':
            sex = 0

      if isinstance(smoker, str):
        if smoker.lower() in ['yes', 'y']:
            smoker = 1
        elif smoker.lower() in ['no', 'not', 'n']:
            smoker = 0

      input_data = pd.DataFrame([{
                'sex': sex,
                'bmi': bmi,
                'region': region,
                'smoker': smoker,
                'age': age,
                'children': children
      }])

      predicted_salary = model.predict(input_data)[0]

      st.success(f'Estimated Charges/Year: ${predicted_salary:.2f}')
      st.write(f'Model Accuracy : {model_accuracy * 100:.1f} %')