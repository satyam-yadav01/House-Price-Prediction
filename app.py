import streamlit as st
import pandas as pd
import joblib


st.title("Real Estate Price Prediction App")
st.write("Enter property details to predict the price.")


# Load trained model
model = joblib.load("model.pkl")


# Inputs
area = st.number_input("Area (sq ft)", min_value=200, max_value=20000, value=2000)
bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6])
bathrooms = st.selectbox("Bathrooms", [1,2,3,4,5])
stories = st.selectbox("Stories", [1,2,3,4])
parking = st.selectbox("Parking", [0,1,2,3])


mainroad = st.selectbox("Main Road Access", ["yes","no"])
guestroom = st.selectbox("Guest Room", ["yes","no"])
basement = st.selectbox("Basement", ["yes","no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes","no"])
airconditioning = st.selectbox("Air Conditioning", ["yes","no"])
prefarea = st.selectbox("Preferred Area", ["yes","no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished","semi-furnished","unfurnished"])


# Create input DataFrame
input_data = pd.DataFrame({
'area': [area],
'bedrooms': [bedrooms],
'bathrooms': [bathrooms],
'stories': [stories],
'parking': [parking],
'mainroad': [mainroad],
'guestroom': [guestroom],
'basement': [basement],
'hotwaterheating': [hotwaterheating],
'airconditioning': [airconditioning],
'prefarea': [prefarea],
'furnishingstatus': [furnishingstatus]
})


# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:,.2f}")