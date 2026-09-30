
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("House Price Prediction")
st.write("Enter the house details below.")

data = joblib.load("house_model.pkl")

area = st.number_input("Area (sq. ft.)", min_value=100, value=5000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Stories", min_value=1, max_value=5, value=2)

mainroad = st.selectbox("Main road", ["yes", "no"])
guestroom = st.selectbox("Guest room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot water heating", ["yes", "no"])
airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1)
prefarea = st.selectbox("Preferred area", ["yes", "no"])
furnishingstatus = st.selectbox(
    "Furnishing status",
    ["furnished", "semi-furnished", "unfurnished"]
)

if st.button("Predict Price"):
    house = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    house = pd.get_dummies(house, drop_first=True, dtype=int)
    house = house.reindex(columns=data["features"], fill_value=0)

    X = data["scaler"].transform(house)
    price = np.dot(X, data["w"]) + data["b"]

    st.success(f"Predicted price: ₹{price[0]:,.2f}")