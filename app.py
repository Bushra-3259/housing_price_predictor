import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load your trained cloud pipeline safely using caching to keep performance fast
@st.cache_resource
def load_model():
    return joblib.load("california_housing_model.pkl")

model = load_model()

# 2. Design your website's user interface headers
st.title("🏡 California House Price Predictor")
st.markdown("Enter neighborhood metrics below to estimate median home values using our tuned LightGBM Pipeline.")

st.header("Neighborhood Parameters")
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude (e.g., -122.23)", value=-122.23)
    latitude = st.number_input("Latitude (e.g., 37.88)", value=37.88)
    housing_median_age = st.number_input("Median House Age (Years)", value=41.0)
    total_rooms = st.number_input("Total Rooms in Block", value=880.0)

with col2:
    total_bedrooms = st.number_input("Total Bedrooms in Block", value=129.0)
    population = st.number_input("Block Population", value=322.0)
    households = st.number_input("Total Households", value=126.0)
    median_income = st.number_input("Median Income (in tens of thousands, e.g., 8.3 = $83,000)", value=8.32)

ocean_proximity = st.selectbox(
    "Ocean Proximity Category",
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
)

# 3. Run prediction calculations when the user clicks the action button
if st.button("Calculate Estimated Value"):
    # Convert inputs into a structured DataFrame matching your original X_train layout
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])
    
    # Your pipeline handles custom spatial distances, scaling, and target inversion automatically!
    predicted_price = model.predict(input_df)[0]
    
    # Display the final dollar prediction beautifully on screen
    st.success(f"🎉 Estimated Median Neighborhood House Value: ${predicted_price:,.2f}")
