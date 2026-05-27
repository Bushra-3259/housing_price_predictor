import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import types
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Define the custom transformer class matching your notebook exactly
class AdvancedHousingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sf_coords = (37.7749, -122.4194)
        self.la_coords = (34.0522, -118.2437)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        X_out["dist_to_sf"] = np.sqrt((X_out["latitude"] - self.sf_coords[0])**2 + 
                                      (X_out["longitude"] - self.sf_coords[1])**2)
        X_out["dist_to_la"] = np.sqrt((X_out["latitude"] - self.la_coords[0])**2 + 
                                      (X_out["longitude"] - self.la_coords[1])**2)
        return X_out

# 2. Inject our class directly into __main__ namespaces to satisfy the pickle loader
# This bypasses the need to subclass or monkey-patch joblib entirely
try:
    setattr(sys.modules['__main__'], 'AdvancedHousingFeatures', AdvancedHousingFeatures)
except Exception:
    pass

# Create an fallback fake module just in case Streamlit isolates '__main__' completely
fake_main = types.ModuleType('__main__')
fake_main.AdvancedHousingFeatures = AdvancedHousingFeatures
sys.modules.setdefault('__main__', fake_main)

# 3. Load your trained cloud pipeline safely using standard joblib with caching
@st.cache_resource
def load_model():
    return joblib.load("california_housing_model.pkl")

model = load_model()

# 4. User Interface Setup
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
    median_income = st.number_input("Median Income (in tens of thousands)", value=8.32)

ocean_proximity = st.selectbox(
    "Ocean Proximity Category",
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
)

# 5. Run prediction calculations on click
if st.button("Calculate Estimated Value"):
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
    
    predicted_price = model.predict(input_df)[0]
    st.success(f"🎉 Estimated Median Neighborhood House Value: ${predicted_price:,.2f}")
