import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="dabirsagar/tourism-prediction-model",
        filename="best_tourism_pkg_prediction_model_v1.joblib",
    )
    return joblib.load(path)

model = load_model()

st.title("Tourism Package Purchase Predictor")

st.write("Enter customer details to predict purchase likelihood.")

# Inputs
Age = st.number_input("Age", 18, 100, 30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP"])

NumberOfPersonVisiting = st.number_input("Persons Visiting", 1, 10, 2)
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)
NumberOfTrips = st.number_input("Trips per Year", 0, 50, 2)
PreferredPropertyStar = st.selectbox("Hotel Rating", [1, 2, 3, 4, 5])

MonthlyIncome = st.number_input("Monthly Income", 5000, 500000, 50000)
PitchSatisfactionScore = st.slider("Pitch Satisfaction", 1, 5, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
NumberOfFollowups = st.number_input("Follow-ups", 0, 20, 2)
DurationOfPitch = st.number_input("Pitch Duration (min)", 1, 120, 15)

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

# Prepare input
input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
}])

# Predict
if st.button("Predict"):
    probability = model.predict_proba(input_df)[0, 1]
    if probability >= 0.5:
        st.success(f"Likely to purchase (probability: {probability:.2f})")
    else:
        st.error(f"Unlikely to purchase (probability: {probability:.2f})")
