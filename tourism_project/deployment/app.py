import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -------------------------------
# LOAD MODEL FROM HUGGING FACE HUB
# -------------------------------
model_path = hf_hub_download(
    repo_id="viveksardey/tourism-package-prediction-model",
    filename="tourism-package-prediction_model.joblib"
)
model = joblib.load(model_path)

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("Tourism Package Purchase Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the **Tourism Package**
offered by *Visit with Us*.

Please enter the customer details below to get the prediction.
""")

# -------------------------------
# USER INPUT FIELDS
# -------------------------------
Age = st.number_input("Customer Age", min_value=0, max_value=100, value=30)

Gender = st.selectbox("Gender", ["Male", "Female"])

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])

CityTier = st.selectbox("City Tier", [1, 2, 3])

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Self Employed", "Freelancer", "Company Owner", "Other"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "King", "Super Deluxe"]
)

Designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)

NumberOfTrips = st.number_input("Average Trips per Year", min_value=0, value=1)

NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)

PreferredPropertyStar = st.selectbox("Preferred Hotel Star Rating", [1, 2, 3, 4, 5])

NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)

Passport = st.selectbox("Passport Available?", [0, 1])

OwnCar = st.selectbox("Owns a Car?", [0, 1])

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=2)

DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)

# -------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "MaritalStatus": MaritalStatus,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase Package" if prediction == 1 else "Will Not Purchase Package"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
