# Data manipulation
import pandas as pd
import sklearn
import os

# Preprocessing
from sklearn.model_selection import train_test_split

# Hugging Face
from huggingface_hub import HfApi

# Load dataset
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/dabirsagar/tourism-dataset/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Target variable
target = "ProdTaken"

# Numerical features
numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome",
    "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

# Categorical features
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched", "Passport", "OwnCar"
]

# Predictors and target
X = df[numeric_features + categorical_features]
y = df[target]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to Hugging Face
for file in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="dabirsagar/mlops-tourism-prediction",
        repo_type="dataset",
    )
