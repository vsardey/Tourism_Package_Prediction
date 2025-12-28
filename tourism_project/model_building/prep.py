import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from huggingface_hub import login


# -------------------------
# READ DATA
# -------------------------
DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")

# -------------------------
# TARGET VARIABLE
# -------------------------
target_col = "ProdTaken"   # 1 = Purchased, 0 = Not Purchased

# -------------------------
# HANDLE MISSING VALUES (simple strategy)
# -------------------------
df = df.dropna()

# -------------------------
# DROP UNIQUE IDENTIFIER
# -------------------------
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# -------------------------
# LABEL ENCODE BINARY COLUMNS
# -------------------------
binary_cols = ["Passport", "OwnCar"]

for col in binary_cols:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])

# -------------------------
# SPLIT X AND y
# -------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# SAVE FILES LOCALLY
# -------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
Ytrain.to_csv("Ytrain.csv", index=False)
Ytest.to_csv("Ytest.csv", index=False)

print("Train–test data created and saved successfully.")

# -------------------------
# UPLOAD TO HUGGING FACE
# -------------------------
api = HfApi()
#HF_TOKEN = os.getenv("TPP_HF_TOKEN")

#if HF_TOKEN is None:
#    raise ValueError("Environment variable 'TPP_HF_TOKEN' is not set. Please add your HF token.")

api = HfApi(token=os.getenv("TPP_HF_TOKEN"))

repo_id = "viveksardey/tourism-package-prediction"

# Create dataset repo if not present
api.create_repo(
    repo_id="tourism-package-prediction",
    repo_type="dataset",
    private=False,
    exist_ok=True
)

files = ["Xtrain.csv", "Xtest.csv", "Ytrain.csv", "Ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
    )

print("Train–test datasets uploaded to Hugging Face successfully.")
