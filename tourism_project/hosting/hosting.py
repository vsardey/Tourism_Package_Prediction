from huggingface_hub import HfApi, create_repo
import os

# -------------------------------
# AUTHENTICATION
# -------------------------------
#HF_TOKEN = os.getenv("TPP_HF_TOKEN")

#if HF_TOKEN is None:
#    raise ValueError("Environment variable 'TPP_HF_TOKEN' is not set. Please add your HF token.")

api = HfApi(token=os.getenv("TPP_HF_TOKEN"))

# -------------------------------
# HUGGING FACE SPACE DETAILS
# -------------------------------
space_repo_id = "viveksardey/TourismPackagePredictionFrontend"   # updated repo_id

# -------------------------------
# UPLOAD STREAMLIT APP FILES
# -------------------------------
api.upload_folder(
    folder_path="/content/tourism_project/deployment",   # local deployment app folder
    repo_id=space_repo_id,
    repo_type="space",
    path_in_repo="",                            # root of space repo
)

print("App successfully deployed to Hugging Face Space.")
print(f"Visit: https://huggingface.co/spaces/{space_repo_id}")
