import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo

# -------------------------
# MLflow configuration
# -------------------------
mlflow.set_tracking_uri("https://unsymmetrically-nonvagrant-shaquana.ngrok-free.dev")
mlflow.set_experiment("Tourism_Package_Prediction_MLOps_Experiment")

# -------------------------
# LOAD SPLIT FILES FROM HF DATASET
# -------------------------
Xtrain_path = "hf://datasets/vsardey/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/vsardey/tourism-package-prediction/Xtest.csv"
Ytrain_path = "hf://datasets/vsardey/tourism-package-prediction/Ytrain.csv"
Ytest_path = "hf://datasets/vsardey/tourism-package-prediction/Ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
Ytrain = pd.read_csv(Ytrain_path)
Ytest = pd.read_csv(Ytest_path)

# -------------------------
# FEATURE GROUPS
# -------------------------
numeric_features = [
    "Age",
    "MonthlyIncome",
    "NumberOfTrips",
    "DurationOfPitch",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfChildrenVisiting",
]

categorical_features = [
    "Gender",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "MaritalStatus",
    "ProductPitched",
    "Designation",
]

# -------------------------
# PREPROCESSOR
# -------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# -------------------------
# HANDLE CLASS IMBALANCE
# -------------------------
pos = Ytrain.value_counts()[1]
neg = Ytrain.value_counts()[0]
class_weight = neg / pos

# -------------------------
# MODEL
# -------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# -------------------------
# HYPERPARAMETERS
# -------------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------
# TRAIN UNDER MLflow RUN
# -------------------------
with mlflow.start_run():

    grid = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1)
    grid.fit(Xtrain, Ytrain)

    best_model = grid.best_estimator_

    y_pred_test = best_model.predict(Xtest)

    report = classification_report(Ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    })

    mlflow.log_params(grid.best_params_)

    # Save model
    model_file = "tourism-package-prediction_model.joblib"
    joblib.dump(best_model, model_file)

    mlflow.log_artifact(model_file)

    # Upload model to Hugging Face Hub
    api = HfApi()
    repo_id = "vsardey/tourism-package-prediction-model"

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=model_file,
        path_in_repo=model_file,
        repo_id=repo_id,
        repo_type="model"
    )

print("Model training, tracking, and upload complete.")
