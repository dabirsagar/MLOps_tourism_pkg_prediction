# Data + preprocessing
import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain = pd.read_csv("hf://datasets/dabirsagar/tourism-dataset/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/dabirsagar/tourism-dataset/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/dabirsagar/tourism-dataset/ytrain.csv")
ytest  = pd.read_csv("hf://datasets/dabirsagar/tourism-dataset/ytest.csv")

numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome",
    "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched", "Passport", "OwnCar"
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    "xgbclassifier__n_estimators": [100],
    "xgbclassifier__max_depth": [3],
    "xgbclassifier__colsample_bytree": [0.5],
    "xgbclassifier__colsample_bylevel": [0.5],
    "xgbclassifier__learning_rate": [0.05],
    "xgbclassifier__reg_lambda": [0.5],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i, param_set in enumerate(results["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })

    model_path = "best_tourism_pkg_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model saved:", model_path)

    repo_id, repo_type = "dabirsagar/tourism-prediction-model", "model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' exists. Uploading...")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
