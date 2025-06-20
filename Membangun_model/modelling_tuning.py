import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, precision_score, f1_score
from mlflow.models import infer_signature
from dotenv import load_dotenv
import os
import joblib

# Load kredensial dari .env
load_dotenv()
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not username or not password:
    raise EnvironmentError("MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD harus di-set")

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

# Set URI dan eksperimen
mlflow.set_tracking_uri("https://dagshub.com/DianAzizah13/Membangun-Model.mlflow")
mlflow.set_experiment("Crop Recommendation - Random Forest + GridSearch")

# Muat dataset
try:
    X_train = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/X_train.csv")
    X_test = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/X_test.csv")
    y_train = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/y_train.csv").values.squeeze()
    y_test = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/y_test.csv").values.squeeze()
except FileNotFoundError as e:
    print(f"Error: Pastikan file dataset ada di folder 'data_preprocessing/'.\n{e}")
    exit()

# GridSearch Hyperparameter
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# Manual Logging ke MLflow
with mlflow.start_run(run_name="RF_GridSearch_Manual_DagsHub"):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Logging parameter terbaik
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    # Logging metrik manual (lebih dari autolog)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score_macro", f1)
    mlflow.log_metric("precision_macro", precision)

    # Buat signature dari input dan output model
    y_sample = best_model.predict(X_train.iloc[:5])
    signature = infer_signature(X_train.iloc[:5], y_sample)
    
    # Simpan model secara lokal
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    print(f"[INFO] Best Params: {grid_search.best_params_}")
    print(f"[INFO] Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")