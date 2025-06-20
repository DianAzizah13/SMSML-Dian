import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment("Crop Recommendation - Random Forest")

mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

# Load dataset
try:
    X_train = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/X_train.csv")
    X_test = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/X_test.csv")
    y_train = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/y_train.csv").values.ravel()
    y_test = pd.read_csv("Code/Eksperimen_SML/preprocessing/data_preprocessing/y_test.csv").values.ravel()
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

with mlflow.start_run(run_name="RF_Without_Tuning"):
    # Buat dan latih model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Evaluasi model
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    print(f"[Without Tuning] Accuracy: {acc:.4f}")
    print("\nMLflow autologging selesai.")
    print(f"Cek UI MLflow di {mlflow.get_tracking_uri()} untuk melihat hasilnya.")

