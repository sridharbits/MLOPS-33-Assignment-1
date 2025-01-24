import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/sridharbits/MLOPS-33-Assignment-1.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "sridharbits"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "0dbb82ae320f2c39dba3cb988fedd62d5c421ab0"

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

## Load the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/sridharbits/MLOPS-33-Assignment-1.mlflow")

    ## Start multiple runs with different hyperparameters or models
    models = [
        {"model_name": "RandomForest", "n_estimators": 100, "max_depth": 10},
        {"model_name": "RandomForest", "n_estimators": 200, "max_depth": 20},
        {"model_name": "RandomForest", "n_estimators": 300, "max_depth": None},
    ]

    for model_params in models:
        with mlflow.start_run() as run:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            signature = infer_signature(X_train, y_train)

            param_grid = {
                'n_estimators': [model_params["n_estimators"]],
                'max_depth': [model_params["max_depth"]],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
            best_model = grid_search.best_estimator_

            # Predictions and evaluation
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_param("model_name", model_params["model_name"])
            mlflow.log_param("n_estimators", model_params["n_estimators"])
            mlflow.log_param("max_depth", model_params["max_depth"])
            mlflow.log_metric("accuracy", accuracy)

            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)

            mlflow.log_text(str(cm), "confusion_matrix.txt")
            mlflow.log_text(cr, "classification_report.txt")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != 'file':
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
            else:
                mlflow.sklearn.log_model(best_model, "model", signature=signature)

            # Save the model to disk
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            filename = model_path
            pickle.dump(best_model, open(filename, 'wb'))

            print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])
