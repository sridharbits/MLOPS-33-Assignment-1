import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import yaml
import os
print("Current Working Directory:", os.getcwd())

# Hyperparameter tuning function
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

# Training function
def train(data_path):
    # Load dataset
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict and evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Print confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    cr = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(cr)

if __name__ == "__main__":
    train(params['data'])