import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===== LOAD DATA =====
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MLflow =====
mlflow.set_experiment("heart_experiment_fix")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=505,
        max_depth=37
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", 505)
    mlflow.log_param("max_depth", 37)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)
    