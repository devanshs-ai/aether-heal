import os
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(uri)
mlflow.set_experiment("churn-baseline")

def load_data():
    from utils.data import load_and_preprocess_data
    return load_and_preprocess_data()


def train():
    df = load_data()
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save reference distribution for drift detection later
    X_train.describe().to_csv("infra/reference_stats.csv")

    import optuna

    scale_pos_weight = float(len(y_train) - sum(y_train)) / sum(y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        
        # Use early stopping
        model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
        
        preds = model.predict(X_test)
        return roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = 42
    best_params["eval_metric"] = "logloss"

    print(f"Best params: {best_params}")

    with mlflow.start_run(run_name="lkg-optimized") as run:
        model = XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

        mlflow.log_params(best_params)
        mlflow.log_metrics({"accuracy": acc, "auc": auc})
        mlflow.xgboost.log_model(model, "model", registered_model_name="churn-model")

        # Tag as Last Known Good
        client = mlflow.tracking.MlflowClient()
        mv = client.search_model_versions("name='churn-model'")[0]
        client.set_model_version_tag("churn-model", mv.version, "stage", "LKG")

        print(f"[Success] Trained | ACC: {acc:.4f} | AUC: {auc:.4f} | Run ID: {run.info.run_id}")
        return run.info.run_id

if __name__ == "__main__":
    train()