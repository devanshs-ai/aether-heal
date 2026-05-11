import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def get_lkg_model():
    """Load the Last Known Good model version."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='churn-model'")
    lkg = next((v for v in versions if v.tags.get("stage") == "LKG"), versions[0])
    model = mlflow.xgboost.load_model(f"models:/churn-model/{lkg.version}")
    return model, lkg.version

def rollback_to_lkg():
    """Return LKG version number for the agent's circuit breaker."""
    _, version = get_lkg_model()
    print(f"✓ Rolled back to model version {version}")
    return version