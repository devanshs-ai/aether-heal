import os
import json
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from monitor.middleware import DriftMiddleware
from monitor.metrics import registry

load_dotenv()

# Global state for cached model
MODEL_CACHE = {
    "model": None,
    "version": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    from model.registry import get_lkg_model
    try:
        model, version = get_lkg_model()
        MODEL_CACHE["model"] = model
        MODEL_CACHE["version"] = version
        logging.info(f"Loaded LKG model version {version} on startup.")
    except Exception as e:
        logging.error(f"Failed to load model on startup: {e}")
    yield
    # Cleanup on shutdown if needed

app = FastAPI(title="Pipeline Monitor", lifespan=lifespan)
app.add_middleware(DriftMiddleware)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score_batch(payload: dict):
    """Accepts a batch, scores it, middleware intercepts for drift detection."""
    try:
        if MODEL_CACHE["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")
            
        df = pd.DataFrame(payload["data"])
        model = MODEL_CACHE["model"]
        version = MODEL_CACHE["version"]
        
        preds = model.predict(df.drop(columns=["Class"], errors="ignore"))
        return {
            "model_version": version,
            "scored_rows": len(preds),
            "fraud_rate": float(preds.mean())
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error scoring batch")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_model():
    """Endpoint for the agent to trigger a model reload into memory."""
    from model.registry import get_lkg_model
    try:
        model, version = get_lkg_model()
        MODEL_CACHE["model"] = model
        MODEL_CACHE["version"] = version
        return {"status": "success", "message": f"Reloaded version {version}"}
    except Exception as e:
        logging.exception("Error reloading model")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/drift/status")
def drift_status():
    """Latest drift report — agent polls this."""
    report_path = "infra/latest_drift_report.json"
    if not os.path.exists(report_path):
        return {"drift_detected": False, "message": "No report yet"}
    with open(report_path) as f:
        return json.load(f)