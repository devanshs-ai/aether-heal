import sys, os
sys.path.insert(0, os.getcwd())

import json
import subprocess
import psycopg2
import requests
import mlflow
import optuna
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def log_agent_action(run_id: str, reasoning: str, tool: str, outcome: str):
    try:
        from utils.db import engine
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO agent_logs (run_id, reasoning, tool_called, outcome)
                        VALUES (:run_id, :reasoning, :tool, :outcome)"""),
                {"run_id": run_id, "reasoning": reasoning, "tool": tool, "outcome": outcome}
            )
    except Exception as e:
        print(f"[Warning] Could not log to database: {e}")

# ── Tool 1: Log Explorer ──────────────────────────────────────────
def explore_logs(keyword: str = "ERROR", tail: int = 50) -> str:
    """Grep recent logs for a keyword."""
    log_file = "infra/pipeline.log"
    if not os.path.exists(log_file):
        return "No log file found. Pipeline may not have written logs yet."
    with open(log_file) as f:
        lines = f.readlines()
    matched = [l.strip() for l in lines if keyword.lower() in l.lower()]
    return "\n".join(matched[-tail:]) or f"No lines matching '{keyword}' found."

# ── Tool 2: Drift Detector ────────────────────────────────────────
def get_drift_report() -> dict:
    """Fetch the latest drift report from the monitor."""
    try:
        r = requests.get("http://localhost:8000/drift/status", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e), "drift_detected": False}

# ── Tool 3: Circuit Breaker ───────────────────────────────────────
def rollback_model() -> str:
    """Roll back to Last Known Good model version."""
    from model.registry import rollback_to_lkg
    version = rollback_to_lkg()
    return f"Rolled back to LKG model version {version}"

# ── Tool 4: Data Cleaner ──────────────────────────────────────────
def run_data_cleaning(scenario: str = "nulls") -> str:
    """Trigger chaos monkey in reverse — regenerate a clean batch."""
    from utils.data import load_and_preprocess_data
    df = load_and_preprocess_data()
    out = "infra/cleaned_batch.csv"
    df.to_csv(out, index=False)
    return f"Cleaned batch saved to {out} ({len(df)} rows)"

# ── Tool 5: Auto Retrainer ────────────────────────────────────────
def trigger_retraining() -> str:
    """Retrain model on latest clean data and register new version."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "model/train.py"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if result.returncode == 0:
        return f"Retraining complete.\n{result.stdout.strip()}"
    return f"Retraining failed.\n{result.stderr.strip()}"

TOOLS = {
    "explore_logs":     explore_logs,
    "get_drift_report": get_drift_report,
    "rollback_model":   rollback_model,
    "run_data_cleaning": run_data_cleaning,
    "trigger_retraining": trigger_retraining,
}