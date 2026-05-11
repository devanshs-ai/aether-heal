import os
import uuid
import pandas as pd
import psycopg2
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv
from model.registry import get_lkg_model

load_dotenv()

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def log_run(run_id, status, drift_score=None, action=None, model_version=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO pipeline_runs (run_id, status, drift_score, action_taken, model_version)
           VALUES (%s, %s, %s, %s, %s)""",
        (run_id, status, drift_score, action, model_version)
    )
    conn.commit()
    cur.close()
    conn.close()

@task(retries=2, retry_delay_seconds=5)
def ingest(run_id: str) -> pd.DataFrame:
    logger = get_run_logger()
    from utils.data import load_and_preprocess_data
    df = load_and_preprocess_data()
    logger.info(f"[{run_id}] Ingested & Preprocessed {len(df)} rows")
    return df

@task
def preprocess(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    # Preprocessing is now handled in the data loader, but we keep the task for graph structure.
    return df

@task
def score(df: pd.DataFrame, run_id: str) -> dict:
    logger = get_run_logger()
    model, version = get_lkg_model()
    X = df.drop(columns=["Churn"])
    preds = model.predict(X)
    result = {"model_version": version, "scored_rows": len(preds), "churn_rate": float(preds.mean())}
    logger.info(f"[{run_id}] Scored | churn_rate={result['churn_rate']:.3f} | version={version}")
    return result

@flow(name="churn-pipeline", log_prints=True)
def churn_pipeline():
    run_id = str(uuid.uuid4())[:8]
    try:
        df_raw   = ingest(run_id)
        df_clean = preprocess(df_raw, run_id)
        result   = score(df_clean, run_id)
        log_run(run_id, "success", model_version=result["model_version"])
        print(f"✓ Pipeline complete | run={run_id} | {result}")
    except Exception as e:
        log_run(run_id, "failed", action=str(e))
        raise

if __name__ == "__main__":
    churn_pipeline()