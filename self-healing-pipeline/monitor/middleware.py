import os
import json
import time
import logging
import datetime
import pandas as pd
import numpy as np
from scipy.stats import entropy
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool
from monitor.metrics import DRIFT_SCORE, REQUEST_COUNT, REQUEST_LATENCY

REFERENCE_PATH = "infra/reference_stats.csv"
DRIFT_THRESHOLD = 0.1
REPORT_PATH = "infra/latest_drift_report.json"

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.array(p, dtype=float) + 1e-10
    q = np.array(q, dtype=float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))

REF_STATS_CACHE = None

def get_ref_stats():
    global REF_STATS_CACHE
    if REF_STATS_CACHE is None:
        if os.path.exists(REFERENCE_PATH):
            REF_STATS_CACHE = pd.read_csv(REFERENCE_PATH, index_col=0)
    return REF_STATS_CACHE

def compute_drift(df: pd.DataFrame) -> dict:
    ref = get_ref_stats()
    if ref is None:
        return {"drift_detected": False, "message": "No reference stats found"}
    watch_cols = ["Amount", "V1", "V2"]
    results = {}
    max_score = 0.0

    for col in watch_cols:
        if col not in df.columns or col not in ref.columns:
            continue
        curr_vals = df[col].dropna().values
        # Bin both into 20 buckets for JS divergence
        combined_min = min(curr_vals.min(), ref.loc["min", col])
        combined_max = max(curr_vals.max(), ref.loc["max", col])
        bins = np.linspace(combined_min, combined_max, 21)
        p, _ = np.histogram(curr_vals, bins=bins)
        q, _ = np.histogram(
            np.random.normal(ref.loc["mean", col], ref.loc["std", col], 1000),
            bins=bins
        )
        score = jensen_shannon_divergence(p, q)
        results[col] = round(score, 4)
        max_score = max(max_score, score)

    drift_detected = max_score > DRIFT_THRESHOLD
    report = {
        "drift_detected": drift_detected,
        "max_js_divergence": round(max_score, 4),
        "threshold": DRIFT_THRESHOLD,
        "feature_scores": results,
        "null_counts": df[watch_cols].isnull().sum().to_dict(),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    DRIFT_SCORE.set(max_score)
    return report

class DriftMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()

        # Read and cache body BEFORE passing to route handler
        body_bytes = await request.body()

        response = await call_next(request)
        latency = time.time() - start

        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(latency)

        if request.url.path == "/score":
            try:
                payload = json.loads(body_bytes)
                df = pd.DataFrame(payload.get("data", []))
                if not df.empty:
                    # Offload CPU-bound and blocking I/O to threadpool
                    report = await run_in_threadpool(compute_drift, df)
                    if report.get("drift_detected"):
                        logging.warning(f"[DRIFT ALERT] JS={report['max_js_divergence']} > {DRIFT_THRESHOLD}")
            except Exception as e:
                logging.exception("Drift check skipped")

        return response