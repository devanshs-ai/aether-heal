import sys, os
sys.path.insert(0, os.getcwd())
import requests
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv

load_dotenv()

MONITOR_URL = "http://localhost:8000/score"

def inject_drift():
    print("Starting Drift Injection...")
    
    # 1. Load some real data
    from utils.data import load_and_preprocess_data
    df = load_and_preprocess_data().sample(200)
    
    # 2. Artificially "Drifting" the data so that we can get a bit different statistical distribution for this data
    # We'll shift MonthlyCharges and TotalCharges significantly
    print("Corrupting feature distributions...")
    df['MonthlyCharges'] = df['MonthlyCharges'] * np.random.uniform(1.5, 2.5, size=len(df))
    df['TotalCharges'] = df['TotalCharges'] * 0.5
    
    # 3. Converting to JSON payload
    payload = {
        "data": df.to_dict(orient="records")
    }
    
    # 4. Sending to monitor
    try:
        print(f"Sending {len(df)} rows to {MONITOR_URL}...")
        r = requests.post(MONITOR_URL, json=payload, timeout=10)
        r.raise_for_status()
        
        result = r.json()
        print("\nData Accepted by Monitor")
        print(f"Monitor Response: {json.dumps(result, indent=2)}")
        
        print("\nWaiting 2 seconds for monitor to write report...")
        time.sleep(2)
        
        # Check the actual report written to disk
        report_path = "infra/latest_drift_report.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            print("\n🔍 Real-time Drift Report:")
            print(f"   - Max JS Divergence: {report['max_js_divergence']}")
            print(f"   - Drift Detected:    {report['drift_detected']}")
            
    except Exception as e:
        print(f"Failed to inject drift: {e}")

if __name__ == "__main__":
    import json
    inject_drift()
