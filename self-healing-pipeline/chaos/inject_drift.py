import sys, os
sys.path.insert(0, os.getcwd())
import requests
import json
import time
import argparse
import numpy as np
from dotenv import load_dotenv
from chaos.monkey import SCENARIOS, load_clean_data

load_dotenv()

MONITOR_URL = "http://localhost:8000/score"

def inject_drift(scenario: str):
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}. Choose from {list(SCENARIOS.keys())}")
        return

    print(f"Starting Injection for Scenario: [{scenario.upper()}]...")
    
    # 1. Load clean data
    df = load_clean_data().sample(200)
    
    # 2. Corrupt data using the monkey functions
    print("Corrupting feature distributions...")
    df = SCENARIOS[scenario](df)
    
    # 3. Convert to JSON payload (replace NaN with None)
    df = df.replace({np.nan: None})
    payload = {
        "data": df.to_dict(orient="records")
    }
    
    # 4. Send to monitor and measure time
    try:
        print(f"Sending {len(df)} rows to {MONITOR_URL}...")
        start_time = time.time()
        
        r = requests.post(MONITOR_URL, json=payload, timeout=10)
        r.raise_for_status()
        
        end_time = time.time()
        ttd = end_time - start_time
        
        result = r.json()
        print(f"\nData Accepted by Monitor in {ttd:.3f} seconds (Time to Detect / TTD)")
        print(f"Monitor Response: {json.dumps(result, indent=2)}")
        
        print("\nWaiting 2 seconds for monitor to flush report to disk...")
        time.sleep(2)
        
        # Check the actual report written to disk
        report_path = "infra/latest_drift_report.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            print("\nReal-time Drift Report:")
            print(f"   - Max JS Divergence: {report.get('max_js_divergence', 0)}")
            print(f"   - Drift Detected:    {report.get('drift_detected', False)}")
            print(f"   - Null Counts:       {report.get('null_counts', {})}")
            
    except Exception as e:
        print(f"Failed to inject drift: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject chaos into the ML pipeline")
    parser.add_argument("scenario", nargs="?", default="drift", choices=list(SCENARIOS.keys()), help="Which scenario to run")
    args = parser.parse_args()
    
    inject_drift(args.scenario)
