import os
import sys
import json
from mcp.server.fastmcp import FastMCP

# Add parent directory to path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import get_db_session
from sqlalchemy import text

# Initialize FastMCP Server
mcp = FastMCP("AetherHeal")

@mcp.tool()
def get_pipeline_status() -> str:
    """
    Fetches the real-time health and Jensen-Shannon divergence of the production model 
    from the latest drift report.
    """
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "infra", "latest_drift_report.json")
    if not os.path.exists(report_path):
        return "No drift report found. The pipeline may not have processed any data yet."
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    return json.dumps(report, indent=2)

@mcp.tool()
def inject_chaos_drift(scenario: str) -> str:
    """
    Triggers data corruption on demand for chaos testing.
    Valid scenarios: 'drift', 'nulls', 'outliers'.
    """
    if scenario not in ["drift", "nulls", "outliers"]:
        return "Invalid scenario. Choose from: 'drift', 'nulls', 'outliers'."
        
    try:
        # Import dynamically to avoid loading everything on startup
        from chaos.inject_drift import inject_drift
        inject_drift(scenario)
        return f"Successfully injected chaos scenario: {scenario}"
    except Exception as e:
        return f"Failed to inject chaos: {str(e)}"

@mcp.tool()
def trigger_agent_healing() -> str:
    """
    Kicks off the LangGraph agent to diagnose anomalies and heal the system.
    """
    try:
        from agent.graph import run_agent
        run_agent()
        return "Agent healing cycle completed successfully. Check logs for remediation details."
    except Exception as e:
        return f"Failed to run agent: {str(e)}"

@mcp.tool()
def get_model_metrics() -> str:
    """
    Queries the database to return the latest pipeline execution metrics.
    """
    try:
        # Fetch the latest pipeline run
        db_generator = get_db_session()
        db = next(db_generator)
        result = db.execute(text("SELECT * FROM pipeline_runs ORDER BY triggered_at DESC LIMIT 1")).fetchone()
        db.close()
        
        if not result:
            return "No pipeline runs found in the database."
            
        metrics = {
            "run_id": result.run_id,
            "accuracy": float(result.accuracy) if result.accuracy else None,
            "auc": float(result.auc) if result.auc else None,
            "status": result.status,
            "triggered_at": str(result.triggered_at)
        }
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Error fetching metrics: {str(e)}"

@mcp.tool()
def fetch_recent_logs(limit: int = 5) -> str:
    """
    Pulls recent agent reasoning logs from the Postgres database.
    """
    try:
        db_generator = get_db_session()
        db = next(db_generator)
        results = db.execute(text(f"SELECT tool_called, reasoning, status, created_at FROM agent_logs ORDER BY created_at DESC LIMIT {limit}")).fetchall()
        db.close()
        
        if not results:
            return "No recent agent logs found."
            
        logs = []
        for r in results:
            logs.append({
                "tool_called": r.tool_called,
                "reasoning": r.reasoning,
                "status": r.status,
                "created_at": str(r.created_at)
            })
        return json.dumps(logs, indent=2)
    except Exception as e:
        return f"Error fetching logs: {str(e)}"

if __name__ == "__main__":
    mcp.run()
