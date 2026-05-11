import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import psycopg2
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Aether Heal — Pipeline Monitor",
    page_icon="🧠",
    layout="wide"
)

# ── Styles ────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0d1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .reasoning-box {
        background: #0d1117;
        border-left: 3px solid #00e5ff;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #c9d1d9;
        margin: 0.5rem 0;
        white-space: pre-wrap;
    }
    .tool-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .tool-retrain  { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950; }
    .tool-rollback { background:#3a1a1a; color:#f85149; border:1px solid #f85149; }
    .tool-clean    { background:#1a2a3a; color:#58a6ff; border:1px solid #58a6ff; }
    .tool-drift    { background:#2a2a1a; color:#e3b341; border:1px solid #e3b341; }
    .tool-logs     { background:#2a1a3a; color:#bc8cff; border:1px solid #bc8cff; }
    .status-ok     { color: #3fb950; font-weight: 700; }
    .status-warn   { color: #e3b341; font-weight: 700; }
    .status-fail   { color: #f85149; font-weight: 700; }
    .ts            { color: #8b949e; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

def fetch(query, params=None):
    try:
        from utils.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()

# ── Tool badge helper ─────────────────────────────────────────────
TOOL_CLASS = {
    "trigger_retraining": "tool-retrain",
    "rollback_model":     "tool-rollback",
    "run_data_cleaning":  "tool-clean",
    "get_drift_report":   "tool-drift",
    "explore_logs":       "tool-logs",
}

def tool_badge(tool: str) -> str:
    cls = TOOL_CLASS.get(tool, "tool-drift")
    return f'<span class="tool-badge {cls}">{tool}</span>'

# ── Header ────────────────────────────────────────────────────────
st.markdown("#Aether Heal")
st.markdown("**Self-Healing ML Pipeline** — Agent Reasoning Monitor")
st.divider()

# ── Top metrics ───────────────────────────────────────────────────
runs_df   = fetch("SELECT * FROM pipeline_runs ORDER BY triggered_at DESC LIMIT 100")
agents_df = fetch("SELECT * FROM agent_logs   ORDER BY created_at   DESC LIMIT 100")

col1, col2, col3, col4 = st.columns(4)
with col1:
    total = len(runs_df)
    st.metric("Total Pipeline Runs", total)
with col2:
    healed = len(agents_df["run_id"].unique()) if not agents_df.empty else 0
    st.metric("Healing Events", healed)
with col3:
    if not agents_df.empty:
        top = agents_df["tool_called"].value_counts().idxmax()
        st.metric("Most Used Tool", top)
    else:
        st.metric("Most Used Tool", "—")
with col4:
    if not runs_df.empty and "drift_score" in runs_df.columns:
        avg = runs_df["drift_score"].dropna().mean()
        st.metric("Avg Drift Score", f"{avg:.3f}" if avg == avg else "—")
    else:
        st.metric("Avg Drift Score", "—")

st.divider()

# ── Main panels ───────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown("### Agent Reasoning Log")
    st.caption("Every decision the agent made, in order.")

    if agents_df.empty:
        st.info("No agent runs recorded yet. Run `python agent/graph.py` to trigger the agent.")
    else:
        for _, row in agents_df.iterrows():
            with st.expander(
                f"Run `{row['run_id']}` — {row['tool_called']}  •  {str(row['created_at'])[:19]}",
                expanded=True
            ):
                st.markdown(tool_badge(row["tool_called"]), unsafe_allow_html=True)
                st.markdown("**Reasoning:**")
                st.markdown(
                    f'<div class="reasoning-box">{row["reasoning"]}</div>',
                    unsafe_allow_html=True
                )
                if row["outcome"]:
                    st.markdown("**Outcome:**")
                    outcome_text = str(row["outcome"])[:600]
                    color = "#3fb950" if "complete" in outcome_text.lower() or "success" in outcome_text.lower() else "#f85149"
                    st.markdown(
                        f'<div class="reasoning-box" style="border-left-color:{color}">{outcome_text}</div>',
                        unsafe_allow_html=True
                    )

with right:
    st.markdown("### Pipeline Run History")
    if runs_df.empty:
        st.info("No pipeline runs yet.")
    else:
        for _, row in runs_df.head(15).iterrows():
            status = row.get("status", "unknown")
            color_cls = "status-ok" if status == "success" else ("status-warn" if status == "healing" else "status-fail")
            drift = row.get("drift_score")
            drift_str = f"{drift:.3f}" if drift and drift == drift else "—"
            action = row.get("action_taken") or "—"
            ts = str(row.get("triggered_at", ""))[:19]
            st.markdown(f"""
            <div class="metric-card">
                <span class="ts">{ts}</span><br>
                <span class="{color_cls}">● {status.upper()}</span>
                &nbsp;&nbsp;<code>{row.get('run_id','')[:8]}</code><br>
                <small>Drift: <b>{drift_str}</b> &nbsp;|&nbsp; Action: <b>{action[:40]}</b></small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Tool Usage")
    if not agents_df.empty:
        counts = agents_df["tool_called"].value_counts().reset_index()
        counts.columns = ["Tool", "Count"]
        st.bar_chart(counts.set_index("Tool"))

# ── Auto-refresh ──────────────────────────────────────────────────
st.divider()
col_r, col_s = st.columns([3, 1])
with col_s:
    if st.button("Refresh"):
        st.cache_resource.clear()
        st.rerun()
with col_r:
    st.caption("Data updates on each refresh. Run the agent in a separate terminal to see new entries.")