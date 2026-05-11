CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,          -- success | failed | healing
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    drift_score FLOAT,
    action_taken TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    tool_called TEXT,
    outcome TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_agent_logs_run_id ON agent_logs(run_id);