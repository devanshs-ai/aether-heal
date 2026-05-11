# AetherHeal: Self-Healing ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Resilience: High](https://img.shields.io/badge/resilience-high-brightgreen.svg)]()

**AetherHeal** is an advanced, autonomous machine learning pipeline designed for high-availability environments. It integrates real-time monitoring with a Large Language Model (LLM) powered agent to automatically diagnose and remediate issues like data drift, model degradation, and system failures.

---

## Key Features

- **Autonomous Diagnostic Agent**: Powered by `LangGraph` and `Groq` (Llama 3.3), the agent analyzes pipeline logs and monitoring metrics to make high-level decisions.
- **Real-time Drift Detection**: Integrated `Evidently AI` monitoring to catch concept and data drift before they impact business metrics.
- **Automated Remediation**:
    - **Self-Cleaning**: Detects and fixes data corruption issues.
    - **Auto-Retraining**: Triggers retraining cycles when significant drift is confirmed.
    - **Instant Rollback**: Automatically rolls back to the "Last Known Good" (LKG) model version if performance drops.
- **Chaos Engineering**: Built-in failure injection modules to stress-test the system's self-healing capabilities.
- **Observability Dashboard**: A premium `Streamlit` UI for tracking model health, agent reasoning, and pipeline status.

---

## Project Structure

```text
aether-heal/
├── self-healing-pipeline/
│   ├── agent/           # LangGraph Agent (Reasoning & Tools)
│   ├── chaos/           # Chaos Engineering (Failure Injection)
│   ├── model/           # ML Training & Registry Logic
│   ├── monitor/         # Prometheus & FastAPI Monitoring Service
│   ├── pipeline/        # Prefect Orchestration Logic
│   ├── ui/              # Streamlit Dashboard
│   ├── infra/           # Infrastructure logs & configurations
│   └── requirements.txt # Project dependencies
├── venv/                # Local virtual environment
├── .gitignore           # Git exclusion rules
└── README.md            # You are here
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/aether-heal.git
cd aether-heal
```

### 2. Configure Environment
Create a `.env` file in the `self-healing-pipeline/` directory (refer to `.env.example` if available):
```env
GROQ_API_KEY=your_groq_api_key
POSTGRES_DB=pipeline_db
POSTGRES_USER=pipeline_user
POSTGRES_PASSWORD=pipeline_pass
```

### 3. Install Dependencies
```bash
# Activate your venv (Windows)
source venv/Scripts/activate

# Install requirements
pip install -r self-healing-pipeline/requirements.txt
```

### 4. Run the Infrastructure
Use Docker Compose to spin up the required services (PostgreSQL, MLflow, MinIO, Prometheus):
```bash
docker-compose up -d
```

### 5. Launch the Dashboard
```bash
streamlit run self-healing-pipeline/ui/app.py
```

---

## How the Agent Works

When the monitoring service detects an anomaly (e.g., accuracy drop or JS-divergence spike), it triggers the **Aether Agent**.

1. **Observe**: Agent pulls the latest drift reports and log snippets.
2. **Reason**: Uses an LLM to decide if the issue is transient noise, data corruption, or genuine concept drift.
3. **Act**: Calls the appropriate tool (`trigger_retraining`, `run_data_cleaning`, or `rollback_model`).
4. **Verify**: Ensures the pipeline returns to a "Healthy" state.

---

## Resilience & Chaos Testing
The project includes a `chaos/` module that allows you to manually inject:
- **Null Value Spikes**: Tests data cleaning resilience.
- **Feature Distribution Shifts**: Tests drift detection and auto-retraining.
- **Process Termination**: Tests pipeline orchestration recovery.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
