from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "pipeline_requests_total",
    "Total scored batches",
    registry=registry
)
REQUEST_LATENCY = Histogram(
    "pipeline_request_latency_seconds",
    "Scoring latency",
    registry=registry
)
DRIFT_SCORE = Gauge(
    "pipeline_drift_score",
    "Latest JS divergence score",
    registry=registry
)