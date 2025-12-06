from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'app_request_count',
    'Total request count',
    ['method', 'endpoint', 'status']
)

ERROR_COUNT = Counter(
    'app_error_count',
    'Total error count',
    ['endpoint', 'error_type']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)
