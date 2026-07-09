use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry, TextEncoder,
};
use std::sync::LazyLock;

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

pub static REQUEST_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::with_opts(Opts::new(
        "gguf_switchboard_requests_total",
        "Total number of HTTP requests processed",
    ))
    .expect("failed to create REQUEST_TOTAL metric")
});

pub static INFERENCE_LATENCY: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(HistogramOpts::new(
        "gguf_switchboard_inference_latency_seconds",
        "End-to-end inference latency in seconds",
    ))
    .expect("failed to create INFERENCE_LATENCY metric")
});

pub static MODEL_LOAD_LATENCY: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(HistogramOpts::new(
        "gguf_switchboard_model_load_latency_seconds",
        "Time to load a model from cold start in seconds",
    ))
    .expect("failed to create MODEL_LOAD_LATENCY metric")
});

pub static ACTIVE_REQUESTS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_active_requests",
        "Number of requests currently in-flight",
    ))
    .expect("failed to create ACTIVE_REQUESTS metric")
});

pub static LOADED_MODEL: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_loaded_model",
        "Whether a model is currently loaded (1 = yes, 0 = no)",
    ))
    .expect("failed to create LOADED_MODEL metric")
});

pub static BACKEND_HEALTH: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_backend_healthy",
        "Whether the backend is healthy (1 = yes, 0 = no)",
    ))
    .expect("failed to create BACKEND_HEALTH metric")
});

pub static STREAMING_REQUESTS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_streaming_requests",
        "Number of streaming requests currently active",
    ))
    .expect("failed to create STREAMING_REQUESTS metric")
});

pub static MEMORY_USAGE_PERCENT: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_memory_usage_percent",
        "Current system memory usage as a percentage (0-100)",
    ))
    .expect("failed to create MEMORY_USAGE_PERCENT metric")
});

/// Register all metrics with the custom registry.
pub fn register_all() {
    let r = &*REGISTRY;
    r.register(Box::new(REQUEST_TOTAL.clone()))
        .expect("register REQUEST_TOTAL");
    r.register(Box::new(INFERENCE_LATENCY.clone()))
        .expect("register INFERENCE_LATENCY");
    r.register(Box::new(MODEL_LOAD_LATENCY.clone()))
        .expect("register MODEL_LOAD_LATENCY");
    r.register(Box::new(ACTIVE_REQUESTS.clone()))
        .expect("register ACTIVE_REQUESTS");
    r.register(Box::new(LOADED_MODEL.clone()))
        .expect("register LOADED_MODEL");
    r.register(Box::new(BACKEND_HEALTH.clone()))
        .expect("register BACKEND_HEALTH");
    r.register(Box::new(STREAMING_REQUESTS.clone()))
        .expect("register STREAMING_REQUESTS");
    r.register(Box::new(MEMORY_USAGE_PERCENT.clone()))
        .expect("register MEMORY_USAGE_PERCENT");
}

/// Gather all metrics as a Prometheus text-format string.
pub fn gather() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .expect("failed to encode metrics");
    String::from_utf8(buffer).expect("metrics output is not valid UTF-8")
}
