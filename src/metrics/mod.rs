use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry, TextEncoder,
};
use std::sync::LazyLock;

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// Buckets for per-request inference latency. The Prometheus defaults top out at 10s,
/// which is shorter than a typical multi-hundred-token completion on a large local
/// model, so everything used to land in `+Inf`.
const INFERENCE_BUCKETS: &[f64] = &[
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

/// Buckets for model load / switch phases. GGUF loads are measured in seconds to
/// minutes (disk → page cache → VRAM), so the default `0.005..10` buckets are useless.
#[rustfmt::skip]
const LOAD_BUCKETS: &[f64] = &[
    0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0,
    300.0, 600.0,
];

/// Phase label values used by [`MODEL_SWITCH_PHASE_SECONDS`].
pub mod phase {
    /// Waiting for in-flight requests on the previous model to finish.
    pub const DRAIN: &str = "drain";
    /// SIGTERM/wait of the previous `llama-server` (frees VRAM).
    pub const UNLOAD_PREVIOUS: &str = "unload_previous";
    /// `nvidia-smi` probe, GGUF stat, profile cache lookup, auto_ngl — everything before spawn.
    pub const PLAN: &str = "plan";
    /// From `llama-server` spawn until `/health` returns 200 (one attempt).
    pub const SPAWN_TO_HEALTHY: &str = "spawn_to_healthy";
    /// Re-loading the previous model after a failed switch.
    pub const ROLLBACK: &str = "rollback";
}

/// Result label values.
pub mod result {
    pub const OK: &str = "ok";
    pub const ERROR: &str = "error";
    /// Load attempt failed with an OOM classification and the scheduler retried
    /// with a smaller context / fewer GPU layers.
    pub const OOM_RETRY: &str = "oom_retry";
    /// Load attempt timed out waiting for `/health`.
    pub const TIMEOUT: &str = "timeout";
}

pub static REQUEST_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::with_opts(Opts::new(
        "gguf_switchboard_requests_total",
        "Total number of HTTP requests processed",
    ))
    .expect("failed to create REQUEST_TOTAL metric")
});

/// Inference-only latency (measured after the model is resident; model load /
/// switch time is reported separately by `REQUEST_MODEL_WAIT_SECONDS`).
pub static INFERENCE_LATENCY: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "gguf_switchboard_inference_latency_seconds",
            "Inference latency in seconds, excluding any model load/switch wait \
             (see gguf_switchboard_request_model_wait_seconds)",
        )
        .buckets(INFERENCE_BUCKETS.to_vec()),
    )
    .expect("failed to create INFERENCE_LATENCY metric")
});

/// Observes `INFERENCE_LATENCY` when dropped. Start it *after* `ensure_loaded`
/// and, for streaming responses, move it into the response stream's guard list so
/// the observation covers the whole generation rather than just time-to-headers.
pub struct InferenceTimer {
    started: std::time::Instant,
}

impl InferenceTimer {
    pub fn start() -> Self {
        Self {
            started: std::time::Instant::now(),
        }
    }
}

impl Drop for InferenceTimer {
    fn drop(&mut self) {
        INFERENCE_LATENCY.observe(self.started.elapsed().as_secs_f64());
    }
}

/// Time a request spent inside `ensure_loaded` when the requested model was not
/// already resident (includes waiting for another request's in-progress load).
pub static REQUEST_MODEL_WAIT_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "gguf_switchboard_request_model_wait_seconds",
            "Seconds a request waited for its model to become resident (only observed \
             when a load/switch was needed)",
        )
        .buckets(LOAD_BUCKETS.to_vec()),
        &["model"],
    )
    .expect("failed to create REQUEST_MODEL_WAIT_SECONDS metric")
});

/// Requests that found their model already loaded vs. had to wait for a load.
pub static REQUEST_MODEL_HIT_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "gguf_switchboard_request_model_hit_total",
            "Requests by whether the requested model was already resident (result=hit|miss)",
        ),
        &["model", "result"],
    )
    .expect("failed to create REQUEST_MODEL_HIT_TOTAL metric")
});

/// Unlabelled cold-start histogram, kept for dashboards built against earlier
/// releases. Prefer `MODEL_LOAD_SECONDS` (per-model) for new dashboards.
pub static MODEL_LOAD_LATENCY: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "gguf_switchboard_model_load_latency_seconds",
            "Time from llama-server spawn until healthy for successful loads, in seconds \
             (all models)",
        )
        .buckets(LOAD_BUCKETS.to_vec()),
    )
    .expect("failed to create MODEL_LOAD_LATENCY metric")
});

/// Per-model, per-result load attempt duration (spawn → healthy / failure).
pub static MODEL_LOAD_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "gguf_switchboard_model_load_seconds",
            "Duration of a single llama-server load attempt (spawn until healthy or failure), \
             per model and result",
        )
        .buckets(LOAD_BUCKETS.to_vec()),
        &["model", "result"],
    )
    .expect("failed to create MODEL_LOAD_SECONDS metric")
});

pub static MODEL_LOAD_ATTEMPTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "gguf_switchboard_model_load_attempts_total",
            "llama-server load attempts per model and result (ok|oom_retry|timeout|error)",
        ),
        &["model", "result"],
    )
    .expect("failed to create MODEL_LOAD_ATTEMPTS_TOTAL metric")
});

/// Most recent successful load time per model — handy as a single-stat panel.
pub static MODEL_LAST_LOAD_SECONDS: LazyLock<prometheus::GaugeVec> = LazyLock::new(|| {
    prometheus::GaugeVec::new(
        Opts::new(
            "gguf_switchboard_model_last_load_seconds",
            "Spawn-to-healthy duration of the most recent successful load, per model",
        ),
        &["model"],
    )
    .expect("failed to create MODEL_LAST_LOAD_SECONDS metric")
});

/// End-to-end cost of making `model` resident as seen by the scheduler: drain +
/// unload of the previous model + planning + all load attempts (+ rollback on error).
pub static MODEL_SWITCH_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "gguf_switchboard_model_switch_seconds",
            "Total time to make a model resident (drain + unload previous + plan + load \
             attempts), per target model and result",
        )
        .buckets(LOAD_BUCKETS.to_vec()),
        &["model", "result"],
    )
    .expect("failed to create MODEL_SWITCH_SECONDS metric")
});

/// Breakdown of a switch into phases so you can see *where* the time goes.
pub static MODEL_SWITCH_PHASE_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "gguf_switchboard_model_switch_phase_seconds",
            "Per-phase duration of a model switch \
             (phase=drain|unload_previous|plan|spawn_to_healthy|rollback), per target model",
        )
        .buckets(LOAD_BUCKETS.to_vec()),
        &["model", "phase"],
    )
    .expect("failed to create MODEL_SWITCH_PHASE_SECONDS metric")
});

pub static MODEL_SWITCHES_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "gguf_switchboard_model_switches_total",
            "Model residency changes by previous model (\"none\" when the slot was empty), \
             target model, trigger and result",
        ),
        &["from", "to", "trigger", "result"],
    )
    .expect("failed to create MODEL_SWITCHES_TOTAL metric")
});

pub static MODEL_UNLOADS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "gguf_switchboard_model_unloads_total",
            "Model unloads per model and reason \
             (switch|idle_priority|memory_pressure|unhealthy|registry_refresh|shutdown)",
        ),
        &["model", "reason"],
    )
    .expect("failed to create MODEL_UNLOADS_TOTAL metric")
});

pub static ACTIVE_REQUESTS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_active_requests",
        "Number of requests currently in-flight",
    ))
    .expect("failed to create ACTIVE_REQUESTS metric")
});

pub static EMBEDDING_QUEUE_DEPTH: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "gguf_switchboard_embedding_queue_depth",
            "Embedding requests waiting for a backend permit",
        ),
        &["model"],
    )
    .expect("failed to create EMBEDDING_QUEUE_DEPTH")
});

pub static EMBEDDING_QUEUE_REJECTED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "gguf_switchboard_embedding_queue_rejected_total",
            "Embedding requests rejected after queue timeout",
        ),
        &["model"],
    )
    .expect("failed to create EMBEDDING_QUEUE_REJECTED_TOTAL")
});

pub static EMBEDDING_QUEUE_WAIT_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "gguf_switchboard_embedding_queue_wait_seconds",
            "Time embedding requests wait for a backend permit",
        )
        .buckets(INFERENCE_BUCKETS.to_vec()),
        &["model"],
    )
    .expect("failed to create EMBEDDING_QUEUE_WAIT_SECONDS")
});

pub static LOADED_MODEL: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::with_opts(Opts::new(
        "gguf_switchboard_loaded_model",
        "Whether a model is currently loaded (1 = yes, 0 = no)",
    ))
    .expect("failed to create LOADED_MODEL metric")
});

/// `1` for the model id that currently occupies the slot, `0` for others that
/// have been resident at some point during this process lifetime.
pub static LOADED_MODEL_INFO: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "gguf_switchboard_loaded_model_info",
            "Which model currently occupies the slot (1 for the resident model, 0 otherwise)",
        ),
        &["model"],
    )
    .expect("failed to create LOADED_MODEL_INFO metric")
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

/// Model ids that have had a `LOADED_MODEL_INFO` series created, so we can zero
/// them when the slot changes hands.
static SEEN_LOADED_MODELS: LazyLock<parking_lot::Mutex<std::collections::HashSet<String>>> =
    LazyLock::new(|| parking_lot::Mutex::new(std::collections::HashSet::new()));

/// Point `LOADED_MODEL_INFO` at `model` (or clear it when `None`).
pub fn set_loaded_model_info(model: Option<&str>) {
    let mut seen = SEEN_LOADED_MODELS.lock();
    for id in seen.iter() {
        LOADED_MODEL_INFO.with_label_values(&[id.as_str()]).set(0);
    }
    if let Some(model) = model {
        seen.insert(model.to_string());
        LOADED_MODEL_INFO.with_label_values(&[model]).set(1);
    }
}

/// Record a single load attempt outcome for `model`.
pub fn record_load_attempt(model: &str, outcome: &str, secs: f64) {
    MODEL_LOAD_ATTEMPTS_TOTAL
        .with_label_values(&[model, outcome])
        .inc();
    MODEL_LOAD_SECONDS
        .with_label_values(&[model, outcome])
        .observe(secs);
    MODEL_SWITCH_PHASE_SECONDS
        .with_label_values(&[model, phase::SPAWN_TO_HEALTHY])
        .observe(secs);
    if outcome == result::OK {
        MODEL_LOAD_LATENCY.observe(secs);
        MODEL_LAST_LOAD_SECONDS
            .with_label_values(&[model])
            .set(secs);
    }
}

pub fn record_phase(model: &str, phase_name: &str, secs: f64) {
    MODEL_SWITCH_PHASE_SECONDS
        .with_label_values(&[model, phase_name])
        .observe(secs);
}

/// Register all metrics with the custom registry. Idempotent, so it is safe to
/// call from both `main` and tests that live in the same process.
pub fn register_all() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(register_all_inner);
}

fn register_all_inner() {
    let r = &*REGISTRY;
    r.register(Box::new(REQUEST_TOTAL.clone()))
        .expect("register REQUEST_TOTAL");
    r.register(Box::new(INFERENCE_LATENCY.clone()))
        .expect("register INFERENCE_LATENCY");
    r.register(Box::new(REQUEST_MODEL_WAIT_SECONDS.clone()))
        .expect("register REQUEST_MODEL_WAIT_SECONDS");
    r.register(Box::new(REQUEST_MODEL_HIT_TOTAL.clone()))
        .expect("register REQUEST_MODEL_HIT_TOTAL");
    r.register(Box::new(MODEL_LOAD_LATENCY.clone()))
        .expect("register MODEL_LOAD_LATENCY");
    r.register(Box::new(MODEL_LOAD_SECONDS.clone()))
        .expect("register MODEL_LOAD_SECONDS");
    r.register(Box::new(MODEL_LOAD_ATTEMPTS_TOTAL.clone()))
        .expect("register MODEL_LOAD_ATTEMPTS_TOTAL");
    r.register(Box::new(MODEL_LAST_LOAD_SECONDS.clone()))
        .expect("register MODEL_LAST_LOAD_SECONDS");
    r.register(Box::new(MODEL_SWITCH_SECONDS.clone()))
        .expect("register MODEL_SWITCH_SECONDS");
    r.register(Box::new(MODEL_SWITCH_PHASE_SECONDS.clone()))
        .expect("register MODEL_SWITCH_PHASE_SECONDS");
    r.register(Box::new(MODEL_SWITCHES_TOTAL.clone()))
        .expect("register MODEL_SWITCHES_TOTAL");
    r.register(Box::new(MODEL_UNLOADS_TOTAL.clone()))
        .expect("register MODEL_UNLOADS_TOTAL");
    r.register(Box::new(ACTIVE_REQUESTS.clone()))
        .expect("register ACTIVE_REQUESTS");
    r.register(Box::new(EMBEDDING_QUEUE_DEPTH.clone()))
        .expect("register EMBEDDING_QUEUE_DEPTH");
    r.register(Box::new(EMBEDDING_QUEUE_REJECTED_TOTAL.clone()))
        .expect("register EMBEDDING_QUEUE_REJECTED_TOTAL");
    r.register(Box::new(EMBEDDING_QUEUE_WAIT_SECONDS.clone()))
        .expect("register EMBEDDING_QUEUE_WAIT_SECONDS");
    r.register(Box::new(LOADED_MODEL.clone()))
        .expect("register LOADED_MODEL");
    r.register(Box::new(LOADED_MODEL_INFO.clone()))
        .expect("register LOADED_MODEL_INFO");
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
