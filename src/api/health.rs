use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use utoipa::ToSchema;

use crate::state::AppState;

/// Per-GPU telemetry sampled from `nvidia-smi` (cached briefly server-side).
#[derive(Serialize, ToSchema)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    /// Compute / core utilization ("load"), 0-100.
    pub gpu_utilization_pct: u32,
    /// Memory-controller bandwidth utilization, 0-100.
    pub memory_utilization_pct: u32,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    /// `memory_used_mb / memory_total_mb`, rounded, 0-100.
    pub memory_used_pct: u32,
    /// Core temperature in Celsius (0 when the driver does not report it).
    pub temperature_c: u32,
}

/// Host CPU / RAM telemetry (cached briefly server-side).
#[derive(Serialize, ToSchema)]
pub struct HostInfo {
    /// Aggregate CPU utilization across all logical cores, 0-100.
    pub cpu_utilization_pct: u32,
    pub cpu_count: usize,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    /// `memory_used_mb / memory_total_mb`, rounded, 0-100.
    pub memory_used_pct: u32,
}

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub loaded_model: Option<String>,
    pub llama_server_version: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
    pub loaded_model: Option<String>,
    pub llama_server_version: Option<String>,
    pub priority_model: Option<String>,
    pub configured_models: Vec<serde_json::Value>,
    pub uptime_secs: u64,
    /// In-flight requests across all models right now (backs the "processing"
    /// indicator on the Swagger UI badge). 0 means the loaded model is idle.
    pub active_requests: u32,
    /// Per-GPU utilization / memory / temperature. Empty when `nvidia-smi`
    /// is unavailable (non-NVIDIA host or driver not installed).
    pub gpus: Vec<GpuInfo>,
    /// Host CPU / RAM utilization.
    pub host: HostInfo,
    /// Timing breakdown of the most recent model load/switch (ms per phase).
    /// The same data is exported to Prometheus as
    /// `gguf_switchboard_model_switch_seconds` / `..._switch_phase_seconds`.
    pub last_switch: Option<serde_json::Value>,
}

/// Basic liveness probe.
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let loaded = state.scheduler.loaded_model().await;
    let llama_server_version = state.scheduler.loaded_server_version().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_model: loaded,
        llama_server_version,
    })
}

/// Detailed status report.
#[utoipa::path(
    get,
    path = "/status",
    tag = "health",
    responses(
        (status = 200, description = "Detailed service status", body = StatusResponse)
    )
)]
pub async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let loaded = state.scheduler.loaded_model().await;
    let llama_server_version = state.scheduler.loaded_server_version().await;
    let priority = state.scheduler.priority_model();
    let models = state
        .scheduler
        .model_ids()
        .into_iter()
        .map(|id| {
            let cfg = state.scheduler.model_config(&id);
            serde_json::json!({
                "id": id,
                "display_name": cfg.as_ref().map(|c| c.display_name.as_str()).unwrap_or(""),
                "backend": cfg.as_ref().map(|c| c.backend.as_str()).unwrap_or(""),
                "priority": cfg.as_ref().map(|c| c.priority).unwrap_or(false),
            })
        })
        .collect();

    let uptime_secs = state.started_at.elapsed().as_secs();
    let active_requests = state.scheduler.total_active_requests();

    // `nvidia-smi` (subprocess) and the sysinfo refresh are both blocking; keep
    // them off the async worker and cap fan-out with a short server-side cache.
    let (gpus, host) = tokio::task::spawn_blocking(|| {
        let host = {
            let h = crate::sysmon::host_stats_cached(Duration::from_millis(2000));
            HostInfo {
                cpu_utilization_pct: h.cpu_util_pct,
                cpu_count: h.cpu_count,
                memory_total_mb: h.mem_total_mb,
                memory_used_mb: h.mem_used_mb,
                memory_used_pct: h.mem_used_pct,
            }
        };
        let gpus = crate::gpu::probe_all_gpus_cached(Duration::from_millis(2000))
            .into_iter()
            .map(|g| GpuInfo {
                index: g.index,
                name: g.name,
                gpu_utilization_pct: g.gpu_util_pct,
                memory_utilization_pct: g.mem_util_pct,
                memory_total_mb: g.total_mb,
                memory_used_mb: g.used_mb,
                memory_free_mb: g.free_mb,
                memory_used_pct: if g.total_mb > 0 {
                    ((g.used_mb as f64 / g.total_mb as f64) * 100.0).round() as u32
                } else {
                    0
                },
                temperature_c: g.temperature_c,
            })
            .collect::<Vec<_>>();
        (gpus, host)
    })
    .await
    .unwrap_or_else(|_| (Vec::new(), default_host_info()));
    let last_switch = state
        .scheduler
        .last_switch()
        .await
        .and_then(|report| serde_json::to_value(report).ok());

    Json(StatusResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_model: loaded,
        llama_server_version,
        priority_model: priority,
        configured_models: models,
        uptime_secs,
        active_requests,
        gpus,
        host,
        last_switch,
    })
}

fn default_host_info() -> HostInfo {
    HostInfo {
        cpu_utilization_pct: 0,
        cpu_count: 0,
        memory_total_mb: 0,
        memory_used_mb: 0,
        memory_used_pct: 0,
    }
}
