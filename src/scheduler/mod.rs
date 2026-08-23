use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::backend::{Backend, create_backend};
use crate::config::{Config, ModelConfig, SwitchStrategy};
use crate::context::{get_context_size, next_lower_context, with_context_size};
use crate::errors::RuntimeError;
use crate::load_failure::{LoadFailureKind, classify_load_failure};
use crate::memory;
use crate::metrics::{
    self, BACKEND_HEALTH, LOADED_MODEL, MEMORY_USAGE_PERCENT, MODEL_UNLOADS_TOTAL,
    REQUEST_MODEL_HIT_TOTAL, REQUEST_MODEL_WAIT_SECONDS,
};
use crate::ngl::{compute_auto_ngl, with_ngl};

#[derive(Debug, Clone, Copy)]
enum LoadOrigin {
    UserRequest,
    PriorityWatcher,
}

impl LoadOrigin {
    fn label(self) -> &'static str {
        match self {
            LoadOrigin::UserRequest => "request",
            LoadOrigin::PriorityWatcher => "priority",
        }
    }
}

/// Snapshot of the most recent switch, exposed on `/status` for quick debugging
/// without a Prometheus scrape.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LastSwitchReport {
    pub from: Option<String>,
    pub to: String,
    pub trigger: &'static str,
    pub ok: bool,
    pub total_ms: u64,
    pub drain_ms: u64,
    pub unload_previous_ms: u64,
    pub load_ms: u64,
    pub rollback_ms: u64,
    pub finished_at: String,
}

struct SchedulerInner {
    config: Config,
    models: parking_lot::RwLock<HashMap<String, ModelConfig>>,
    backends: RwLock<HashMap<String, Arc<dyn Backend>>>,
    runtime_args: RwLock<HashMap<String, Vec<String>>>,
    loaded: RwLock<Option<String>>,
    load_lock: AsyncMutex<()>,
    recent_models: RwLock<VecDeque<String>>,
    last_activity: RwLock<HashMap<String, Instant>>,
    active_requests: Mutex<HashMap<String, u32>>,
    last_user_switch_at: RwLock<Option<Instant>>,
    last_priority_load_failed_at: RwLock<Option<Instant>>,
    /// Number of ids kept in `recent_models` (resident model + prewarm candidates).
    recent_history: usize,
    /// Cancellation token for the in-flight page-cache prewarm task, if any.
    prewarm: Mutex<Option<CancellationToken>>,
    last_switch: RwLock<Option<LastSwitchReport>>,
    /// Load-time tool-call probe verdict per model id. In-memory only —
    /// recomputed on every load, never persisted to `models.toml`.
    tool_capability: RwLock<HashMap<String, crate::backend::tool_probe::ToolCapability>>,
}

/// Holds background watcher tasks; cancel and join on shutdown.
pub struct WatcherHandles {
    cancel: CancellationToken,
    priority: JoinHandle<()>,
    memory: JoinHandle<()>,
}

impl WatcherHandles {
    pub async fn shutdown(self) {
        self.cancel.cancel();
        let _ = tokio::join!(self.priority, self.memory);
    }
}

/// RAII guard that decrements per-model active request count when dropped.
pub struct RequestGuard {
    scheduler: Arc<SchedulerInner>,
    model_id: String,
}

unsafe impl Send for RequestGuard {}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let mut counts = self.scheduler.active_requests.lock();
        if let Some(count) = counts.get_mut(&self.model_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                counts.remove(&self.model_id);
            }
        }
    }
}

/// Core model scheduler: single-slot swapping with memory-pressure monitoring.
pub struct Scheduler {
    inner: Arc<SchedulerInner>,
}

impl Scheduler {
    pub async fn new(config: Config) -> Result<Self, RuntimeError> {
        let models = parking_lot::RwLock::new(config.models.clone());
        let prewarm = config.prewarm_recent_models;
        if config.switch_strategy == SwitchStrategy::LoadFirst {
            warn!(
                strategy = "load_first",
                "Next model starts while the previous one still holds the GPU; \
                 use unload_first unless VRAM can hold two models at once"
            );
        }
        let inner = SchedulerInner {
            config,
            models,
            backends: RwLock::new(HashMap::new()),
            runtime_args: RwLock::new(HashMap::new()),
            loaded: RwLock::new(None),
            load_lock: AsyncMutex::new(()),
            recent_models: RwLock::new(VecDeque::new()),
            last_activity: RwLock::new(HashMap::new()),
            active_requests: Mutex::new(HashMap::new()),
            last_user_switch_at: RwLock::new(None),
            last_priority_load_failed_at: RwLock::new(None),
            recent_history: 1 + prewarm,
            prewarm: Mutex::new(None),
            last_switch: RwLock::new(None),
            tool_capability: RwLock::new(HashMap::new()),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Spawn priority and memory background watchers.
    pub fn start_watchers(self: &Arc<Self>) -> WatcherHandles {
        let cancel = CancellationToken::new();
        let priority = self.spawn_priority_watcher(cancel.clone());
        let memory = self.spawn_memory_watcher(cancel.clone());
        WatcherHandles {
            cancel,
            priority,
            memory,
        }
    }

    /// Track an in-flight request for `model_id` until the returned guard is dropped.
    pub fn track_request(self: &Arc<Self>, model_id: &str) -> RequestGuard {
        {
            let mut counts = self.inner.active_requests.lock();
            *counts.entry(model_id.to_string()).or_insert(0) += 1;
        }
        RequestGuard {
            scheduler: Arc::clone(&self.inner),
            model_id: model_id.to_string(),
        }
    }

    pub fn active_requests_for(&self, model_id: &str) -> u32 {
        self.inner
            .active_requests
            .lock()
            .get(model_id)
            .copied()
            .unwrap_or(0)
    }

    /// Ensure the given model is loaded and ready.
    ///
    /// Fast path: the model is resident and healthy → returned immediately.
    /// Slow path: drain the resident model, swap according to
    /// [`Config::switch_strategy`], and roll back to the previous model on failure.
    pub async fn ensure_loaded(&self, model_id: &str) -> Result<Arc<dyn Backend>, RuntimeError> {
        // Reject unknown ids before they become Prometheus label values.
        if !self.inner.models.read().contains_key(model_id) {
            return Err(RuntimeError::ModelNotFound(model_id.to_string()));
        }

        if let Some(backend) = self.live_loaded_backend(model_id).await? {
            self.touch(model_id).await;
            REQUEST_MODEL_HIT_TOTAL
                .with_label_values(&[model_id, "hit"])
                .inc();
            return Ok(backend);
        }

        REQUEST_MODEL_HIT_TOTAL
            .with_label_values(&[model_id, "miss"])
            .inc();
        let waited = Instant::now();
        let result = self.load_model_id(model_id, LoadOrigin::UserRequest).await;
        // Includes time spent queued behind another request's in-progress load.
        REQUEST_MODEL_WAIT_SECONDS
            .with_label_values(&[model_id])
            .observe(waited.elapsed().as_secs_f64());
        result
    }

    async fn load_model_id(
        &self,
        model_id: &str,
        origin: LoadOrigin,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        if let Some(backend) = self.live_loaded_backend(model_id).await? {
            self.touch(model_id).await;
            return Ok(backend);
        }

        let _guard = self.inner.load_lock.lock().await;

        if let Some(backend) = self.live_loaded_backend(model_id).await? {
            self.touch(model_id).await;
            return Ok(backend);
        }

        if !self.inner.models.read().contains_key(model_id) {
            return Err(RuntimeError::ModelNotFound(model_id.to_string()));
        }

        // A load is about to hit the disk; make sure a background prewarm is not
        // competing with it for I/O bandwidth.
        self.inner.cancel_prewarm();

        let switch_started = Instant::now();
        let mut report = LastSwitchReport {
            from: self.inner.loaded.read().await.clone(),
            to: model_id.to_string(),
            trigger: origin.label(),
            ok: false,
            total_ms: 0,
            drain_ms: 0,
            unload_previous_ms: 0,
            load_ms: 0,
            rollback_ms: 0,
            finished_at: String::new(),
        };

        // Stale "loaded" slot (process died / health lost) — clear and reload.
        if self.inner.loaded.read().await.as_deref() == Some(model_id) {
            warn!(model = %model_id, "Loaded model is no longer healthy; reloading");
            let _ = self.unload_model_no_drain(model_id).await;
            MODEL_UNLOADS_TOTAL
                .with_label_values(&[model_id, "unhealthy"])
                .inc();
            self.clear_loaded_slot().await;
            report.from = None;
        }

        let previous = self.inner.loaded.read().await.clone();
        let switching_from = previous
            .as_deref()
            .filter(|prev| *prev != model_id)
            .map(str::to_string);
        let from_label = switching_from.as_deref().unwrap_or("none").to_string();
        let unload_reason = match origin {
            LoadOrigin::UserRequest => "switch",
            LoadOrigin::PriorityWatcher => "idle_priority",
        };

        // Phase 1: drain in-flight requests on the resident model.
        let mut previous_unloaded = false;
        if let Some(ref prev_id) = switching_from {
            let t = Instant::now();
            let drained = self.drain_model(prev_id).await;
            report.drain_ms = millis(t.elapsed());
            metrics::record_phase(model_id, metrics::phase::DRAIN, t.elapsed().as_secs_f64());
            if let Err(e) = drained {
                self.finish_switch(report, switch_started, &from_label, metrics::result::ERROR)
                    .await;
                return Err(e);
            }

            // Phase 2 (unload_first): free the GPU before the new model starts, so
            // auto_ngl / the fit planner see the full VRAM and llama-server does not
            // OOM → retry → partially offload to CPU.
            if self.inner.config.switch_strategy == SwitchStrategy::UnloadFirst {
                let t = Instant::now();
                if let Err(e) = self.unload_model_no_drain(prev_id).await {
                    error!(
                        model = %prev_id,
                        error = %e,
                        "Failed to unload previous model before switch — aborting switch"
                    );
                    self.finish_switch(report, switch_started, &from_label, metrics::result::ERROR)
                        .await;
                    return Err(e);
                }

                // Verify the old backend process is actually dead before loading
                // the new model.  A stale process still holds VRAM and the CUDA
                // driver may not have reclaimed it yet.
                self.wait_for_backend_drain(prev_id).await;

                MODEL_UNLOADS_TOTAL
                    .with_label_values(&[prev_id.as_str(), unload_reason])
                    .inc();
                self.clear_loaded_slot().await;
                previous_unloaded = true;
                report.unload_previous_ms = millis(t.elapsed());
                metrics::record_phase(
                    model_id,
                    metrics::phase::UNLOAD_PREVIOUS,
                    t.elapsed().as_secs_f64(),
                );
                info!(
                    from = %prev_id,
                    to = %model_id,
                    unload_ms = report.unload_previous_ms,
                    "Previous model unloaded; starting target model"
                );
            }
        }

        // Phase 3: load the target (planning + one or more spawn attempts).
        let t = Instant::now();
        let loaded = self.inner.load_model_with_context_fallback(model_id).await;
        report.load_ms = millis(t.elapsed());

        match loaded {
            Ok(backend) => {
                // Phase 2 (load_first): previous model is still resident; drop it now.
                if let Some(ref prev_id) = switching_from
                    && !previous_unloaded
                {
                    let t = Instant::now();
                    if let Err(e) = self.unload_model(prev_id).await {
                        warn!(
                            model = %prev_id,
                            error = %e,
                            "Failed to unload previous model after switch"
                        );
                    }
                    MODEL_UNLOADS_TOTAL
                        .with_label_values(&[prev_id.as_str(), unload_reason])
                        .inc();
                    report.unload_previous_ms = millis(t.elapsed());
                    metrics::record_phase(
                        model_id,
                        metrics::phase::UNLOAD_PREVIOUS,
                        t.elapsed().as_secs_f64(),
                    );
                }
                self.mark_loaded(model_id).await;
                self.touch(model_id).await;
                record_recent_model(
                    &self.inner.recent_models,
                    model_id,
                    self.inner.recent_history,
                )
                .await;
                if matches!(origin, LoadOrigin::UserRequest) {
                    *self.inner.last_user_switch_at.write().await = Some(Instant::now());
                }
                report.ok = true;
                self.finish_switch(report, switch_started, &from_label, metrics::result::OK)
                    .await;
                self.spawn_prewarm(model_id).await;
                Ok(backend)
            }
            Err(e) => {
                warn!(
                    model = %model_id,
                    error = %e,
                    previous = ?switching_from,
                    strategy = ?self.inner.config.switch_strategy,
                    "Model switch failed"
                );
                // Rollback: with unload_first the previous model is gone and must be
                // re-loaded; with load_first it never stopped serving.
                if let Some(ref prev_id) = switching_from
                    && previous_unloaded
                {
                    let t = Instant::now();
                    match self.inner.load_model_with_context_fallback(prev_id).await {
                        Ok(_) => {
                            self.mark_loaded(prev_id).await;
                            info!(
                                model = %prev_id,
                                rollback_ms = t.elapsed().as_millis(),
                                "Restored previous model after failed switch"
                            );
                        }
                        Err(restore_err) => {
                            error!(
                                model = %prev_id,
                                error = %restore_err,
                                "Failed to restore previous model after failed switch"
                            );
                            self.clear_loaded_slot().await;
                        }
                    }
                    report.rollback_ms = millis(t.elapsed());
                    metrics::record_phase(
                        model_id,
                        metrics::phase::ROLLBACK,
                        t.elapsed().as_secs_f64(),
                    );
                } else if switching_from.is_some() {
                    info!(previous = ?switching_from, "Previous model kept loaded (load_first)");
                }
                self.finish_switch(report, switch_started, &from_label, metrics::result::ERROR)
                    .await;
                Err(e)
            }
        }
    }

    async fn mark_loaded(&self, model_id: &str) {
        *self.inner.loaded.write().await = Some(model_id.to_string());
        LOADED_MODEL.set(1);
        BACKEND_HEALTH.set(1);
        metrics::set_loaded_model_info(Some(model_id));
    }

    async fn clear_loaded_slot(&self) {
        *self.inner.loaded.write().await = None;
        LOADED_MODEL.set(0);
        BACKEND_HEALTH.set(0);
        metrics::set_loaded_model_info(None);
    }

    async fn finish_switch(
        &self,
        mut report: LastSwitchReport,
        started: Instant,
        from_label: &str,
        outcome: &str,
    ) {
        let elapsed = started.elapsed();
        report.total_ms = millis(elapsed);
        report.finished_at = chrono::Utc::now().to_rfc3339();
        metrics::MODEL_SWITCH_SECONDS
            .with_label_values(&[report.to.as_str(), outcome])
            .observe(elapsed.as_secs_f64());
        metrics::MODEL_SWITCHES_TOTAL
            .with_label_values(&[from_label, report.to.as_str(), report.trigger, outcome])
            .inc();
        info!(
            from = %from_label,
            to = %report.to,
            trigger = report.trigger,
            result = outcome,
            total_ms = report.total_ms,
            drain_ms = report.drain_ms,
            unload_previous_ms = report.unload_previous_ms,
            load_ms = report.load_ms,
            rollback_ms = report.rollback_ms,
            "Model switch finished"
        );
        *self.inner.last_switch.write().await = Some(report);
    }

    /// Most recent switch timing breakdown (also exported to Prometheus).
    pub async fn last_switch(&self) -> Option<LastSwitchReport> {
        self.inner.last_switch.read().await.clone()
    }

    /// Re-read recently used (non-resident) GGUF files into the page cache in the
    /// background so the next switch back to them is served from RAM. Opt-in via
    /// `prewarm_recent_models`; cancelled as soon as another load starts.
    async fn spawn_prewarm(&self, resident: &str) {
        if self.inner.config.prewarm_recent_models == 0 {
            return;
        }
        let candidates: Vec<String> = self
            .inner
            .recent_models
            .read()
            .await
            .iter()
            .filter(|id| id.as_str() != resident)
            .cloned()
            .collect();
        let mut paths = Vec::new();
        {
            let models = self.inner.models.read();
            for id in candidates {
                if let Some(path) = models
                    .get(&id)
                    .and_then(|cfg| crate::ngl::model_path_from_args(&cfg.args))
                {
                    paths.push((id, path.to_string()));
                }
            }
        }
        if paths.is_empty() {
            return;
        }

        self.inner.cancel_prewarm();
        let token = CancellationToken::new();
        *self.inner.prewarm.lock() = Some(token.clone());

        tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut buf = vec![0u8; 8 * 1024 * 1024];
            for (id, path) in paths {
                let started = Instant::now();
                let Ok(mut file) = std::fs::File::open(&path) else {
                    continue;
                };
                let mut bytes: u64 = 0;
                loop {
                    if token.is_cancelled() {
                        debug!(model = %id, "Page-cache prewarm cancelled");
                        return;
                    }
                    match file.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => bytes += n as u64,
                        Err(e) => {
                            debug!(model = %id, error = %e, "Page-cache prewarm read error");
                            break;
                        }
                    }
                }
                info!(
                    model = %id,
                    mb = bytes / (1024 * 1024),
                    elapsed_ms = started.elapsed().as_millis(),
                    "Page-cache prewarm complete"
                );
            }
        });
    }

    pub async fn get_backend(&self, model_id: &str) -> Result<Arc<dyn Backend>, RuntimeError> {
        let backs = self.inner.backends.read().await;
        backs
            .get(model_id)
            .cloned()
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))
    }

    /// Return the backend only when `model_id` is the loaded slot and still alive.
    async fn live_loaded_backend(
        &self,
        model_id: &str,
    ) -> Result<Option<Arc<dyn Backend>>, RuntimeError> {
        if self.inner.loaded.read().await.as_deref() != Some(model_id) {
            return Ok(None);
        }
        let backend = self.get_backend(model_id).await?;
        if backend.process_running().await && backend.health().await.unwrap_or(false) {
            return Ok(Some(backend));
        }
        Ok(None)
    }

    /// Unload without waiting for in-flight drains (used when the process is already dead).
    async fn unload_model_no_drain(&self, model_id: &str) -> Result<(), RuntimeError> {
        let mut backs = self.inner.backends.write().await;
        if let Some(backend) = backs.remove(model_id) {
            backend.unload().await?;
        }
        Ok(())
    }

    pub async fn loaded_model(&self) -> Option<String> {
        self.inner.loaded.read().await.clone()
    }

    pub fn priority_model(&self) -> Option<String> {
        self.inner
            .models
            .read()
            .iter()
            .find(|(_, cfg)| cfg.priority)
            .map(|(id, _)| id.clone())
    }

    pub fn model_config(&self, model_id: &str) -> Option<ModelConfig> {
        self.inner.models.read().get(model_id).cloned()
    }

    /// The most recent tool-call probe verdict for `model_id`, if the model
    /// has ever been loaded and probed. `None` means the model has never
    /// been loaded (not the same as `Skipped`, which means it loaded but
    /// never claimed `tools` capability).
    pub async fn tool_capability(
        &self,
        model_id: &str,
    ) -> Option<crate::backend::tool_probe::ToolCapability> {
        self.inner
            .tool_capability
            .read()
            .await
            .get(model_id)
            .copied()
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.inner.models.read().keys().cloned().collect()
    }

    /// Replace the live model map. Unloads the current model if it was removed.
    pub async fn replace_models(
        &self,
        new_models: HashMap<String, ModelConfig>,
    ) -> Result<(), RuntimeError> {
        let _guard = self.inner.load_lock.lock().await;

        let loaded = self.inner.loaded.read().await.clone();
        if let Some(ref id) = loaded
            && !new_models.contains_key(id)
        {
            info!(model = %id, "Unloading model removed by registry refresh");
            let _ = self.unload_model_no_drain(id).await;
            MODEL_UNLOADS_TOTAL
                .with_label_values(&[id.as_str(), "registry_refresh"])
                .inc();
            self.clear_loaded_slot().await;
        }

        {
            let mut backs = self.inner.backends.write().await;
            let stale: Vec<String> = backs
                .keys()
                .filter(|id| !new_models.contains_key(*id))
                .cloned()
                .collect();
            for id in stale {
                if let Some(backend) = backs.remove(&id) {
                    let _ = backend.unload().await;
                }
                self.inner.runtime_args.write().await.remove(&id);
            }
        }

        *self.inner.models.write() = new_models;
        Ok(())
    }

    pub async fn loaded_server_version(&self) -> Option<String> {
        let loaded = self.inner.loaded.read().await.clone()?;
        let backs = self.inner.backends.read().await;
        let backend = backs.get(&loaded)?;
        backend.server_version().await
    }

    pub fn _config(&self) -> &Config {
        &self.inner.config
    }

    pub async fn shutdown(&self) -> Result<(), RuntimeError> {
        info!("Shutting down scheduler");
        self.inner.cancel_prewarm();
        let loaded = self.inner.loaded.read().await.clone();
        let backs = self.inner.backends.read().await;
        for (id, backend) in backs.iter() {
            if let Err(e) = backend.unload().await {
                error!(model = %id, error = %e, "Error during shutdown unload");
            }
        }
        drop(backs);
        if let Some(id) = loaded {
            MODEL_UNLOADS_TOTAL
                .with_label_values(&[id.as_str(), "shutdown"])
                .inc();
        }
        self.clear_loaded_slot().await;
        Ok(())
    }

    async fn touch(&self, model_id: &str) {
        self.inner
            .last_activity
            .write()
            .await
            .insert(model_id.to_string(), Instant::now());
    }

    async fn drain_model(&self, model_id: &str) -> Result<(), RuntimeError> {
        let timeout = Duration::from_secs(self.inner.config.switch_drain_timeout_secs);
        let deadline = Instant::now() + timeout;

        loop {
            let active = self.active_requests_for(model_id);
            if active == 0 {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RuntimeError::ModelBusy(format!(
                    "Model '{model_id}' still has {active} active request(s) after {}s drain timeout",
                    timeout.as_secs()
                )));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// After unloading a backend, wait for the process to fully exit and for the
    /// CUDA driver to release VRAM.  This prevents the next model from seeing
    /// stale "used" VRAM and OOMing.
    async fn wait_for_backend_drain(&self, model_id: &str) {
        let max_wait = Duration::from_secs(10);
        let poll_interval = Duration::from_millis(200);
        let deadline = Instant::now() + max_wait;

        // Poll until the backend reports the process is dead.
        loop {
            let backs = self.inner.backends.read().await;
            let alive = match backs.get(model_id) {
                Some(backend) => backend.process_running().await,
                None => false,
            };
            drop(backs);

            if !alive {
                break;
            }

            if Instant::now() >= deadline {
                warn!(
                    model = %model_id,
                    "Backend process still alive after {}s drain wait; proceeding anyway",
                    max_wait.as_secs()
                );
                break;
            }

            tokio::time::sleep(poll_interval).await;
        }

        // Give the CUDA driver a moment to reclaim VRAM after the process exits.
        // On multi-GPU setups with large models this can take a noticeable amount
        // of time; a short sleep is cheap insurance against false OOM.
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    async fn unload_model(&self, model_id: &str) -> Result<(), RuntimeError> {
        self.drain_model(model_id).await?;
        let backs = self.inner.backends.read().await;
        if let Some(backend) = backs.get(model_id) {
            let started = Instant::now();
            info!(model = %model_id, "Unloading model");
            backend.unload().await?;
            info!(
                model = %model_id,
                elapsed_ms = started.elapsed().as_millis(),
                "Model unloaded"
            );
            BACKEND_HEALTH.set(0);
            LOADED_MODEL.set(0);
        }
        Ok(())
    }

    fn spawn_priority_watcher(self: &Arc<Self>, cancel: CancellationToken) -> JoinHandle<()> {
        let idle_timeout = Duration::from_secs(self.inner.config.idle_timeout);
        let cooldown = Duration::from_secs(self.inner.config.priority_load_cooldown_secs);
        let inner = Arc::clone(&self.inner);
        let scheduler = Arc::clone(self);

        tokio::spawn(async move {
            info!(
                timeout_secs = idle_timeout.as_secs(),
                "Priority model watcher started"
            );
            loop {
                tokio::select! {
                    () = cancel.cancelled() => {
                        info!("Priority model watcher stopped");
                        break;
                    }
                    () = tokio::time::sleep(Duration::from_secs(30)) => {}
                }

                let Some(priority_id) = scheduler.priority_model() else {
                    continue;
                };

                let current = inner.loaded.read().await.clone();
                if current.as_deref() == Some(&priority_id) {
                    continue;
                }

                if inner.has_active_requests() {
                    debug!("Skipping priority load: active requests in progress");
                    continue;
                }

                if let Some(failed_at) = *inner.last_priority_load_failed_at.read().await
                    && failed_at.elapsed() < cooldown
                {
                    debug!("Skipping priority load: cooldown after recent failure");
                    continue;
                }

                if let Some(switched_at) = *inner.last_user_switch_at.read().await
                    && switched_at.elapsed() < Duration::from_secs(30)
                {
                    debug!("Skipping priority load: recent user-initiated switch");
                    continue;
                }

                let should_load = {
                    let activity = inner.last_activity.read().await;
                    match current {
                        Some(ref model_id) => {
                            let last = activity.get(model_id).copied();
                            match last {
                                Some(t) => t.elapsed() >= idle_timeout,
                                None => true,
                            }
                        }
                        None => true,
                    }
                };

                if !should_load {
                    continue;
                }

                info!(model = %priority_id, "Idle timeout reached, loading priority model");

                if let Err(e) = scheduler
                    .load_model_id(&priority_id, LoadOrigin::PriorityWatcher)
                    .await
                {
                    error!(model = %priority_id, error = %e, "Failed to load priority model");
                    *inner.last_priority_load_failed_at.write().await = Some(Instant::now());
                }
            }
        })
    }

    fn spawn_memory_watcher(self: &Arc<Self>, cancel: CancellationToken) -> JoinHandle<()> {
        let inner = Arc::clone(&self.inner);
        let scheduler = Arc::clone(self);
        let interval = Duration::from_secs(inner.config.memory_check_interval_secs);
        let warning = inner.config.memory_warning_threshold;
        let critical = inner.config.memory_critical_threshold;

        tokio::spawn(async move {
            info!(
                interval_secs = interval.as_secs(),
                warning_pct = warning,
                critical_pct = critical,
                "Memory watcher started"
            );

            loop {
                tokio::select! {
                    () = cancel.cancelled() => {
                        info!("Memory watcher stopped");
                        break;
                    }
                    () = tokio::time::sleep(interval) => {}
                }

                let stats = match memory::check_memory() {
                    Some(s) => s,
                    None => {
                        debug!("Memory stats unavailable, skipping check");
                        continue;
                    }
                };

                MEMORY_USAGE_PERCENT.set(stats.used_percent as i64);

                if stats.used_percent >= critical {
                    error!(
                        used_percent = stats.used_percent,
                        total_mb = stats.total_mb,
                        available_mb = stats.available_mb,
                        "CRITICAL: system memory pressure — unloading model"
                    );

                    let _guard = inner.load_lock.lock().await;
                    let current = inner.loaded.read().await.clone();
                    if let Some(ref model_id) = current {
                        if inner.has_active_requests() {
                            warn!(
                                model = %model_id,
                                "Skipping memory-pressure unload while requests are active"
                            );
                            continue;
                        }
                        if let Err(e) = scheduler.unload_model(model_id).await {
                            error!(model = %model_id, error = %e, "Failed to unload model under memory pressure");
                        } else {
                            MODEL_UNLOADS_TOTAL
                                .with_label_values(&[model_id.as_str(), "memory_pressure"])
                                .inc();
                            scheduler.clear_loaded_slot().await;
                            info!(model = %model_id, "Model unloaded due to critical memory pressure");
                        }
                    }
                } else if stats.used_percent >= warning {
                    warn!(
                        used_percent = stats.used_percent,
                        total_mb = stats.total_mb,
                        available_mb = stats.available_mb,
                        "WARNING: high system memory usage"
                    );
                } else {
                    debug!(
                        used_percent = stats.used_percent,
                        available_mb = stats.available_mb,
                        "Memory OK"
                    );
                }
            }
        })
    }
}

impl SchedulerInner {
    fn cancel_prewarm(&self) {
        if let Some(token) = self.prewarm.lock().take() {
            token.cancel();
        }
    }

    fn has_active_requests(&self) -> bool {
        self.active_requests.lock().values().any(|&count| count > 0)
    }

    async fn effective_args(&self, model_id: &str) -> Result<Vec<String>, RuntimeError> {
        if let Some(args) = self.runtime_args.read().await.get(model_id) {
            return Ok(args.clone());
        }

        self.models
            .read()
            .get(model_id)
            .map(|cfg| cfg.args.clone())
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))
    }

    async fn recreate_backend(
        &self,
        model_id: &str,
        args: Vec<String>,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        let model_cfg = self
            .models
            .read()
            .get(model_id)
            .cloned()
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))?;
        let backend_cfg = model_config_with_args(&model_cfg, args.clone());
        let backend: Arc<dyn Backend> = Arc::from(create_backend(model_id, &backend_cfg));
        self.backends
            .write()
            .await
            .insert(model_id.to_string(), backend.clone());
        self.runtime_args
            .write()
            .await
            .insert(model_id.to_string(), args);
        Ok(backend)
    }

    async fn get_or_create_backend(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        if let Some(backend) = self.backends.read().await.get(model_id).cloned() {
            return Ok(backend);
        }

        let args = self.effective_args(model_id).await?;
        self.recreate_backend(model_id, args).await
    }

    async fn wait_until_healthy(
        &self,
        model_id: &str,
        backend: &Arc<dyn Backend>,
    ) -> Result<(), RuntimeError> {
        let deadline = Instant::now() + Duration::from_secs(self.config.startup_timeout);
        loop {
            if Instant::now() > deadline {
                let stderr = backend.take_startup_stderr().await;
                let _ = backend.unload().await;
                return Err(RuntimeError::ModelLoadingTimeout(format!(
                    "Model '{model_id}' did not become healthy within {}s{stderr_suffix}",
                    self.config.startup_timeout,
                    stderr_suffix = if stderr.is_empty() {
                        String::new()
                    } else {
                        format!("\n{stderr}")
                    }
                )));
            }

            if !backend.process_running().await {
                let stderr = backend.take_startup_stderr().await;
                let _ = backend.unload().await;
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Model '{model_id}' backend process exited before becoming healthy{stderr_suffix}",
                    stderr_suffix = if stderr.is_empty() {
                        String::new()
                    } else {
                        format!("\n{stderr}")
                    }
                )));
            }

            match backend.health().await {
                Ok(true) => {
                    self.run_tool_probe(model_id, backend).await;
                    return Ok(());
                }
                Ok(false) => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(e) => {
                    debug!(model = %model_id, error = %e, "Health check error, retrying");
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }

    /// Run the tool-call probe for `model_id` if its static capabilities
    /// claim `tools`, and store the verdict. Runs once per successful
    /// health check (i.e. once per load) — cheap enough not to need
    /// caching beyond the `tool_capability` map itself, since this method
    /// is only called from `wait_until_healthy`'s single success path.
    async fn run_tool_probe(&self, model_id: &str, backend: &Arc<dyn Backend>) {
        use crate::backend::tool_probe::{
            ToolCapability, normalize_tool_call_arguments, probe_request, validate_probe_response,
        };

        let claims_tools = self
            .models
            .read()
            .get(model_id)
            .is_some_and(|cfg| cfg.capabilities.iter().any(|c| c == "tools"));

        let verdict = if !claims_tools {
            ToolCapability::Skipped
        } else {
            match backend.chat(probe_request(model_id)).await {
                Ok(response) => {
                    let mut value =
                        serde_json::to_value(&response).unwrap_or(serde_json::Value::Null);
                    normalize_tool_call_arguments(&mut value);
                    match validate_probe_response(&value) {
                        Ok(()) => ToolCapability::Verified,
                        Err(reason) => ToolCapability::Failed(reason),
                    }
                }
                Err(e) => {
                    warn!(model = %model_id, error = %e, "Tool-call probe request failed");
                    ToolCapability::Failed("backend_error")
                }
            }
        };

        info!(model = %model_id, verdict = ?verdict, "Tool-call capability probe complete");
        self.tool_capability
            .write()
            .await
            .insert(model_id.to_string(), verdict);
    }

    async fn load_model_with_context_fallback(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        // vLLM has no `-ngl`/`-c` flags, so the llama.cpp-specific auto_ngl and
        // FitPlanner paths below don't apply — they'd rewrite/append flags vLLM's
        // argparse doesn't recognize. vLLM always goes through its own planner
        // (not gated by `fit.enabled`: an initial --gpu-memory-utilization /
        // --tensor-parallel-size still needs to be derived from hardware).
        let is_vllm = self
            .models
            .read()
            .get(model_id)
            .is_some_and(|cfg| cfg.backend == "vllm");
        if is_vllm {
            return self.load_vllm_model_with_fit(model_id).await;
        }

        // Embedding models use the balanced VRAM planner by default. Other model
        // kinds retain the opt-in global fit behavior.
        let embedding_fit = self.config.embedding_fit.enabled
            && self
                .models
                .read()
                .get(model_id)
                .is_some_and(|cfg| cfg.kind == "embedding");
        if self.config.fit.enabled || embedding_fit {
            return self.load_model_with_fit_planner(model_id).await;
        }

        // Legacy path: auto_ngl + context halving.
        let plan_started = Instant::now();
        self.apply_auto_ngl(model_id).await?;
        metrics::record_phase(
            model_id,
            metrics::phase::PLAN,
            plan_started.elapsed().as_secs_f64(),
        );

        loop {
            let backend = self.get_or_create_backend(model_id).await?;
            let args = self.effective_args(model_id).await?;
            let current_ctx = get_context_size(&args);
            let current_ngl = crate::ngl::get_ngl(&args);

            info!(model = %model_id, context = ?current_ctx, ngl = ?current_ngl, "Loading model");

            let start = Instant::now();
            if let Err(e) = backend.load().await {
                let stderr = backend.take_startup_stderr().await;
                let message = e.to_string();
                warn!(model = %model_id, error = %message, "Model load failed");
                let _ = backend.unload().await;

                if self
                    .should_reduce_context(model_id, &message, &stderr, current_ctx)
                    .await?
                {
                    metrics::record_load_attempt(
                        model_id,
                        metrics::result::OOM_RETRY,
                        start.elapsed().as_secs_f64(),
                    );
                    let next_ctx = get_context_size(&self.effective_args(model_id).await?);
                    warn!(
                        model = %model_id,
                        from = ?current_ctx,
                        to = ?next_ctx,
                        "Retrying model load with reduced context after OOM"
                    );
                    continue;
                }

                metrics::record_load_attempt(
                    model_id,
                    metrics::result::ERROR,
                    start.elapsed().as_secs_f64(),
                );
                self.backends.write().await.remove(model_id);
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Failed to start model '{model_id}': {message}"
                )));
            }

            match self.wait_until_healthy(model_id, &backend).await {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    metrics::record_load_attempt(
                        model_id,
                        metrics::result::OK,
                        elapsed.as_secs_f64(),
                    );
                    LOADED_MODEL.set(1);
                    BACKEND_HEALTH.set(1);
                    info!(
                        model = %model_id,
                        context = ?current_ctx,
                        ngl = ?current_ngl,
                        elapsed_ms = elapsed.as_millis(),
                        "Model loaded and healthy"
                    );
                    return Ok(backend);
                }
                Err(e) => {
                    let stderr = backend.take_startup_stderr().await;
                    let message = e.to_string();
                    warn!(model = %model_id, error = %message, "Model health check failed");

                    if self
                        .should_reduce_context(model_id, &message, &stderr, current_ctx)
                        .await?
                    {
                        metrics::record_load_attempt(
                            model_id,
                            metrics::result::OOM_RETRY,
                            start.elapsed().as_secs_f64(),
                        );
                        let next_ctx = get_context_size(&self.effective_args(model_id).await?);
                        warn!(
                            model = %model_id,
                            from = ?current_ctx,
                            to = ?next_ctx,
                            "Retrying model load with reduced context after OOM"
                        );
                        continue;
                    }

                    metrics::record_load_attempt(
                        model_id,
                        load_error_label(&e),
                        start.elapsed().as_secs_f64(),
                    );
                    self.backends.write().await.remove(model_id);
                    return Err(e);
                }
            }
        }
    }

    /// Load a model using the ModelFitPlanner: hardware-aware planning + bounded fallback.
    async fn load_model_with_fit_planner(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        use crate::fit::{FitPlanner, HardwareSummary, ModelSummary};
        use crate::fit_profile::{ProfileStore, default_profile_store_path};

        let model_cfg = self
            .models
            .read()
            .get(model_id)
            .cloned()
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))?;

        let base_args = self.effective_args(model_id).await?;
        let model_path = crate::ngl::model_path_from_args(&base_args)
            .unwrap_or("")
            .to_string();
        let requested_ctx =
            get_context_size(&base_args).unwrap_or(self.config.fit.context_minimum());

        // nvidia-smi + file stat + profile-store read are blocking syscalls/process
        // spawns; keep them off the async workers and account for them as "plan" time.
        let plan_started = Instant::now();
        let vram_gb = self.config.vram_gb;
        let hardware = tokio::task::spawn_blocking(move || HardwareSummary::probe(vram_gb))
            .await
            .map_err(|e| {
                RuntimeError::ModelLoadingFailed(format!("Hardware probe task failed: {e}"))
            })?;
        let mut model = ModelSummary::from_file(
            &model_path,
            model_cfg.block_count,
            None, // architecture — not critical for fit
            model_cfg.max_context_from_gguf,
            requested_ctx,
            model_cfg.ngl_pinned,
            crate::ngl::get_ngl(&base_args),
        )
        .with_kind(&model_cfg.kind)
        .with_context_floor(model_cfg.ctx_floor);
        if model_cfg.kind == "embedding" && self.config.embedding_fit.enabled {
            model.model_fingerprint = format!(
                "{}:embedding-{}-v1",
                model.model_fingerprint, self.config.embedding_fit.profile
            );
        }

        // Check profile cache first.
        let mut profile_store = if self.config.fit.cache_profiles {
            Some(ProfileStore::load(&default_profile_store_path()))
        } else {
            None
        };
        metrics::record_phase(
            model_id,
            metrics::phase::PLAN,
            plan_started.elapsed().as_secs_f64(),
        );
        debug!(
            model = %model_id,
            free_vram_mb = hardware.free_vram_mb,
            total_vram_mb = hardware.total_vram_mb,
            plan_ms = plan_started.elapsed().as_millis(),
            "FitPlanner: hardware probed"
        );

        if let Some(store) = &profile_store
            && let Some(cached) =
                store.get(&hardware.hardware_fingerprint, &model.model_fingerprint)
        {
            info!(
                model = %model_id,
                ctx = cached.plan.context_size,
                ngl = cached.plan.ngl,
                "Using cached known-good profile"
            );
            let fit_args = FitPlanner::apply_plan_to_args(&base_args, &cached.plan);
            self.backends.write().await.remove(model_id);
            self.runtime_args
                .write()
                .await
                .insert(model_id.to_string(), fit_args);

            let backend = self.get_or_create_backend(model_id).await?;
            let start = Instant::now();
            match backend.load().await {
                Ok(()) => match self.wait_until_healthy(model_id, &backend).await {
                    Ok(()) => {
                        let elapsed = start.elapsed();
                        metrics::record_load_attempt(
                            model_id,
                            metrics::result::OK,
                            elapsed.as_secs_f64(),
                        );
                        LOADED_MODEL.set(1);
                        BACKEND_HEALTH.set(1);
                        self.store_runtime_profile(model_id, &cached.plan, "cached")
                            .await;
                        info!(
                            model = %model_id,
                            elapsed_ms = elapsed.as_millis(),
                            "Model loaded from cached profile"
                        );
                        return Ok(backend);
                    }
                    Err(e) => {
                        metrics::record_load_attempt(
                            model_id,
                            load_error_label(&e),
                            start.elapsed().as_secs_f64(),
                        );
                        warn!(
                            model = %model_id,
                            error = %e,
                            "Cached profile failed health check; falling back to planner"
                        );
                        let _ = backend.unload().await;
                        self.backends.write().await.remove(model_id);
                    }
                },
                Err(e) => {
                    metrics::record_load_attempt(
                        model_id,
                        metrics::result::ERROR,
                        start.elapsed().as_secs_f64(),
                    );
                    warn!(
                        model = %model_id,
                        error = %e,
                        "Cached profile failed to load; falling back to planner"
                    );
                    let _ = backend.unload().await;
                    self.backends.write().await.remove(model_id);
                }
            }
        }

        // Build and run the fit planner.
        let embedding_reserve =
            if model_cfg.kind == "embedding" && self.config.embedding_fit.enabled {
                let percentage = hardware
                    .free_vram_mb
                    .saturating_mul(u64::from(self.config.embedding_fit.vram_reserve_percent))
                    / 100;
                u32::try_from(percentage)
                    .unwrap_or(u32::MAX)
                    .max(self.config.embedding_fit.vram_reserve_min_mb)
            } else {
                self.config.fit.vram_reserve_mb
            };
        let fit_config = crate::fit::FitConfig {
            enabled: true,
            vram_reserve_mb: embedding_reserve,
            multi_gpu: self.config.fit.multi_gpu.clone(),
            split_mode: self.config.fit.split_mode.clone(),
            max_attempts: self.config.fit.max_attempts,
            cache_profiles: self.config.fit.cache_profiles,
        };
        let mut planner = FitPlanner::new(hardware, model, fit_config);

        loop {
            let plan = planner.current_plan();
            let attempt = plan.attempt;
            info!(
                model = %model_id,
                attempt,
                plan = %plan,
                "FitPlanner: attempting load"
            );

            let fit_args = FitPlanner::apply_plan_to_args(&base_args, plan);
            self.backends.write().await.remove(model_id);
            self.runtime_args
                .write()
                .await
                .insert(model_id.to_string(), fit_args);

            let backend = self.get_or_create_backend(model_id).await?;
            let start = Instant::now();

            if let Err(e) = backend.load().await {
                let stderr = backend.take_startup_stderr().await;
                let message = e.to_string();
                let kind = crate::load_failure::classify_load_failure(&message, &stderr);
                warn!(
                    model = %model_id,
                    attempt,
                    error = %message,
                    failure_kind = ?kind,
                    "FitPlanner: load failed"
                );
                let _ = backend.unload().await;

                if kind.is_oom()
                    && let Some(next) = planner.advance(&message)
                {
                    metrics::record_load_attempt(
                        model_id,
                        metrics::result::OOM_RETRY,
                        start.elapsed().as_secs_f64(),
                    );
                    warn!(
                        model = %model_id,
                        next_attempt = next.attempt,
                        next_plan = %next,
                        "FitPlanner: retrying with fallback"
                    );
                    continue;
                }

                // Non-OOM or planner exhausted.
                metrics::record_load_attempt(
                    model_id,
                    metrics::result::ERROR,
                    start.elapsed().as_secs_f64(),
                );
                self.backends.write().await.remove(model_id);
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Failed to start model '{model_id}' after {attempt} attempt(s): {message}"
                )));
            }

            match self.wait_until_healthy(model_id, &backend).await {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    metrics::record_load_attempt(
                        model_id,
                        metrics::result::OK,
                        elapsed.as_secs_f64(),
                    );
                    LOADED_MODEL.set(1);
                    BACKEND_HEALTH.set(1);
                    self.store_runtime_profile(model_id, plan, "auto-fit").await;

                    // Cache successful profile.
                    if let Some(store) = &mut profile_store {
                        use crate::fit_profile::KnownGoodProfile;
                        store.put(KnownGoodProfile {
                            hardware_fingerprint: planner.hardware().hardware_fingerprint.clone(),
                            model_fingerprint: planner.model().model_fingerprint.clone(),
                            plan: plan.clone(),
                            validated_at: chrono::Utc::now().to_rfc3339(),
                        });
                        let _ = store.flush();
                    }

                    // Persist effective params to models.toml so next startup uses them directly.
                    if let Some(ref models_file) = self.config.models_file {
                        let fit_extra = build_fit_extra_args(plan);
                        if let Err(e) = crate::config::ModelsRegistry::persist_fit_params(
                            models_file,
                            model_id,
                            plan.context_size,
                            plan.ngl,
                            &fit_extra,
                        ) {
                            warn!(model = %model_id, error = %e, "Failed to persist fit params to registry");
                        }
                    }

                    info!(
                        model = %model_id,
                        attempt,
                        plan = %plan,
                        elapsed_ms = elapsed.as_millis(),
                        "FitPlanner: model loaded successfully"
                    );
                    return Ok(backend);
                }
                Err(e) => {
                    let stderr = backend.take_startup_stderr().await;
                    let message = e.to_string();
                    let kind = crate::load_failure::classify_load_failure(&message, &stderr);
                    warn!(
                        model = %model_id,
                        attempt,
                        error = %message,
                        failure_kind = ?kind,
                        "FitPlanner: health check failed"
                    );

                    if kind.is_oom()
                        && let Some(next) = planner.advance(&message)
                    {
                        metrics::record_load_attempt(
                            model_id,
                            metrics::result::OOM_RETRY,
                            start.elapsed().as_secs_f64(),
                        );
                        warn!(
                            model = %model_id,
                            next_attempt = next.attempt,
                            next_plan = %next,
                            "FitPlanner: retrying with fallback"
                        );
                        continue;
                    }

                    metrics::record_load_attempt(
                        model_id,
                        load_error_label(&e),
                        start.elapsed().as_secs_f64(),
                    );
                    self.backends.write().await.remove(model_id);
                    return Err(e);
                }
            }
        }
    }

    /// Load a vLLM model using the vLLM-specific fit planner: probes hardware,
    /// derives an initial `--gpu-memory-utilization` / `--tensor-parallel-size`
    /// / `--max-model-len`, and retries with a reduced context on OOM. The
    /// llama.cpp `FitPlanner` (ngl/KV-cache-type/tensor-split) doesn't apply
    /// here — see [`load_model_with_context_fallback`].
    async fn load_vllm_model_with_fit(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        use crate::fit::{HardwareSummary, VllmFitPlanner};

        let base_args = self.effective_args(model_id).await?;
        let requested_context = crate::fit::vllm_max_model_len_from_args(&base_args)
            .unwrap_or(self.config.fit.context_minimum());
        let model_size_mb = crate::fit::vllm_model_size_mb_from_args(&base_args);
        let pinned_tp = crate::fit::vllm_tensor_parallel_size_from_args(&base_args);
        let pinned_gmu = crate::fit::vllm_gpu_memory_utilization_from_args(&base_args);

        let plan_started = Instant::now();
        let vram_gb = self.config.vram_gb;
        let hardware = tokio::task::spawn_blocking(move || HardwareSummary::probe(vram_gb))
            .await
            .map_err(|e| {
                RuntimeError::ModelLoadingFailed(format!("Hardware probe task failed: {e}"))
            })?;
        metrics::record_phase(
            model_id,
            metrics::phase::PLAN,
            plan_started.elapsed().as_secs_f64(),
        );

        let mut planner = VllmFitPlanner::new(
            &hardware,
            model_size_mb,
            requested_context,
            pinned_tp,
            pinned_gmu,
            &self.config.fit,
        );

        loop {
            let plan = planner.current_plan();
            let attempt = plan.attempt;
            info!(model = %model_id, attempt, plan = %plan, "VllmFitPlanner: attempting load");

            let fit_args = VllmFitPlanner::apply_plan_to_args(&base_args, plan);
            self.backends.write().await.remove(model_id);
            self.runtime_args
                .write()
                .await
                .insert(model_id.to_string(), fit_args);

            let backend = self.get_or_create_backend(model_id).await?;
            let start = Instant::now();

            if let Err(e) = backend.load().await {
                let stderr = backend.take_startup_stderr().await;
                let message = e.to_string();
                let kind = classify_load_failure(&message, &stderr);
                warn!(model = %model_id, attempt, error = %message, failure_kind = ?kind, "VllmFitPlanner: load failed");
                let _ = backend.unload().await;

                if kind.is_oom()
                    && let Some(next) = planner.advance(&message)
                {
                    metrics::record_load_attempt(
                        model_id,
                        metrics::result::OOM_RETRY,
                        start.elapsed().as_secs_f64(),
                    );
                    warn!(model = %model_id, next_attempt = next.attempt, next_plan = %next, "VllmFitPlanner: retrying with fallback");
                    continue;
                }

                metrics::record_load_attempt(
                    model_id,
                    metrics::result::ERROR,
                    start.elapsed().as_secs_f64(),
                );
                self.backends.write().await.remove(model_id);
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Failed to start model '{model_id}' after {attempt} attempt(s): {message}"
                )));
            }

            match self.wait_until_healthy(model_id, &backend).await {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    metrics::record_load_attempt(model_id, metrics::result::OK, elapsed.as_secs_f64());
                    LOADED_MODEL.set(1);
                    BACKEND_HEALTH.set(1);
                    info!(model = %model_id, attempt, plan = %plan, elapsed_ms = elapsed.as_millis(), "VllmFitPlanner: model loaded successfully");
                    return Ok(backend);
                }
                Err(e) => {
                    let stderr = backend.take_startup_stderr().await;
                    let message = e.to_string();
                    let kind = classify_load_failure(&message, &stderr);
                    warn!(model = %model_id, attempt, error = %message, failure_kind = ?kind, "VllmFitPlanner: health check failed");

                    if kind.is_oom()
                        && let Some(next) = planner.advance(&message)
                    {
                        metrics::record_load_attempt(
                            model_id,
                            metrics::result::OOM_RETRY,
                            start.elapsed().as_secs_f64(),
                        );
                        warn!(model = %model_id, next_attempt = next.attempt, next_plan = %next, "VllmFitPlanner: retrying with fallback");
                        continue;
                    }

                    metrics::record_load_attempt(model_id, load_error_label(&e), start.elapsed().as_secs_f64());
                    self.backends.write().await.remove(model_id);
                    return Err(e);
                }
            }
        }
    }

    /// Store the effective runtime profile on the model config.
    async fn store_runtime_profile(
        &self,
        model_id: &str,
        plan: &crate::fit::FitPlan,
        source: &str,
    ) {
        use crate::config::RuntimeProfile;
        let profile = RuntimeProfile::from_fit_plan(plan, source);
        // We store it by mutating the model in the models map.
        // Since ModelConfig has runtime_profile as skip_serializing, this is safe.
        let mut models = self.models.write();
        if let Some(cfg) = models.get_mut(model_id) {
            cfg.runtime_profile = Some(profile);
        }
    }

    /// Opt-in: rewrite `-ngl` from free VRAM + GGUF size unless the model pins ngl.
    async fn apply_auto_ngl(&self, model_id: &str) -> Result<(), RuntimeError> {
        if !self.config.auto_ngl {
            return Ok(());
        }

        let model_cfg = self
            .models
            .read()
            .get(model_id)
            .cloned()
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))?;

        if model_cfg.ngl_pinned {
            debug!(model = %model_id, "auto_ngl skipped; ngl is pinned");
            return Ok(());
        }

        let args = self.effective_args(model_id).await?;
        let Some(model_path) = crate::ngl::model_path_from_args(&args) else {
            warn!(model = %model_id, "auto_ngl skipped; no -m/--model in args");
            return Ok(());
        };

        let Some(ngl) = compute_auto_ngl(
            model_id,
            model_path,
            self.config.vram_gb,
            model_cfg.block_count,
        ) else {
            warn!(model = %model_id, "auto_ngl skipped; could not read model file size");
            return Ok(());
        };

        let updated = with_ngl(&args, ngl);
        self.backends.write().await.remove(model_id);
        self.runtime_args
            .write()
            .await
            .insert(model_id.to_string(), updated);
        Ok(())
    }

    async fn should_reduce_context(
        &self,
        model_id: &str,
        message: &str,
        stderr: &str,
        current_ctx: Option<u32>,
    ) -> Result<bool, RuntimeError> {
        let kind = classify_load_failure(message, stderr);
        debug!(?kind, "Classified model load failure");
        if kind != LoadFailureKind::Oom {
            return Ok(false);
        }
        Ok(self
            .try_reduce_context(model_id, current_ctx)
            .await?
            .is_some())
    }

    async fn try_reduce_context(
        &self,
        model_id: &str,
        current_ctx: Option<u32>,
    ) -> Result<Option<u32>, RuntimeError> {
        let Some(current) = current_ctx else {
            return Ok(None);
        };

        let per_model_floor = self
            .models
            .read()
            .get(model_id)
            .and_then(|cfg| cfg.ctx_floor);
        let min = self
            .config
            .context_fallback_min
            .max(per_model_floor.unwrap_or(0));
        let Some(next) = next_lower_context(current, min) else {
            return Ok(None);
        };

        let args = self.effective_args(model_id).await?;
        let reduced_args = with_context_size(&args, next);
        self.backends.write().await.remove(model_id);
        self.runtime_args
            .write()
            .await
            .insert(model_id.to_string(), reduced_args);
        Ok(Some(next))
    }
}

fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

fn load_error_label(e: &RuntimeError) -> &'static str {
    if matches!(e, RuntimeError::ModelLoadingTimeout(_)) {
        metrics::result::TIMEOUT
    } else {
        metrics::result::ERROR
    }
}

async fn record_recent_model(recent: &RwLock<VecDeque<String>>, model_id: &str, max_loaded: usize) {
    let mut queue = recent.write().await;
    queue.retain(|id| id != model_id);
    queue.push_front(model_id.to_string());
    while queue.len() > max_loaded {
        if let Some(evicted) = queue.pop_back() {
            debug!(model = %evicted, "Evicted from recent-model queue");
        }
    }
}

fn model_config_with_args(base: &ModelConfig, args: Vec<String>) -> ModelConfig {
    ModelConfig {
        args,
        ..base.clone()
    }
}

/// Extract only the fit-related extra args (`--split-mode`, `--tensor-split`,
/// `--cache-type-k`, `--cache-type-v`, `-b`, `-ub`) from a [`FitPlan`] as flag/value pairs.
fn build_fit_extra_args(plan: &crate::fit::FitPlan) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(ref mode) = plan.split_mode {
        args.push("--split-mode".to_string());
        args.push(mode.clone());
    }
    if let Some(ref ts) = plan.tensor_split {
        args.push("--tensor-split".to_string());
        args.push(
            ts.iter()
                .map(|v| format!("{v:.1}"))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if let Some(ref ct) = plan.cache_type_k {
        args.push("--cache-type-k".to_string());
        args.push(ct.clone());
    }
    if let Some(ref ct) = plan.cache_type_v {
        args.push("--cache-type-v".to_string());
        args.push(ct.clone());
    }
    // Batch sizes for embedding models
    if let Some(batch) = plan.batch_size {
        args.push("-b".to_string());
        args.push(batch.to_string());
    }
    if let Some(ubatch) = plan.ubatch_size {
        args.push("-ub".to_string());
        args.push(ubatch.to_string());
    }
    args
}

#[cfg(test)]
mod tool_capability_tests {
    use crate::backend::tool_probe::ToolCapability;

    #[test]
    fn skipped_when_capabilities_do_not_claim_tools() {
        let capabilities: Vec<String> = vec!["vision".to_string()];
        let should_probe = capabilities.iter().any(|c| c == "tools");
        assert!(!should_probe);
    }

    #[test]
    fn probes_when_capabilities_claim_tools() {
        let capabilities: Vec<String> = vec!["tools".to_string()];
        let should_probe = capabilities.iter().any(|c| c == "tools");
        assert!(should_probe);
    }

    #[test]
    fn verdict_serializes_with_reason_for_failed() {
        let verdict = ToolCapability::Failed("no_tool_calls");
        let json = serde_json::to_value(verdict).unwrap();
        assert_eq!(json, serde_json::json!({"failed": "no_tool_calls"}));
    }
}
