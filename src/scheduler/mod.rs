use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, RwLock};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::backend::{Backend, create_backend};
use crate::config::{Config, ModelConfig};
use crate::context::{get_context_size, next_lower_context, with_context_size};
use crate::errors::RuntimeError;
use crate::load_failure::{LoadFailureKind, classify_load_failure};
use crate::memory;
use crate::metrics::{BACKEND_HEALTH, LOADED_MODEL, MEMORY_USAGE_PERCENT, MODEL_LOAD_LATENCY};

#[derive(Debug, Clone, Copy)]
enum LoadOrigin {
    UserRequest,
    PriorityWatcher,
}

struct SchedulerInner {
    config: Config,
    backends: RwLock<HashMap<String, Arc<dyn Backend>>>,
    runtime_args: RwLock<HashMap<String, Vec<String>>>,
    loaded: RwLock<Option<String>>,
    load_lock: AsyncMutex<()>,
    recent_models: RwLock<VecDeque<String>>,
    last_activity: RwLock<HashMap<String, Instant>>,
    active_requests: Mutex<HashMap<String, u32>>,
    last_user_switch_at: RwLock<Option<Instant>>,
    last_priority_load_failed_at: RwLock<Option<Instant>>,
    max_loaded: usize,
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
        let inner = SchedulerInner {
            config,
            backends: RwLock::new(HashMap::new()),
            runtime_args: RwLock::new(HashMap::new()),
            loaded: RwLock::new(None),
            load_lock: AsyncMutex::new(()),
            recent_models: RwLock::new(VecDeque::new()),
            last_activity: RwLock::new(HashMap::new()),
            active_requests: Mutex::new(HashMap::new()),
            last_user_switch_at: RwLock::new(None),
            last_priority_load_failed_at: RwLock::new(None),
            max_loaded: 1,
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

    /// Ensure the given model is loaded and ready. Uses load-then-unload with rollback.
    pub async fn ensure_loaded(&self, model_id: &str) -> Result<Arc<dyn Backend>, RuntimeError> {
        self.load_model_id(model_id, LoadOrigin::UserRequest).await
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

        // Stale "loaded" slot (process died / health lost) — clear and reload.
        if self.inner.loaded.read().await.as_deref() == Some(model_id) {
            warn!(
                model = %model_id,
                "Loaded model is no longer healthy; reloading"
            );
            let _ = self.unload_model_no_drain(model_id).await;
            *self.inner.loaded.write().await = None;
            LOADED_MODEL.set(0);
            BACKEND_HEALTH.set(0);
        }

        if !self.inner.config.models.contains_key(model_id) {
            return Err(RuntimeError::ModelNotFound(model_id.to_string()));
        }

        let previous = self.inner.loaded.read().await.clone();
        if let Some(ref prev_id) = previous
            && prev_id != model_id
        {
            self.drain_model(prev_id).await?;
        }

        match self.inner.load_model_with_context_fallback(model_id).await {
            Ok(backend) => {
                if let Some(ref prev_id) = previous
                    && prev_id != model_id
                    && let Err(e) = self.unload_model(prev_id).await
                {
                    warn!(model = %prev_id, error = %e, "Failed to unload previous model after switch");
                }
                *self.inner.loaded.write().await = Some(model_id.to_string());
                self.touch(model_id).await;
                record_recent_model(&self.inner.recent_models, model_id, self.inner.max_loaded)
                    .await;
                if matches!(origin, LoadOrigin::UserRequest) {
                    *self.inner.last_user_switch_at.write().await = Some(Instant::now());
                }
                Ok(backend)
            }
            Err(e) => {
                warn!(
                    model = %model_id,
                    error = %e,
                    previous = ?previous,
                    "Model switch failed; keeping previous model loaded"
                );
                if let Some(ref prev_id) = previous
                    && self.inner.loaded.read().await.as_deref() != Some(prev_id.as_str())
                {
                    match self.inner.load_model_with_context_fallback(prev_id).await {
                        Ok(_) => {
                            *self.inner.loaded.write().await = Some(prev_id.clone());
                            info!(model = %prev_id, "Restored previous model after failed switch");
                        }
                        Err(restore_err) => {
                            error!(
                                model = %prev_id,
                                error = %restore_err,
                                "Failed to restore previous model after failed switch"
                            );
                            *self.inner.loaded.write().await = None;
                        }
                    }
                }
                Err(e)
            }
        }
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
        let backs = self.inner.backends.read().await;
        if let Some(backend) = backs.get(model_id) {
            backend.unload().await?;
        }
        Ok(())
    }

    pub async fn loaded_model(&self) -> Option<String> {
        self.inner.loaded.read().await.clone()
    }

    pub fn priority_model(&self) -> Option<String> {
        self.inner.config.priority_model_id()
    }

    pub fn model_config(&self, model_id: &str) -> Option<&ModelConfig> {
        self.inner.config.models.get(model_id)
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.inner.config.models.keys().cloned().collect()
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
        let backs = self.inner.backends.read().await;
        for (id, backend) in backs.iter() {
            if let Err(e) = backend.unload().await {
                error!(model = %id, error = %e, "Error during shutdown unload");
            }
        }
        *self.inner.loaded.write().await = None;
        LOADED_MODEL.set(0);
        BACKEND_HEALTH.set(0);
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

    async fn unload_model(&self, model_id: &str) -> Result<(), RuntimeError> {
        self.drain_model(model_id).await?;
        let backs = self.inner.backends.read().await;
        if let Some(backend) = backs.get(model_id) {
            info!(model = %model_id, "Unloading model");
            backend.unload().await?;
            BACKEND_HEALTH.set(0);
            LOADED_MODEL.set(0);
        }
        Ok(())
    }

    fn spawn_priority_watcher(self: &Arc<Self>, cancel: CancellationToken) -> JoinHandle<()> {
        let priority_id = match self.inner.config.priority_model_id() {
            Some(id) => id,
            None => {
                return tokio::spawn(async {
                    info!("No priority model configured, skipping priority watcher");
                });
            }
        };

        let idle_timeout = Duration::from_secs(self.inner.config.idle_timeout);
        let cooldown = Duration::from_secs(self.inner.config.priority_load_cooldown_secs);
        let inner = Arc::clone(&self.inner);
        let scheduler = Arc::clone(self);

        tokio::spawn(async move {
            info!(
                model = %priority_id,
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
                            *inner.loaded.write().await = None;
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
    fn has_active_requests(&self) -> bool {
        self.active_requests.lock().values().any(|&count| count > 0)
    }

    async fn effective_args(&self, model_id: &str) -> Result<Vec<String>, RuntimeError> {
        if let Some(args) = self.runtime_args.read().await.get(model_id) {
            return Ok(args.clone());
        }

        self.config
            .models
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
            .config
            .models
            .get(model_id)
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))?;
        let backend_cfg = model_config_with_args(model_cfg, args.clone());
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
                Ok(true) => return Ok(()),
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

    async fn load_model_with_context_fallback(
        &self,
        model_id: &str,
    ) -> Result<Arc<dyn Backend>, RuntimeError> {
        loop {
            let backend = self.get_or_create_backend(model_id).await?;
            let args = self.effective_args(model_id).await?;
            let current_ctx = get_context_size(&args);

            info!(model = %model_id, context = ?current_ctx, "Loading model");

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
                    let next_ctx = get_context_size(&self.effective_args(model_id).await?);
                    warn!(
                        model = %model_id,
                        from = ?current_ctx,
                        to = ?next_ctx,
                        "Retrying model load with reduced context after OOM"
                    );
                    continue;
                }

                self.backends.write().await.remove(model_id);
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Failed to start model '{model_id}': {message}"
                )));
            }

            match self.wait_until_healthy(model_id, &backend).await {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    MODEL_LOAD_LATENCY.observe(elapsed.as_secs_f64());
                    LOADED_MODEL.set(1);
                    BACKEND_HEALTH.set(1);
                    info!(
                        model = %model_id,
                        context = ?current_ctx,
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
                        let next_ctx = get_context_size(&self.effective_args(model_id).await?);
                        warn!(
                            model = %model_id,
                            from = ?current_ctx,
                            to = ?next_ctx,
                            "Retrying model load with reduced context after OOM"
                        );
                        continue;
                    }

                    self.backends.write().await.remove(model_id);
                    return Err(e);
                }
            }
        }
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

        let min = self.config.context_fallback_min;
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
