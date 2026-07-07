use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use crate::backend::{Backend, create_backend};
use crate::config::{Config, ModelConfig};
use crate::context::{get_context_size, next_lower_context, with_context_size};
use crate::errors::RuntimeError;
use crate::memory;
use crate::metrics::{BACKEND_HEALTH, LOADED_MODEL, MEMORY_USAGE_PERCENT, MODEL_LOAD_LATENCY};

struct SchedulerInner {
    config: Config,
    backends: RwLock<HashMap<String, Arc<dyn Backend>>>,
    runtime_args: RwLock<HashMap<String, Vec<String>>>,
    loaded: RwLock<Option<String>>,
    load_lock: Mutex<()>,
    lru: RwLock<VecDeque<String>>,
    last_activity: RwLock<HashMap<String, Instant>>,
    max_loaded: usize,
}

/// Core model scheduler that manages loading, unloading, and LRU eviction
/// of inference backends.
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
            load_lock: Mutex::new(()),
            lru: RwLock::new(VecDeque::new()),
            last_activity: RwLock::new(HashMap::new()),
            max_loaded: 1,
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Spawn a background task that loads the priority model after the
    /// configured idle timeout.
    pub async fn start_priority_watcher(&self) {
        let priority_id = match self.inner.config.priority_model_id() {
            Some(id) => id,
            None => {
                info!("No priority model configured, skipping priority watcher");
                return;
            }
        };

        let idle_timeout = Duration::from_secs(self.inner.config.idle_timeout);
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            info!(
                model = %priority_id,
                timeout_secs = idle_timeout.as_secs(),
                "Priority model watcher started"
            );
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;

                let current = inner.loaded.read().await.clone();
                if current.as_deref() == Some(&priority_id) {
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

                let should_load_priority = {
                    let _guard = inner.load_lock.lock().await;

                    // Double-check: already loaded?
                    let current = inner.loaded.read().await.clone();
                    if current.as_deref() == Some(&priority_id) {
                        false
                    } else {
                        // Unload current model if any
                        if let Some(ref model_id) = current {
                            let backs = inner.backends.read().await;
                            if let Some(backend) = backs.get(model_id) {
                                if let Err(e) = backend.unload().await {
                                    error!(model = %model_id, error = %e, "Failed to unload model");
                                }
                                BACKEND_HEALTH.set(0);
                                LOADED_MODEL.set(0);
                            }
                        }
                        true
                    }
                    // _guard dropped here — load_lock released
                };

                if !should_load_priority {
                    continue;
                }

                if !inner.config.models.contains_key(&priority_id) {
                    error!(model = %priority_id, "Priority model not found in config");
                    continue;
                }

                match inner.load_model_with_context_fallback(&priority_id).await {
                    Ok(_backend) => {
                        *inner.loaded.write().await = Some(priority_id.clone());
                        lru_write(&inner.lru, &priority_id, inner.max_loaded).await;
                    }
                    Err(e) => {
                        error!(model = %priority_id, error = %e, "Failed to load priority model");
                    }
                }
            }
        });
    }

    /// Ensure the given model is loaded and ready. If a different model is
    /// currently loaded, it will be unloaded first (LRU eviction).
    pub async fn ensure_loaded(&self, model_id: &str) -> Result<Arc<dyn Backend>, RuntimeError> {
        // Fast path: already loaded
        {
            let loaded = self.inner.loaded.read().await;
            if loaded.as_deref() == Some(model_id) {
                self.touch(model_id).await;
                return self.get_backend(model_id).await;
            }
        }

        // Slow path: acquire the load lock
        let _guard = self.inner.load_lock.lock().await;

        // Double-check after acquiring lock
        {
            let loaded = self.inner.loaded.read().await;
            if loaded.as_deref() == Some(model_id) {
                self.touch(model_id).await;
                return self.get_backend(model_id).await;
            }
        }

        // Validate model exists in config
        if !self.inner.config.models.contains_key(model_id) {
            return Err(RuntimeError::ModelNotFound(model_id.to_string()));
        }

        // Unload current model if any
        {
            let current = self.inner.loaded.read().await.clone();
            if let Some(ref old_model) = current {
                let backs = self.inner.backends.read().await;
                if let Some(backend) = backs.get(old_model) {
                    info!(model = %old_model, "Unloading model");
                    if let Err(e) = backend.unload().await {
                        error!(model = %old_model, error = %e, "Error unloading model");
                    }
                    BACKEND_HEALTH.set(0);
                    LOADED_MODEL.set(0);
                }
            }
        }

        // Get or create the backend, load, and reduce context on failure.
        let backend = self
            .inner
            .load_model_with_context_fallback(model_id)
            .await?;

        *self.inner.loaded.write().await = Some(model_id.to_string());
        self.touch(model_id).await;
        lru_write(&self.inner.lru, model_id, self.inner.max_loaded).await;
        Ok(backend)
    }

    /// Get the backend for a model that is already loaded.
    pub async fn get_backend(&self, model_id: &str) -> Result<Arc<dyn Backend>, RuntimeError> {
        let backs = self.inner.backends.read().await;
        backs
            .get(model_id)
            .cloned()
            .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))
    }

    /// Return the currently loaded model id, if any.
    pub async fn loaded_model(&self) -> Option<String> {
        self.inner.loaded.read().await.clone()
    }

    /// Return the model id of the priority model from config.
    pub fn priority_model(&self) -> Option<String> {
        self.inner.config.priority_model_id()
    }

    /// Return model config for a given model id.
    pub fn model_config(&self, model_id: &str) -> Option<&ModelConfig> {
        self.inner.config.models.get(model_id)
    }

    /// Return all configured model ids.
    pub fn model_ids(&self) -> Vec<String> {
        self.inner.config.models.keys().cloned().collect()
    }

    /// Return a reference to the config.
    pub fn _config(&self) -> &Config {
        &self.inner.config
    }

    /// Gracefully shut down all backends.
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

    /// Spawn a background task that monitors system memory pressure.
    ///
    /// At the warning threshold a warning is logged. At the critical threshold
    /// the currently loaded model is automatically unloaded to reclaim memory.
    pub async fn start_memory_watcher(&self) {
        let inner = Arc::clone(&self.inner);
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
                tokio::time::sleep(interval).await;

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
                        "CRITICAL: Memory pressure — unloading model"
                    );

                    let _guard = inner.load_lock.lock().await;
                    let current = inner.loaded.read().await.clone();
                    if let Some(ref model_id) = current {
                        let backs = inner.backends.read().await;
                        if let Some(backend) = backs.get(model_id) {
                            if let Err(e) = backend.unload().await {
                                error!(model = %model_id, error = %e, "Failed to unload model under memory pressure");
                            }
                            BACKEND_HEALTH.set(0);
                            LOADED_MODEL.set(0);
                            *inner.loaded.write().await = None;
                            info!(model = %model_id, "Model unloaded due to critical memory pressure");
                        }
                    }
                } else if stats.used_percent >= warning {
                    warn!(
                        used_percent = stats.used_percent,
                        total_mb = stats.total_mb,
                        available_mb = stats.available_mb,
                        "WARNING: High memory usage"
                    );
                } else {
                    debug!(
                        used_percent = stats.used_percent,
                        available_mb = stats.available_mb,
                        "Memory OK"
                    );
                }
            }
        });
    }
}

impl SchedulerInner {
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
                let _ = backend.unload().await;
                return Err(RuntimeError::ModelLoadingTimeout(format!(
                    "Model '{model_id}' did not become healthy within {}s",
                    self.config.startup_timeout
                )));
            }

            if !backend.process_running().await {
                let _ = backend.unload().await;
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Model '{model_id}' backend process exited before becoming healthy"
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
                warn!(model = %model_id, error = %e, "Model load failed");
                let _ = backend.unload().await;

                if let Some(next_ctx) = self.try_reduce_context(model_id, current_ctx).await? {
                    warn!(
                        model = %model_id,
                        from = ?current_ctx,
                        to = next_ctx,
                        "Retrying model load with reduced context"
                    );
                    continue;
                }

                self.backends.write().await.remove(model_id);
                return Err(RuntimeError::ModelLoadingFailed(format!(
                    "Failed to start model '{model_id}': {e}"
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
                    warn!(model = %model_id, error = %e, "Model health check failed");

                    if let Some(next_ctx) = self.try_reduce_context(model_id, current_ctx).await? {
                        warn!(
                            model = %model_id,
                            from = ?current_ctx,
                            to = next_ctx,
                            "Retrying model load with reduced context after health failure"
                        );
                        continue;
                    }

                    self.backends.write().await.remove(model_id);
                    return Err(e);
                }
            }
        }
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

fn model_config_with_args(base: &ModelConfig, args: Vec<String>) -> ModelConfig {
    ModelConfig {
        args,
        ..base.clone()
    }
}

/// Update the LRU queue, evicting the oldest entry if at capacity.
async fn lru_write(lru: &RwLock<VecDeque<String>>, model_id: &str, max_loaded: usize) {
    let mut queue = lru.write().await;
    queue.retain(|id| id != model_id);
    queue.push_front(model_id.to_string());
    while queue.len() > max_loaded {
        if let Some(evicted) = queue.pop_back() {
            debug!(model = %evicted, "Evicted from LRU queue");
        }
    }
}
