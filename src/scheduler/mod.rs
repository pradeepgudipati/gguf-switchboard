use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use crate::backend::{create_backend, Backend};
use crate::config::{Config, ModelConfig};
use crate::errors::RuntimeError;
use crate::metrics::{BACKEND_HEALTH, LOADED_MODEL, MODEL_LOAD_LATENCY};

struct SchedulerInner {
    config: Config,
    backends: RwLock<HashMap<String, Arc<dyn Backend>>>,
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
                let _guard = inner.load_lock.lock().await;

                // Double-check
                let current = inner.loaded.read().await.clone();
                if current.as_deref() == Some(&priority_id) {
                    continue;
                }

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

                // Ensure backend exists
                {
                    let mut backs = inner.backends.write().await;
                    if !backs.contains_key(&priority_id) {
                        if let Some(model_cfg) = inner.config.models.get(&priority_id) {
                            let backend = create_backend(&priority_id, model_cfg);
                            backs.insert(priority_id.clone(), Arc::from(backend));
                        } else {
                            error!(model = %priority_id, "Priority model not found in config");
                            continue;
                        }
                    }
                }

                // Load and wait for health
                {
                    let backs = inner.backends.read().await;
                    if let Some(backend) = backs.get(&priority_id) {
                        let start = Instant::now();
                        if let Err(e) = backend.load().await {
                            error!(model = %priority_id, error = %e, "Failed to load priority model");
                            let _ = backend.unload().await;
                            continue;
                        }

                        let deadline =
                            Instant::now() + Duration::from_secs(inner.config.startup_timeout);
                        loop {
                            if Instant::now() > deadline {
                                error!(model = %priority_id, "Priority model health check timed out");
                                let _ = backend.unload().await;
                                break;
                            }
                            match backend.health().await {
                                Ok(true) => {
                                    let elapsed = start.elapsed();
                                    MODEL_LOAD_LATENCY.observe(elapsed.as_secs_f64());
                                    LOADED_MODEL.set(1);
                                    BACKEND_HEALTH.set(1);
                                    info!(
                                        model = %priority_id,
                                        elapsed_ms = elapsed.as_millis(),
                                        "Priority model loaded"
                                    );
                                    *inner.loaded.write().await = Some(priority_id.clone());
                                    lru_write(&inner.lru, &priority_id, inner.max_loaded).await;
                                    break;
                                }
                                Ok(false) => {
                                    tokio::time::sleep(Duration::from_millis(500)).await;
                                }
                                Err(e) => {
                                    warn!(model = %priority_id, error = %e, "Health check error");
                                    tokio::time::sleep(Duration::from_millis(500)).await;
                                }
                            }
                        }
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

        // Get or create the backend
        let backend = {
            let mut backs = self.inner.backends.write().await;
            if !backs.contains_key(model_id) {
                let model_cfg = self
                    .inner
                    .config
                    .models
                    .get(model_id)
                    .ok_or_else(|| RuntimeError::ModelNotFound(model_id.to_string()))?;
                let b = create_backend(model_id, model_cfg);
                backs.insert(model_id.to_string(), Arc::from(b));
            }
            backs[model_id].clone()
        };

        // Load and wait for health
        let start = Instant::now();
        info!(model = %model_id, "Loading model");
        backend.load().await.map_err(|e| {
            RuntimeError::ModelLoadingFailed(format!(
                "Failed to start model '{model_id}': {e}"
            ))
        })?;

        let deadline = Instant::now() + Duration::from_secs(self.inner.config.startup_timeout);
        loop {
            if Instant::now() > deadline {
                let _ = backend.unload().await;
                *self.inner.loaded.write().await = None;
                return Err(RuntimeError::ModelLoadingTimeout(format!(
                    "Model '{model_id}' did not become healthy within {}s",
                    self.inner.config.startup_timeout
                )));
            }
            match backend.health().await {
                Ok(true) => {
                    let elapsed = start.elapsed();
                    MODEL_LOAD_LATENCY.observe(elapsed.as_secs_f64());
                    LOADED_MODEL.set(1);
                    BACKEND_HEALTH.set(1);
                    info!(
                        model = %model_id,
                        elapsed_ms = elapsed.as_millis(),
                        "Model loaded and healthy"
                    );
                    *self.inner.loaded.write().await = Some(model_id.to_string());
                    self.touch(model_id).await;
                    lru_write(&self.inner.lru, model_id, self.inner.max_loaded).await;
                    return Ok(backend);
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
    pub fn config(&self) -> &Config {
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
