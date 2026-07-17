use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex as AsyncMutex, RwLock};
use tracing::{info, warn};

use crate::config::{
    Config, ModelConfig, ModelsRegistry, RescanResult, SyncSummary, sync_registry_from_hf,
};
use crate::db::TokenDb;
use crate::errors::RuntimeError;
use crate::scheduler::Scheduler;

/// Shared application state passed to all API handlers.
pub struct AppState {
    pub scheduler: Arc<Scheduler>,
    pub token_db: Arc<TokenDb>,
    pub registry_json: RwLock<String>,
    pub models_file: Option<String>,
    pub default_backend: String,
    pub vram_gb: u32,
    pub models_rescan_interval_secs: u64,
    pub refresh_lock: AsyncMutex<()>,
    pub started_at: Instant,
}

impl AppState {
    pub fn new(config: Config, scheduler: Arc<Scheduler>, token_db: Arc<TokenDb>) -> Self {
        Self {
            registry_json: RwLock::new(config.registry_json.clone()),
            models_file: config.models_file.clone(),
            default_backend: config.default_backend.clone(),
            vram_gb: config.vram_gb,
            models_rescan_interval_secs: config.models_rescan_interval_secs,
            refresh_lock: AsyncMutex::new(()),
            scheduler,
            token_db,
            started_at: Instant::now(),
        }
    }

    /// Rescan model dirs, enrich metadata from Hugging Face, persist registry, hot-swap live models.
    pub async fn refresh_models(&self) -> Result<RescanResult, RuntimeError> {
        let _guard = self.refresh_lock.lock().await;

        let models_file = self.models_file.as_deref().ok_or_else(|| {
            RuntimeError::ConfigError(
                "models_file is not configured; cannot refresh the model registry".to_string(),
            )
        })?;

        let mut result = ModelsRegistry::rescan_and_write(
            models_file,
            None,
            &self.default_backend,
            self.vram_gb,
        )?;

        match enrich_registry_file(models_file, &self.default_backend, self.vram_gb).await {
            Ok((models, registry_json, summary)) => {
                info!(
                    matched = summary.matched,
                    missed = summary.missed,
                    skipped = summary.skipped,
                    "HF metadata sync complete during model refresh"
                );
                result.models = models;
                result.registry_json = registry_json;
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "HF metadata sync failed during model refresh; using local registry only"
                );
            }
        }

        self.apply_rescan(&result).await?;
        Ok(result)
    }

    async fn apply_rescan(&self, result: &RescanResult) -> Result<(), RuntimeError> {
        let models: HashMap<String, ModelConfig> = result.models.clone();
        self.scheduler.replace_models(models).await?;
        *self.registry_json.write().await = result.registry_json.clone();
        info!(
            added = result.added,
            removed = result.removed,
            total = result.total,
            models_dir = %result.models_dir,
            "Model registry refreshed"
        );
        Ok(())
    }

    /// Background daily (or configured-interval) rescan.
    pub fn spawn_models_rescan_watcher(
        self: &Arc<Self>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let interval_secs = self.models_rescan_interval_secs;
        if interval_secs == 0 {
            info!("Model rescan watcher disabled (models_rescan_interval_secs = 0)");
            return None;
        }

        let state = Arc::clone(self);
        Some(tokio::spawn(async move {
            info!(interval_secs, "Model rescan watcher started");
            let interval = std::time::Duration::from_secs(interval_secs);
            loop {
                tokio::select! {
                    () = cancel.cancelled() => {
                        info!("Model rescan watcher stopped");
                        break;
                    }
                    () = tokio::time::sleep(interval) => {}
                }

                match state.refresh_models().await {
                    Ok(result) => {
                        info!(
                            added = result.added,
                            removed = result.removed,
                            total = result.total,
                            "Scheduled model rescan complete"
                        );
                    }
                    Err(err) => {
                        warn!(error = %err, "Scheduled model rescan failed");
                    }
                }
            }
        }))
    }
}

async fn enrich_registry_file(
    models_file: &str,
    default_backend: &str,
    vram_gb: u32,
) -> Result<(HashMap<String, ModelConfig>, String, SyncSummary), RuntimeError> {
    let mut registry = ModelsRegistry::load(models_file)?;
    let summary = sync_registry_from_hf(&mut registry).await?;
    registry.write(models_file)?;
    let models = registry.expand(default_backend, vram_gb)?;
    let registry_json = serde_json::to_string_pretty(&registry.to_json_export())
        .map_err(|e| RuntimeError::ConfigError(format!("Failed to serialize models JSON: {e}")))?;
    Ok((models, registry_json, summary))
}
