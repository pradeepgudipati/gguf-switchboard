//! Persistence layer for known-good model load profiles.
//!
//! After a model loads successfully the resulting [`FitPlan`] is stored in
//! `model-profiles.json` keyed by `(hardware_fingerprint, model_fingerprint)`.
//! On subsequent loads with matching hardware, the cached profile is reused
//! directly — skipping the fallback ladder entirely.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use crate::fit::FitPlan;

/// A validated load profile stored on disk.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnownGoodProfile {
    pub hardware_fingerprint: String,
    pub model_fingerprint: String,
    pub plan: FitPlan,
    /// ISO-8601 timestamp of when this profile was last validated.
    pub validated_at: String,
}

/// Map key: `"{hardware_fingerprint}:{model_fingerprint}"`.
type ProfileMap = HashMap<String, KnownGoodProfile>;

/// On-disk store for known-good model load profiles.
pub struct ProfileStore {
    path: PathBuf,
    profiles: ProfileMap,
}

impl ProfileStore {
    /// Load the profile store from disk, or create an empty one.
    pub fn load(path: &Path) -> Self {
        let profiles = match std::fs::read_to_string(path) {
            Ok(content) => serde_json::from_str::<ProfileMap>(&content).unwrap_or_else(|e| {
                warn!(error = %e, path = %path.display(), "Failed to parse model-profiles.json; starting fresh");
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        };
        debug!(count = profiles.len(), "Loaded model profiles");
        Self {
            path: path.to_path_buf(),
            profiles,
        }
    }

    /// Look up a cached profile that matches the current hardware.
    pub fn get(
        &self,
        hardware_fingerprint: &str,
        model_fingerprint: &str,
    ) -> Option<&KnownGoodProfile> {
        let key = format!("{hardware_fingerprint}:{model_fingerprint}");
        self.profiles.get(&key)
    }

    /// Store or update a successful profile.
    pub fn put(&mut self, profile: KnownGoodProfile) {
        let key = format!(
            "{}:{}",
            profile.hardware_fingerprint, profile.model_fingerprint
        );
        info!(
            key = %key,
            ctx = profile.plan.context_size,
            ngl = profile.plan.ngl,
            "Caching known-good model profile"
        );
        self.profiles.insert(key, profile);
    }

    /// Persist the store to disk.
    pub fn flush(&self) -> Result<(), std::io::Error> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.profiles).map_err(std::io::Error::other)?;
        std::fs::write(&self.path, json)
    }

    /// Number of cached profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

/// Default path for the profile store.
pub fn default_profile_store_path() -> PathBuf {
    PathBuf::from("/var/lib/gguf-switchboard/model-profiles.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::FitPlan;

    fn sample_plan() -> FitPlan {
        FitPlan {
            context_size: 16384,
            ngl: 999,
            split_mode: None,
            tensor_split: None,
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q8_0".to_string()),
            reason: "test".to_string(),
            attempt: 2,
            batch_size: None,
            ubatch_size: None,
        }
    }

    #[test]
    fn put_and_get_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profiles.json");
        let mut store = ProfileStore::load(&path);

        store.put(KnownGoodProfile {
            hardware_fingerprint: "1xRTX4090-24564".to_string(),
            model_fingerprint: "kimi.gguf:42000".to_string(),
            plan: sample_plan(),
            validated_at: "2025-01-01T00:00:00Z".to_string(),
        });

        let found = store.get("1xRTX4090-24564", "kimi.gguf:42000");
        assert!(found.is_some());
        assert_eq!(found.unwrap().plan.context_size, 16384);
    }

    #[test]
    fn flush_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profiles.json");

        {
            let mut store = ProfileStore::load(&path);
            store.put(KnownGoodProfile {
                hardware_fingerprint: "hw".to_string(),
                model_fingerprint: "model".to_string(),
                plan: sample_plan(),
                validated_at: "2025-01-01T00:00:00Z".to_string(),
            });
            store.flush().unwrap();
        }

        let store = ProfileStore::load(&path);
        assert_eq!(store.len(), 1);
        assert!(store.get("hw", "model").is_some());
    }

    #[test]
    fn missing_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profiles.json");
        let store = ProfileStore::load(&path);
        assert!(store.get("nope", "nope").is_none());
    }
}
