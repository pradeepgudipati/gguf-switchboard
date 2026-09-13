//! Shared token-usage capture for streaming API handlers.
//!
//! Non-streaming handlers get `usage` directly from the backend response.
//! Streaming handlers must instead watch the chunk stream for a terminal
//! usage chunk (backends request one via `stream_options.include_usage`),
//! forward it to the client, and record the counts to the token DB when the
//! stream completes. This module holds the accumulator plus a drop-guard that
//! performs the DB write so aborted streams still record what they saw.

use std::sync::{Arc, Mutex};

use crate::db::TokenDb;
use crate::types::Usage;

/// Thread-safe accumulator for per-chunk usage sightings.
///
/// Backends send usage exactly once as a terminal chunk; if several chunks
/// carry usage (retries, proxies), the last one wins.
#[derive(Debug, Default)]
pub struct StreamUsage {
    inner: Mutex<Option<Usage>>,
}

impl StreamUsage {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(None),
        })
    }

    /// Record a usage sighting from a stream chunk.
    pub fn capture(&self, usage: &Usage) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(usage.clone());
        }
    }

    /// The last captured usage, if any.
    pub fn get(&self) -> Option<Usage> {
        self.inner.lock().ok()?.clone()
    }
}

/// Drop-guard that records captured stream usage to the token DB when the
/// SSE stream finishes (normally or via client disconnect).
///
/// Held inside `GuardedStream`'s guard list so it lives for the full stream
/// lifetime. Also records a throughput sample like the non-streaming paths.
pub struct StreamUsageRecorder {
    usage: Arc<StreamUsage>,
    token_db: Arc<TokenDb>,
    model: String,
    endpoint: &'static str,
    started: std::time::Instant,
}

impl StreamUsageRecorder {
    pub fn new(
        usage: Arc<StreamUsage>,
        token_db: Arc<TokenDb>,
        model: &str,
        endpoint: &'static str,
    ) -> Self {
        Self {
            usage,
            token_db,
            model: model.to_string(),
            endpoint,
            started: std::time::Instant::now(),
        }
    }
}

impl Drop for StreamUsageRecorder {
    fn drop(&mut self) {
        let (prompt, completion, total) = self
            .usage
            .get()
            .map(|u| (u.prompt_tokens, u.completion_tokens, u.total_tokens))
            .unwrap_or((0, 0, 0));
        let _ = self
            .token_db
            .record(&self.model, self.endpoint, prompt, completion, total, None);
        let _ = self.token_db.record_throughput(
            &self.model,
            self.endpoint,
            prompt,
            completion,
            self.started.elapsed().as_secs_f64(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_usage_wins() {
        let usage = StreamUsage::new();
        usage.capture(&Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
        });
        usage.capture(&Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        });
        let got = usage.get().unwrap();
        assert_eq!(
            (got.prompt_tokens, got.completion_tokens, got.total_tokens),
            (10, 20, 30)
        );
    }

    #[test]
    fn empty_until_captured() {
        let usage = StreamUsage::new();
        assert!(usage.get().is_none());
    }

    #[test]
    fn recorder_writes_captured_usage_to_db() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = Arc::new(TokenDb::open(&dir.path().join("usage.db")).unwrap());
        let usage = StreamUsage::new();
        usage.capture(&Usage {
            prompt_tokens: 7,
            completion_tokens: 3,
            total_tokens: 10,
        });
        {
            let _recorder = StreamUsageRecorder::new(
                Arc::clone(&usage),
                Arc::clone(&db),
                "m",
                "/v1/chat/completions",
            );
        }
        let stats = db.get_usage_stats().unwrap();
        assert_eq!(stats.total_prompt_tokens, 7);
        assert_eq!(stats.total_completion_tokens, 3);
        assert_eq!(stats.total_tokens, 10);
    }

    #[test]
    fn recorder_writes_zeros_without_usage() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = Arc::new(TokenDb::open(&dir.path().join("usage.db")).unwrap());
        let usage = StreamUsage::new();
        {
            let _recorder = StreamUsageRecorder::new(
                Arc::clone(&usage),
                Arc::clone(&db),
                "m",
                "/v1/chat/completions",
            );
        }
        let stats = db.get_usage_stats().unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.total_tokens, 0);
    }
}
