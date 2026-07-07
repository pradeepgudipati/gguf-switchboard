use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use tracing::info;

use crate::errors::RuntimeError;

#[derive(Debug, Clone, Serialize)]
pub struct TokenUsageRecord {
    pub id: i64,
    pub timestamp: String,
    pub model: String,
    pub endpoint: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelUsageSummary {
    pub model: String,
    pub total_requests: i64,
    pub total_prompt_tokens: i64,
    pub total_completion_tokens: i64,
    pub total_tokens: i64,
    pub first_request: String,
    pub last_request: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageStats {
    pub total_requests: i64,
    pub total_prompt_tokens: i64,
    pub total_completion_tokens: i64,
    pub total_tokens: i64,
    pub per_model: Vec<ModelUsageSummary>,
}

pub struct TokenDb {
    conn: Arc<Mutex<Connection>>,
}

impl TokenDb {
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let conn = Connection::open(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to open token database: {e}"))
        })?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS token_usage (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                model         TEXT NOT NULL,
                endpoint      TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens  INTEGER NOT NULL DEFAULT 0,
                request_id    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_token_usage_model ON token_usage(model);
            CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(timestamp);
            ",
        )
        .map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to initialize token database schema: {e}"))
        })?;

        info!(path = %path.display(), "Token usage database opened");
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn record(
        &self,
        model: &str,
        endpoint: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        request_id: Option<&str>,
    ) -> Result<(), RuntimeError> {
        let conn = self.conn.lock().map_err(|e| {
            RuntimeError::InternalError(format!("Database lock poisoned: {e}"))
        })?;

        let timestamp = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO token_usage (timestamp, model, endpoint, prompt_tokens, completion_tokens, total_tokens, request_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![timestamp, model, endpoint, prompt_tokens, completion_tokens, total_tokens, request_id],
        )
        .map_err(|e| RuntimeError::InternalError(format!("Failed to record token usage: {e}")))?;

        Ok(())
    }

    pub fn get_usage_stats(&self) -> Result<UsageStats, RuntimeError> {
        let conn = self.conn.lock().map_err(|e| {
            RuntimeError::InternalError(format!("Database lock poisoned: {e}"))
        })?;

        // Total stats
        let (total_requests, total_prompt, total_completion, total_total): (i64, i64, i64, i64) =
            conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(total_tokens), 0) FROM token_usage",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .map_err(|e| RuntimeError::InternalError(format!("Failed to query usage stats: {e}")))?;

        // Per-model stats
        let mut stmt = conn
            .prepare(
                "SELECT model,
                        COUNT(*) as total_requests,
                        COALESCE(SUM(prompt_tokens), 0) as total_prompt,
                        COALESCE(SUM(completion_tokens), 0) as total_completion,
                        COALESCE(SUM(total_tokens), 0) as total_total,
                        MIN(timestamp) as first_req,
                        MAX(timestamp) as last_req
                 FROM token_usage
                 GROUP BY model
                 ORDER BY total_total DESC",
            )
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to prepare model stats query: {e}"))
            })?;

        let per_model = stmt
            .query_map([], |row| {
                Ok(ModelUsageSummary {
                    model: row.get(0)?,
                    total_requests: row.get(1)?,
                    total_prompt_tokens: row.get(2)?,
                    total_completion_tokens: row.get(3)?,
                    total_tokens: row.get(4)?,
                    first_request: row.get(5)?,
                    last_request: row.get(6)?,
                })
            })
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to query model stats: {e}"))
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to collect model stats: {e}"))
            })?;

        Ok(UsageStats {
            total_requests,
            total_prompt_tokens: total_prompt,
            total_completion_tokens: total_completion,
            total_tokens: total_total,
            per_model,
        })
    }

    pub fn get_model_usage(&self, model: &str) -> Result<Option<ModelUsageSummary>, RuntimeError> {
        let conn = self.conn.lock().map_err(|e| {
            RuntimeError::InternalError(format!("Database lock poisoned: {e}"))
        })?;

        let mut stmt = conn
            .prepare(
                "SELECT model,
                        COUNT(*) as total_requests,
                        COALESCE(SUM(prompt_tokens), 0) as total_prompt,
                        COALESCE(SUM(completion_tokens), 0) as total_completion,
                        COALESCE(SUM(total_tokens), 0) as total_total,
                        MIN(timestamp) as first_req,
                        MAX(timestamp) as last_req
                 FROM token_usage
                 WHERE model = ?1
                 GROUP BY model",
            )
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to prepare model usage query: {e}"))
            })?;

        let result = stmt
            .query_row(rusqlite::params![model], |row| {
                Ok(ModelUsageSummary {
                    model: row.get(0)?,
                    total_requests: row.get(1)?,
                    total_prompt_tokens: row.get(2)?,
                    total_completion_tokens: row.get(3)?,
                    total_tokens: row.get(4)?,
                    first_request: row.get(5)?,
                    last_request: row.get(6)?,
                })
            })
            .optional()
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to query model usage: {e}"))
            })?;

        Ok(result)
    }

    pub fn get_recent_records(&self, limit: u32) -> Result<Vec<TokenUsageRecord>, RuntimeError> {
        let conn = self.conn.lock().map_err(|e| {
            RuntimeError::InternalError(format!("Database lock poisoned: {e}"))
        })?;

        let mut stmt = conn
            .prepare(
                "SELECT id, timestamp, model, endpoint, prompt_tokens, completion_tokens, total_tokens, request_id
                 FROM token_usage
                 ORDER BY id DESC
                 LIMIT ?1",
            )
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to prepare recent records query: {e}"))
            })?;

        let records = stmt
            .query_map(rusqlite::params![limit], |row| {
                Ok(TokenUsageRecord {
                    id: row.get(0)?,
                    timestamp: row.get(1)?,
                    model: row.get(2)?,
                    endpoint: row.get(3)?,
                    prompt_tokens: row.get(4)?,
                    completion_tokens: row.get(5)?,
                    total_tokens: row.get(6)?,
                    request_id: row.get(7)?,
                })
            })
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to query recent records: {e}"))
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to collect recent records: {e}"))
            })?;

        Ok(records)
    }
}
