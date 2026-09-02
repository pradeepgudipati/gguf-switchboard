//! Persistent history of conformance-console runs.
//!
//! The console's four surfaces (`inspect`, `resolve-template`, `battery`,
//! `compare`) all recompute on every call and otherwise vanish. This module
//! backs a small self-contained SQLite database (`conformance.db`, sibling of
//! the token-usage DB) so the UI can show previous runs and diffs over time.
//!
//! Mirrors the [`crate::db::TokenDb`] pattern: one `Connection` behind a
//! `Mutex`, schema created on open, best-effort recording that never fails a
//! request.

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use serde_json::Value;
use tracing::info;
use utoipa::ToSchema;

use crate::errors::RuntimeError;

/// One history row without the (potentially large) `detail` JSON blob.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ConformanceRunSummary {
    pub id: i64,
    pub run_at: String,
    /// `battery` | `compare` | `inspect` | `resolve_template`
    pub kind: String,
    pub model: Option<String>,
    /// Second model for `compare` runs.
    pub model_b: Option<String>,
    pub summary: String,
    /// `Some(true/false)` for pass/fail-shaped runs (battery, resolve_template),
    /// `None` otherwise (inspect, compare).
    pub passed: Option<bool>,
}

/// A history row plus its full response payload.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ConformanceRunDetail {
    #[serde(flatten)]
    pub summary: ConformanceRunSummary,
    /// The complete response JSON that was returned to the caller.
    pub detail: Value,
}

pub struct ConformanceHistory {
    conn: Arc<Mutex<Connection>>,
}

impl ConformanceHistory {
    pub fn open(path: &Path) -> Result<Self, RuntimeError> {
        let conn = Connection::open(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to open conformance history database: {e}"))
        })?;
        let db = Self::from_conn(conn)?;
        info!(path = %path.display(), "Conformance history database opened");
        Ok(db)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self, RuntimeError> {
        let conn = Connection::open_in_memory().map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to open in-memory conformance DB: {e}"))
        })?;
        Self::from_conn(conn)
    }

    fn from_conn(conn: Connection) -> Result<Self, RuntimeError> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS conformance_runs (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at  TEXT NOT NULL,
                kind    TEXT NOT NULL,
                model   TEXT,
                model_b TEXT,
                summary TEXT NOT NULL,
                passed  INTEGER,
                detail  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conf_runs_run_at ON conformance_runs(run_at);
            CREATE INDEX IF NOT EXISTS idx_conf_runs_kind   ON conformance_runs(kind);
            CREATE INDEX IF NOT EXISTS idx_conf_runs_model  ON conformance_runs(model);
            ",
        )
        .map_err(|e| {
            RuntimeError::ConfigError(format!(
                "Failed to initialize conformance history schema: {e}"
            ))
        })?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>, RuntimeError> {
        self.conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Conformance DB lock poisoned: {e}")))
    }

    /// Persist one run. Returns the new row id.
    pub fn record(
        &self,
        kind: &str,
        model: Option<&str>,
        model_b: Option<&str>,
        summary: &str,
        passed: Option<bool>,
        detail: &Value,
    ) -> Result<i64, RuntimeError> {
        let conn = self.lock()?;
        let detail_text = serde_json::to_string(detail).unwrap_or_else(|_| "null".to_string());
        conn.execute(
            "INSERT INTO conformance_runs (run_at, kind, model, model_b, summary, passed, detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                Utc::now().to_rfc3339(),
                kind,
                model,
                model_b,
                summary,
                passed,
                detail_text,
            ],
        )
        .map_err(|e| {
            RuntimeError::InternalError(format!("Failed to record conformance run: {e}"))
        })?;
        Ok(conn.last_insert_rowid())
    }

    /// Recent runs, newest first. `kind` / `model` are optional exact-match filters.
    pub fn list(
        &self,
        limit: u32,
        kind: Option<&str>,
        model: Option<&str>,
    ) -> Result<Vec<ConformanceRunSummary>, RuntimeError> {
        let conn = self.lock()?;
        let limit = limit.clamp(1, 500);

        // Build the WHERE clause and the positional param list together so the
        // placeholder numbers always line up with `params`.
        let mut sql = String::from(
            "SELECT id, run_at, kind, model, model_b, summary, passed FROM conformance_runs",
        );
        let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
        let mut clauses: Vec<String> = Vec::new();
        if let Some(k) = &kind {
            params.push(k);
            clauses.push(format!("kind = ?{}", params.len()));
        }
        if let Some(m) = &model {
            params.push(m);
            clauses.push(format!("model = ?{}", params.len()));
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        params.push(&limit);
        sql.push_str(&format!(" ORDER BY id DESC LIMIT ?{}", params.len()));

        let mut stmt = conn.prepare(&sql).map_err(|e| {
            RuntimeError::InternalError(format!("Failed to prepare history query: {e}"))
        })?;
        let rows = stmt
            .query_map(params.as_slice(), Self::map_summary)
            .map_err(|e| RuntimeError::InternalError(format!("Failed to query history: {e}")))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to collect history rows: {e}"))
            })?;
        Ok(rows)
    }

    pub fn get(&self, id: i64) -> Result<Option<ConformanceRunDetail>, RuntimeError> {
        let conn = self.lock()?;
        conn.query_row(
            "SELECT id, run_at, kind, model, model_b, summary, passed, detail
             FROM conformance_runs WHERE id = ?1",
            rusqlite::params![id],
            |row| {
                let summary = Self::map_summary(row)?;
                let detail_text: String = row.get(7)?;
                Ok(ConformanceRunDetail {
                    summary,
                    detail: serde_json::from_str(&detail_text).unwrap_or(Value::Null),
                })
            },
        )
        .optional()
        .map_err(|e| RuntimeError::InternalError(format!("Failed to fetch history row: {e}")))
    }

    /// Returns `true` when a row was deleted.
    pub fn delete(&self, id: i64) -> Result<bool, RuntimeError> {
        let conn = self.lock()?;
        let n = conn
            .execute(
                "DELETE FROM conformance_runs WHERE id = ?1",
                rusqlite::params![id],
            )
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to delete history row: {e}"))
            })?;
        Ok(n > 0)
    }

    /// Delete every row. Returns the number removed.
    pub fn clear(&self) -> Result<u64, RuntimeError> {
        let conn = self.lock()?;
        let n = conn
            .execute("DELETE FROM conformance_runs", [])
            .map_err(|e| RuntimeError::InternalError(format!("Failed to clear history: {e}")))?;
        Ok(n as u64)
    }

    fn map_summary(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConformanceRunSummary> {
        Ok(ConformanceRunSummary {
            id: row.get(0)?,
            run_at: row.get(1)?,
            kind: row.get(2)?,
            model: row.get(3)?,
            model_b: row.get(4)?,
            summary: row.get(5)?,
            passed: row.get::<_, Option<i64>>(6)?.map(|v| v != 0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn db() -> ConformanceHistory {
        ConformanceHistory::open_in_memory().unwrap()
    }

    #[test]
    fn record_list_get_round_trip() {
        let h = db();
        let id = h
            .record(
                "battery",
                Some("gemma-4-e4b"),
                None,
                "3/4 pass",
                Some(false),
                &json!({"cases": []}),
            )
            .unwrap();

        let list = h.list(50, None, None).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, id);
        assert_eq!(list[0].kind, "battery");
        assert_eq!(list[0].passed, Some(false));

        let detail = h.get(id).unwrap().unwrap();
        assert_eq!(detail.detail, json!({"cases": []}));
        assert!(h.get(999).unwrap().is_none());
    }

    #[test]
    fn filters_by_kind_and_model() {
        let h = db();
        h.record("battery", Some("a"), None, "s", Some(true), &json!({}))
            .unwrap();
        h.record("inspect", Some("a"), None, "s", None, &json!({}))
            .unwrap();
        h.record("battery", Some("b"), None, "s", Some(true), &json!({}))
            .unwrap();

        assert_eq!(h.list(50, Some("battery"), None).unwrap().len(), 2);
        assert_eq!(h.list(50, None, Some("a")).unwrap().len(), 2);
        assert_eq!(h.list(50, Some("battery"), Some("a")).unwrap().len(), 1);
    }

    #[test]
    fn newest_first_and_limit() {
        let h = db();
        for i in 0..5 {
            h.record(
                "inspect",
                Some(&format!("m{i}")),
                None,
                "s",
                None,
                &json!({}),
            )
            .unwrap();
        }
        let list = h.list(3, None, None).unwrap();
        assert_eq!(list.len(), 3);
        assert!(list[0].id > list[1].id);
    }

    #[test]
    fn delete_and_clear() {
        let h = db();
        let id = h
            .record("compare", Some("a"), Some("b"), "a vs b", None, &json!({}))
            .unwrap();
        assert!(h.delete(id).unwrap());
        assert!(!h.delete(id).unwrap());

        h.record("inspect", Some("a"), None, "s", None, &json!({}))
            .unwrap();
        h.record("inspect", Some("b"), None, "s", None, &json!({}))
            .unwrap();
        assert_eq!(h.clear().unwrap(), 2);
        assert_eq!(h.list(50, None, None).unwrap().len(), 0);
    }
}
