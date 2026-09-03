use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{Duration, Utc};
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use tracing::info;

use crate::errors::RuntimeError;

/// How long per-request throughput samples are retained. Older rows are
/// pruned on open and opportunistically on insert.
pub const THROUGHPUT_RETENTION_DAYS: i64 = 90;

/// Recent-window (days) used for the single tok/s figure shown in the model
/// dropdown. Falls back to the full retention window when the recent window
/// has too few samples.
const THROUGHPUT_RECENT_DAYS: i64 = 14;

/// Minimum samples required in the recent window before it's trusted over
/// the full-history median.
const THROUGHPUT_MIN_RECENT_SAMPLES: usize = 3;

/// Requests below these thresholds are too small to yield a meaningful
/// tokens/second figure and are not recorded.
const THROUGHPUT_MIN_COMPLETION_TOKENS: u32 = 16;
const THROUGHPUT_MIN_ELAPSED_SECS: f64 = 0.05;

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

/// Aggregated throughput for one model over a time window.
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStat {
    pub model: String,
    /// Median generation throughput (completion tokens / wall-clock seconds).
    pub tokens_per_sec: f64,
    pub samples: usize,
    pub window_days: i64,
}

/// One point on a model's throughput trendline (one calendar day, UTC).
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ThroughputTrendPoint {
    /// `YYYY-MM-DD` (UTC).
    pub date: String,
    /// Median tokens/second across that day's samples.
    pub tokens_per_sec: f64,
    pub samples: usize,
}

pub struct TokenDb {
    conn: Arc<Mutex<Connection>>,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
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

            CREATE TABLE IF NOT EXISTS model_throughput (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT NOT NULL,
                model             TEXT NOT NULL,
                endpoint          TEXT NOT NULL,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                elapsed_ms        INTEGER NOT NULL,
                tokens_per_second REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_model_throughput_model_ts
                ON model_throughput(model, timestamp);
            ",
        )
        .map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to initialize token database schema: {e}"))
        })?;

        let cutoff = (Utc::now() - Duration::days(THROUGHPUT_RETENTION_DAYS)).to_rfc3339();
        let pruned = conn
            .execute(
                "DELETE FROM model_throughput WHERE timestamp < ?1",
                rusqlite::params![cutoff],
            )
            .unwrap_or(0);
        if pruned > 0 {
            info!(pruned, "Pruned expired model_throughput rows on open");
        }

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
        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

        let timestamp = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO token_usage (timestamp, model, endpoint, prompt_tokens, completion_tokens, total_tokens, request_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![timestamp, model, endpoint, prompt_tokens, completion_tokens, total_tokens, request_id],
        )
        .map_err(|e| RuntimeError::InternalError(format!("Failed to record token usage: {e}")))?;

        Ok(())
    }

    /// Record a per-request generation-throughput sample into the dedicated
    /// `model_throughput` table. `elapsed_secs` is wall-clock time for the
    /// backend call. Trivially small requests are ignored (they don't yield a
    /// meaningful tok/s). Errors are swallowed by callers — a missing sample
    /// must never fail a completion.
    pub fn record_throughput(
        &self,
        model: &str,
        endpoint: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
        elapsed_secs: f64,
    ) -> Result<(), RuntimeError> {
        if completion_tokens < THROUGHPUT_MIN_COMPLETION_TOKENS
            || elapsed_secs < THROUGHPUT_MIN_ELAPSED_SECS
            || !elapsed_secs.is_finite()
        {
            return Ok(());
        }
        let tps = completion_tokens as f64 / elapsed_secs;
        if !tps.is_finite() || tps <= 0.0 {
            return Ok(());
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

        let now = Utc::now();
        conn.execute(
            "INSERT INTO model_throughput
                (timestamp, model, endpoint, prompt_tokens, completion_tokens, elapsed_ms, tokens_per_second)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                now.to_rfc3339(),
                model,
                endpoint,
                prompt_tokens,
                completion_tokens,
                (elapsed_secs * 1000.0).round() as i64,
                tps,
            ],
        )
        .map_err(|e| {
            RuntimeError::InternalError(format!("Failed to record throughput sample: {e}"))
        })?;

        // Opportunistic prune (~1 in 64 inserts) to bound table growth
        // between restarts without paying the DELETE cost every request.
        if conn.last_insert_rowid() % 64 == 0 {
            let cutoff = (now - Duration::days(THROUGHPUT_RETENTION_DAYS)).to_rfc3339();
            let _ = conn.execute(
                "DELETE FROM model_throughput WHERE timestamp < ?1",
                rusqlite::params![cutoff],
            );
        }

        Ok(())
    }

    /// Median tokens/second per model for the dropdown label. Uses the recent
    /// window when it has enough samples, otherwise the full retention window.
    pub fn throughput_by_model(&self) -> Result<HashMap<String, ThroughputStat>, RuntimeError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

        let full_cutoff =
            (Utc::now() - Duration::days(THROUGHPUT_RETENTION_DAYS)).to_rfc3339();
        let recent_cutoff =
            (Utc::now() - Duration::days(THROUGHPUT_RECENT_DAYS)).to_rfc3339();

        let mut stmt = conn
            .prepare(
                "SELECT model, timestamp, tokens_per_second
                 FROM model_throughput
                 WHERE timestamp >= ?1",
            )
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to prepare throughput query: {e}"))
            })?;

        let mut recent: HashMap<String, Vec<f64>> = HashMap::new();
        let mut full: HashMap<String, Vec<f64>> = HashMap::new();
        let rows = stmt
            .query_map(rusqlite::params![full_cutoff], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to query throughput: {e}"))
            })?;
        for row in rows {
            let (model, ts, tps) =
                row.map_err(|e| RuntimeError::InternalError(format!("throughput row: {e}")))?;
            if ts >= recent_cutoff {
                recent.entry(model.clone()).or_default().push(tps);
            }
            full.entry(model).or_default().push(tps);
        }

        let mut out = HashMap::new();
        for (model, mut full_vals) in full {
            let recent_vals = recent.remove(&model);
            let (mut vals, window_days) = match recent_vals {
                Some(mut r) if r.len() >= THROUGHPUT_MIN_RECENT_SAMPLES => {
                    let n = std::mem::take(&mut r);
                    (n, THROUGHPUT_RECENT_DAYS)
                }
                _ => (std::mem::take(&mut full_vals), THROUGHPUT_RETENTION_DAYS),
            };
            let samples = vals.len();
            if samples == 0 {
                continue;
            }
            out.insert(
                model.clone(),
                ThroughputStat {
                    model,
                    tokens_per_sec: median(&mut vals),
                    samples,
                    window_days,
                },
            );
        }
        Ok(out)
    }

    /// Per-day median throughput for one model over the last `days` days,
    /// oldest first. Days with no samples are omitted.
    pub fn throughput_trend(
        &self,
        model: &str,
        days: i64,
    ) -> Result<Vec<ThroughputTrendPoint>, RuntimeError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

        let cutoff = (Utc::now() - Duration::days(days.max(1))).to_rfc3339();
        let mut stmt = conn
            .prepare(
                "SELECT substr(timestamp, 1, 10) AS day, tokens_per_second
                 FROM model_throughput
                 WHERE model = ?1 AND timestamp >= ?2
                 ORDER BY day ASC",
            )
            .map_err(|e| {
                RuntimeError::InternalError(format!("Failed to prepare trend query: {e}"))
            })?;

        let rows = stmt
            .query_map(rusqlite::params![model, cutoff], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .map_err(|e| RuntimeError::InternalError(format!("Failed to query trend: {e}")))?;

        // Rows arrive day-ordered; group consecutively.
        let mut points: Vec<ThroughputTrendPoint> = Vec::new();
        let mut cur_day: Option<String> = None;
        let mut bucket: Vec<f64> = Vec::new();
        for row in rows {
            let (day, tps) =
                row.map_err(|e| RuntimeError::InternalError(format!("trend row: {e}")))?;
            match &cur_day {
                Some(d) if d == &day => bucket.push(tps),
                _ => {
                    if let Some(d) = cur_day.take() {
                        points.push(ThroughputTrendPoint {
                            date: d,
                            tokens_per_sec: median(&mut bucket),
                            samples: bucket.len(),
                        });
                    }
                    cur_day = Some(day);
                    bucket = vec![tps];
                }
            }
        }
        if let Some(d) = cur_day {
            points.push(ThroughputTrendPoint {
                date: d,
                tokens_per_sec: median(&mut bucket),
                samples: bucket.len(),
            });
        }
        Ok(points)
    }

    pub fn get_usage_stats(&self) -> Result<UsageStats, RuntimeError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

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
            .map_err(|e| RuntimeError::InternalError(format!("Failed to query model stats: {e}")))?
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
        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

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
        let conn = self
            .conn
            .lock()
            .map_err(|e| RuntimeError::InternalError(format!("Database lock poisoned: {e}")))?;

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

#[cfg(test)]
mod throughput_tests {
    use super::*;
    use rusqlite::params;

    fn db() -> TokenDb {
        let f = tempfile::NamedTempFile::new().unwrap();
        TokenDb::open(f.path()).unwrap()
    }

    /// Insert a throughput row at an explicit timestamp (bypasses the
    /// live-clock insert in `record_throughput`).
    fn insert_at(db: &TokenDb, model: &str, ts: &str, tps: f64) {
        let conn = db.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO model_throughput
                (timestamp, model, endpoint, prompt_tokens, completion_tokens, elapsed_ms, tokens_per_second)
             VALUES (?1, ?2, '/v1/chat/completions', 10, 100, 1000, ?3)",
            params![ts, model, tps],
        )
        .unwrap();
    }

    #[test]
    fn ignores_trivial_requests() {
        let db = db();
        db.record_throughput("m", "/v1/chat/completions", 10, 5, 1.0)
            .unwrap();
        db.record_throughput("m", "/v1/chat/completions", 10, 100, 0.001)
            .unwrap();
        assert!(db.throughput_by_model().unwrap().is_empty());
    }

    #[test]
    fn records_and_computes_tps() {
        let db = db();
        // 100 tokens in 2s => 50 tok/s
        db.record_throughput("m", "/v1/chat/completions", 10, 100, 2.0)
            .unwrap();
        let stat = &db.throughput_by_model().unwrap()["m"];
        assert!((stat.tokens_per_sec - 50.0).abs() < 0.001);
        assert_eq!(stat.samples, 1);
    }

    #[test]
    fn recent_window_median_wins_when_enough_samples() {
        let db = db();
        let now = Utc::now();
        // 3 recent samples (median 20) + old samples (would pull toward 100)
        for tps in [10.0, 20.0, 30.0] {
            insert_at(&db, "m", &now.to_rfc3339(), tps);
        }
        for _ in 0..10 {
            let old = (now - Duration::days(30)).to_rfc3339();
            insert_at(&db, "m", &old, 100.0);
        }
        let stat = &db.throughput_by_model().unwrap()["m"];
        assert_eq!(stat.window_days, THROUGHPUT_RECENT_DAYS);
        assert!((stat.tokens_per_sec - 20.0).abs() < 0.001);
    }

    #[test]
    fn falls_back_to_full_history_when_recent_sparse() {
        let db = db();
        let now = Utc::now();
        insert_at(&db, "m", &now.to_rfc3339(), 20.0); // 1 recent < min
        for _ in 0..5 {
            insert_at(&db, "m", &(now - Duration::days(30)).to_rfc3339(), 100.0);
        }
        let stat = &db.throughput_by_model().unwrap()["m"];
        assert_eq!(stat.window_days, THROUGHPUT_RETENTION_DAYS);
        assert!((stat.tokens_per_sec - 100.0).abs() < 0.001);
    }

    #[test]
    fn trend_buckets_by_day() {
        let db = db();
        insert_at(&db, "m", "2026-01-01T01:00:00+00:00", 10.0);
        insert_at(&db, "m", "2026-01-01T05:00:00+00:00", 30.0);
        insert_at(&db, "m", "2026-01-02T05:00:00+00:00", 40.0);
        insert_at(&db, "other", "2026-01-01T05:00:00+00:00", 999.0);
        let trend = db.throughput_trend("m", 100_000).unwrap();
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0].date, "2026-01-01");
        assert!((trend[0].tokens_per_sec - 20.0).abs() < 0.001);
        assert_eq!(trend[0].samples, 2);
        assert_eq!(trend[1].date, "2026-01-02");
    }

    #[test]
    fn prunes_rows_past_retention_on_open() {
        let f = tempfile::NamedTempFile::new().unwrap();
        {
            let db = TokenDb::open(f.path()).unwrap();
            let ancient = (Utc::now() - Duration::days(THROUGHPUT_RETENTION_DAYS + 5)).to_rfc3339();
            insert_at(&db, "m", &ancient, 42.0);
            insert_at(&db, "m", &Utc::now().to_rfc3339(), 42.0);
        }
        let db = TokenDb::open(f.path()).unwrap();
        let conn = db.conn.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM model_throughput", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }
}
