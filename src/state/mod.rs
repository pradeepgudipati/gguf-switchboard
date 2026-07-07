use std::sync::Arc;
use std::time::Instant;

use crate::config::Config;
use crate::db::TokenDb;
use crate::scheduler::Scheduler;

/// Shared application state passed to all API handlers.
pub struct AppState {
    pub _config: Config,
    pub scheduler: Arc<Scheduler>,
    pub token_db: Arc<TokenDb>,
    pub started_at: Instant,
}

impl AppState {
    pub fn new(config: Config, scheduler: Arc<Scheduler>, token_db: Arc<TokenDb>) -> Self {
        Self {
            _config: config,
            scheduler,
            token_db,
            started_at: Instant::now(),
        }
    }
}
