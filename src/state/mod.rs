use std::sync::Arc;

use crate::config::Config;
use crate::db::TokenDb;
use crate::scheduler::Scheduler;

/// Shared application state passed to all API handlers.
pub struct AppState {
    pub config: Config,
    pub scheduler: Arc<Scheduler>,
    pub token_db: Arc<TokenDb>,
}

impl AppState {
    pub fn new(config: Config, scheduler: Arc<Scheduler>, token_db: Arc<TokenDb>) -> Self {
        Self {
            config,
            scheduler,
            token_db,
        }
    }
}
