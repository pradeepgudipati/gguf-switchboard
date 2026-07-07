use std::sync::Arc;

use crate::config::Config;
use crate::scheduler::Scheduler;

/// Shared application state passed to all API handlers.
pub struct AppState {
    pub config: Config,
    pub scheduler: Arc<Scheduler>,
}

impl AppState {
    pub fn new(config: Config, scheduler: Arc<Scheduler>) -> Self {
        Self { config, scheduler }
    }
}
