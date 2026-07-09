mod api;
mod backend;
mod config;
mod context;
mod db;
mod errors;
mod memory;
mod metrics;
mod proxy;
mod sanitize;
mod scheduler;
mod state;
mod types;

use std::sync::Arc;

use tokio::signal;
use tracing::{info, warn};

use std::path::PathBuf;

use crate::config::Config;
use crate::db::TokenDb;
use crate::scheduler::Scheduler;
use crate::state::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .json()
        .init();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    metrics::register_all();

    info!("Loading configuration from {}", config_path);
    let config = Config::load(&config_path)?;

    info!(
        bind = %config.bind,
        backend = %config.default_backend,
        "Starting GGUF Switchboard"
    );

    let db_path = config
        .database_path
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("token_usage.db"));

    let token_db = Arc::new(TokenDb::open(&db_path)?);

    let scheduler = Arc::new(Scheduler::new(config.clone()).await?);
    let app_state = Arc::new(AppState::new(config.clone(), scheduler.clone(), token_db));

    scheduler.start_priority_watcher().await;
    scheduler.start_memory_watcher().await;

    let app = api::create_router(app_state.clone());

    let bind: std::net::SocketAddr = config.bind.parse()?;
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let base_url = format!(
        "http://{}",
        if bind.ip().is_unspecified() {
            format!("localhost:{}", bind.port())
        } else {
            bind.to_string()
        }
    );
    info!(address = %bind, swagger_ui = %format!("{base_url}/swagger-ui/"), "Server listening");

    let shutdown_signal = async {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        warn!("Shutdown signal received, starting graceful shutdown");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("Shutting down scheduler");
    scheduler.shutdown().await?;

    info!("GGUF Switchboard stopped");
    Ok(())
}
