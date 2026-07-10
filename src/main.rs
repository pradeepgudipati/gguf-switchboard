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
use crate::config::ModelsRegistry;
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

    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 && args[1] == "discover-models" {
        return run_discover_models(&args);
    }

    if args.len() >= 3 && args[1] == "export-registry" {
        return run_export_registry(&args);
    }

    let config_path = args
        .get(1)
        .cloned()
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

fn run_discover_models(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut models_dir = "/models".to_string();
    let mut output = "models.toml".to_string();
    let mut merge_from: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if let Some(path) = args.get(i + 1) {
                    output = path.clone();
                    i += 2;
                } else {
                    return Err("discover-models: missing value for --output".into());
                }
            }
            "--merge" => {
                if let Some(path) = args.get(i + 1) {
                    merge_from = Some(path.clone());
                    i += 2;
                } else if std::path::Path::new(&output).is_file() {
                    merge_from = Some(output.clone());
                    i += 1;
                } else {
                    return Err("discover-models: missing value for --merge".into());
                }
            }
            arg if arg.starts_with('-') => {
                return Err(format!("discover-models: unknown flag '{arg}'").into());
            }
            path => {
                models_dir = path.to_string();
                i += 1;
            }
        }
    }

    let registry = match merge_from.as_deref() {
        Some(path) => {
            let existing = ModelsRegistry::load(path)?;
            ModelsRegistry::discover_with_merge(&models_dir, Some(&existing))?
        }
        None => ModelsRegistry::discover(&models_dir)?,
    };
    let discovered_count = registry.models.len();
    registry.write(&output)?;

    if discovered_count == 0 {
        println!(
            "Warning: no llama.cpp-loadable .gguf files found in {models_dir}; kept existing registry entries"
        );
    } else {
        println!("Discovered {discovered_count} model(s) in {models_dir}");
    }
    println!("Wrote {output}");
    let json_output = json_sibling_path_for_output(&output);
    println!("Wrote {json_output}");
    if let Some(ref merge_path) = merge_from {
        println!("Merged customizations from {merge_path}");
    }
    println!();
    println!("  {:<24} {:<6} FILE", "ALIAS", "PRI");
    for entry in &registry.models {
        let pri = if entry.priority { "yes" } else { "" };
        println!("  {:<24} {:<6} {}", entry.alias, pri, entry.file);
    }
    println!();
    println!("Defaults:");
    println!("  models_dir   = {}", registry.defaults.models_dir);
    println!("  llama_server = {}", registry.defaults.llama_server);
    println!("  base_port    = {}", registry.defaults.base_port);
    println!();
    println!("Point config.toml at the registry with: models_file = \"{output}\"");

    Ok(())
}

fn run_export_registry(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let input = args
        .get(2)
        .ok_or("export-registry: missing input path (models.toml)")?;
    let mut output = json_sibling_path_for_output(input);

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if let Some(path) = args.get(i + 1) {
                    output = path.clone();
                    i += 2;
                } else {
                    return Err("export-registry: missing value for --output".into());
                }
            }
            arg => return Err(format!("export-registry: unknown argument '{arg}'").into()),
        }
    }

    let registry = ModelsRegistry::load(input)?;
    registry.write_json(&output)?;
    println!("Exported {output}");
    Ok(())
}

fn json_sibling_path_for_output(toml_path: &str) -> String {
    if let Some(idx) = toml_path.rfind(".toml") {
        format!("{}json{}", &toml_path[..idx], &toml_path[idx + 5..])
    } else {
        format!("{toml_path}.json")
    }
}
