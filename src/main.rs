use std::sync::Arc;

use tokio::signal;
use tracing::{info, warn};

use std::path::PathBuf;

use gguf_switchboard::api;
use gguf_switchboard::config::{
    Config, ModelsRegistry, cmd_delete_local, cmd_files, cmd_list_local, cmd_pull, cmd_search,
    sync_registry_from_hf,
};
use gguf_switchboard::conformance::ConformanceHistory;
use gguf_switchboard::db::TokenDb;
use gguf_switchboard::metrics;
use gguf_switchboard::scheduler::Scheduler;
use gguf_switchboard::state::AppState;

const CLI_HELP: &str = r#"GGUF Switchboard

One machine. Many local models. One API.
GGUF via llama.cpp, SafeTensors via vLLM.

Usage:
  ggs [<config.toml>]
  ggs <command> [arguments]

Commands:
  ggs models search <query>                 Search GGUF/llama.cpp models
  ggs models search vllm <query>            Search safetensors/vLLM models
  ggs models files <repo-id>                List files in a Hugging Face repository
  ggs models list                           List GGUF/safetensors models on disk
  ggs models delete <name|#>                Delete a model (file + registry entry)
  ggs models pull <repo-id> --quant Q4_K_M  Download and register a GGUF model
  ggs models pull vllm <repo-id>             Download and register a vLLM model
  ggs discover-models <models-dir>           Discover local GGUF models
  ggs sync-hf-metadata                       Refresh Hugging Face metadata
  ggs export-registry <models.toml>          Export a registry as JSON
  ggs status                                 Show whether the system service is running
  ggs stop                                   Stop the system service
  ggs restart                                Restart the system service
  ggs logs                                   Show the latest 100 service log entries
  ggs logs watch                             Watch service logs
  ggs logs --tail 250                        Show the latest 250 service log entries
  ggs version                                Show version information
  ggs <config.toml>                          Start the server with a config file
  ggs help                                   Show this help

Examples:

  # Search for GGUF models that fit your hardware
  ggs models search "Qwen 7B"
  ggs models search nemotron

  # Search for SafeTensors/vLLM models
  ggs models search vllm "Qwen3.5-9B-AWQ"
  ggs models search vllm "Muse"

  # Download and register models
  ggs models pull bartowski/Qwen2.5-7B-Instruct-GGUF --quant Q4_K_M
  ggs models pull vllm Qwen/Qwen2.5-7B-Instruct

  # List and manage models
  ggs models list
  ggs models list --json
  ggs models delete qwen3-embedding-4b
  ggs models delete 2 --yes

  # Discover local models
  ggs discover-models ~/models -o models.toml

  # Service management
  ggs status
  ggs logs
  ggs logs watch
  ggs logs --tail 100
  ggs restart

  # Start the server
  ggs config.toml
"#;

struct CliUsageError(String);

impl std::fmt::Display for CliUsageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::fmt::Debug for CliUsageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for CliUsageError {}

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

    if args
        .get(1)
        .is_some_and(|arg| matches!(arg.as_str(), "help" | "--help" | "-h"))
    {
        print!("{CLI_HELP}");
        return Ok(());
    }

    if args
        .get(1)
        .is_some_and(|arg| matches!(arg.as_str(), "version" | "--version" | "-V"))
    {
        println!("gguf-switchboard {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    if args.get(1).is_some_and(|arg| arg == "model") {
        let corrected = corrected_models_command(&args);
        return Err(cli_error(format!(
            "Unknown command 'model'. Did you mean 'models'?\n\nTry:\n  {corrected}"
        )));
    }

    if args.len() >= 2 && args[1] == "discover-models" {
        return run_discover_models(&args).map_err(add_help_to_cli_usage_error);
    }

    if args.len() >= 2 && args[1] == "sync-hf-metadata" {
        return run_sync_hf_metadata(&args)
            .await
            .map_err(add_help_to_cli_usage_error);
    }

    if args.len() >= 2 && args[1] == "export-registry" {
        return run_export_registry(&args).map_err(add_help_to_cli_usage_error);
    }

    if args.len() >= 2 && args[1] == "models" {
        return run_models_cmd(&args)
            .await
            .map_err(add_help_to_cli_usage_error);
    }

    if args.len() >= 2 && args[1] == "stop" {
        return run_service_ctl("stop");
    }

    if args.len() >= 2 && args[1] == "restart" {
        return run_service_ctl("restart");
    }

    if args.len() >= 2 && args[1] == "status" {
        return run_service_status();
    }

    if args.len() >= 2 && args[1] == "logs" {
        let command = parse_logs_command(&args[2..])?;
        return run_service_command(&command);
    }

    if let Some(command) = args.get(1)
        && !looks_like_config_path(command)
    {
        return Err(cli_error(format!("unknown command '{command}'")));
    }

    let config_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "config.toml".to_string());

    metrics::register_all();

    info!("Loading configuration from {}", config_path);
    let mut config = Config::load(&config_path)?;

    if config.models_file.is_some() {
        match config.sync_hf_metadata().await {
            Ok(summary) => {
                info!(
                    matched = summary.matched,
                    missed = summary.missed,
                    skipped = summary.skipped,
                    "HF metadata sync complete during launch"
                );
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "HF metadata sync failed during launch; continuing with local registry"
                );
            }
        }
    }

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

    let conformance_db_path = db_path.with_file_name("conformance.db");
    let conformance_history = Arc::new(ConformanceHistory::open(&conformance_db_path)?);

    let scheduler = Arc::new(Scheduler::new(config.clone()).await?);
    let watcher_handles = scheduler.start_watchers();
    let app_state = Arc::new(AppState::new(
        config.clone(),
        scheduler.clone(),
        token_db,
        conformance_history,
    ));

    let rescan_cancel = tokio_util::sync::CancellationToken::new();
    let rescan_handle = app_state.spawn_models_rescan_watcher(rescan_cancel.clone());

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
    rescan_cancel.cancel();
    if let Some(handle) = rescan_handle {
        let _ = handle.await;
    }
    watcher_handles.shutdown().await;
    scheduler.shutdown().await?;

    info!("GGUF Switchboard stopped");
    Ok(())
}

/// Name of the systemd unit installed by deploy.sh.
const SERVICE_NAME: &str = "gguf-switchboard";

const LOGS_USAGE: &str = r#"Examples:
  ggs logs
  ggs logs watch
  ggs logs --tail 100"#;

#[derive(Debug, PartialEq, Eq)]
struct ServiceCommand {
    program: &'static str,
    args: Vec<String>,
}

impl ServiceCommand {
    #[cfg(any(target_os = "linux", test))]
    fn command_line_for(&self, is_root: bool) -> Vec<&str> {
        let command = std::iter::once(self.program)
            .chain(self.args.iter().map(String::as_str))
            .collect::<Vec<_>>();
        if is_root {
            command
        } else {
            std::iter::once("sudo").chain(command).collect()
        }
    }
}

fn parse_logs_command(args: &[String]) -> Result<ServiceCommand, Box<dyn std::error::Error>> {
    let journal_args = match args {
        [] => vec!["-u", SERVICE_NAME, "-n", "100", "--no-pager"],
        [arg] if arg == "watch" => vec!["-u", SERVICE_NAME, "-f"],
        [flag, value] if flag == "--tail" => {
            let tail = value
                .parse::<usize>()
                .map_err(|_| format!("logs: --tail requires a positive integer\n\n{LOGS_USAGE}"))?;
            if tail == 0 {
                return Err(
                    format!("logs: --tail requires a positive integer\n\n{LOGS_USAGE}").into(),
                );
            }
            vec!["-u", SERVICE_NAME, "-n", value, "--no-pager"]
        }
        [flag] if flag == "--tail" => {
            return Err(format!(
                "logs: missing value for --tail; expected a positive integer\n\n{LOGS_USAGE}"
            )
            .into());
        }
        _ => {
            return Err(format!("logs: invalid arguments\n\n{LOGS_USAGE}").into());
        }
    };

    Ok(ServiceCommand {
        program: "journalctl",
        args: journal_args.into_iter().map(str::to_string).collect(),
    })
}

fn run_service_status() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_os = "linux"))]
    {
        Err("'status' manages the systemd service and is only supported on Linux".into())
    }

    #[cfg(target_os = "linux")]
    {
        use std::process::Command;

        let status = Command::new("systemctl")
            .args(["is-active", "--quiet", SERVICE_NAME])
            .status()
            .map_err(|error| format!("failed to invoke systemctl: {error}"))?;

        let state = if status.success() {
            "running"
        } else {
            "stopped"
        };
        println!("{SERVICE_NAME}: {state}");
        Ok(())
    }
}

fn run_service_command(command: &ServiceCommand) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = command;
        Err("'logs' reads the systemd journal and is only supported on Linux".into())
    }

    #[cfg(target_os = "linux")]
    {
        use std::process::Command;

        let command_line = command.command_line_for(nix::unistd::Uid::effective().is_root());
        let (program, args) = command_line
            .split_first()
            .expect("service command must include a program");
        let status = Command::new(program)
            .args(args)
            .status()
            .map_err(|error| format!("failed to invoke {program}: {error}"))?;

        if !status.success() {
            return Err(format!(
                "{} failed (exit {})",
                command_line.join(" "),
                status
                    .code()
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            )
            .into());
        }

        Ok(())
    }
}

/// Handles `stop` / `restart`: shells out to `systemctl` to control the
/// gguf-switchboard system service, prefixing with `sudo` when not already
/// running as root.
fn run_service_ctl(action: &str) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(unix))]
    {
        let _ = action;
        return Err(
            "'stop'/'restart' manage the systemd service and are only supported on Linux".into(),
        );
    }

    #[cfg(unix)]
    {
        use std::process::Command;

        let is_root = nix::unistd::Uid::effective().is_root();

        let status = if is_root {
            Command::new("systemctl")
                .arg(action)
                .arg(SERVICE_NAME)
                .status()
        } else {
            Command::new("sudo")
                .arg("systemctl")
                .arg(action)
                .arg(SERVICE_NAME)
                .status()
        }
        .map_err(|e| format!("failed to invoke systemctl: {e}"))?;

        if !status.success() {
            return Err(format!(
                "systemctl {action} {SERVICE_NAME} failed (exit {})",
                status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            )
            .into());
        }

        println!("{SERVICE_NAME}: {action} OK");
        Ok(())
    }
}

fn run_discover_models(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut models_dir: Option<String> = None;
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
                models_dir = Some(path.to_string());
                i += 1;
            }
        }
    }

    let merge_registry = match merge_from.as_deref() {
        Some(path) => Some(ModelsRegistry::load(path)?),
        None => None,
    };

    let result = ModelsRegistry::rescan(
        models_dir.as_deref(),
        merge_registry.as_ref(),
        "llama.cpp",
        12,
    )?;
    result.registry.write(&output)?;

    let discovered_count = result.total;
    let models_dir_display = result.models_dir.clone();

    if discovered_count == 0 {
        println!(
            "Warning: no llama.cpp-loadable .gguf files found under {models_dir_display}; wrote empty registry"
        );
    } else {
        println!("Discovered {discovered_count} model(s) in {models_dir_display}");
    }
    println!("Wrote {output}");
    let json_output = json_sibling_path_for_output(&output);
    println!("Wrote {json_output}");
    if let Some(ref merge_path) = merge_from {
        println!("Merged customizations from {merge_path}");
    }
    println!();
    println!("  {:<24} {:<6} FILE", "ALIAS", "PRI");
    for entry in &result.registry.models {
        let pri = if entry.priority { "yes" } else { "" };
        println!("  {:<24} {:<6} {}", entry.alias, pri, entry.file);
    }
    println!();
    println!("Defaults:");
    println!("  models_dir   = {}", result.registry.defaults.models_dir);
    println!("  llama_server = {}", result.registry.defaults.llama_server);
    println!("  base_port    = {}", result.registry.defaults.base_port);
    println!();
    println!("Point config.toml at the registry with: models_file = \"{output}\"");

    Ok(())
}

async fn run_sync_hf_metadata(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut input = "models.toml".to_string();
    let mut output: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if let Some(path) = args.get(i + 1) {
                    output = Some(path.clone());
                    i += 2;
                } else {
                    return Err("sync-hf-metadata: missing value for --output".into());
                }
            }
            arg if arg.starts_with('-') => {
                return Err(format!("sync-hf-metadata: unknown flag '{arg}'").into());
            }
            path => {
                input = path.to_string();
                i += 1;
            }
        }
    }

    let output = output.unwrap_or_else(|| input.clone());
    let mut registry = ModelsRegistry::load(&input)?;
    let summary = sync_registry_from_hf(&mut registry).await?;
    registry.write(&output)?;

    println!(
        "HF sync: matched={} missed={} skipped={}",
        summary.matched, summary.missed, summary.skipped
    );
    println!("Wrote {output}");
    println!("Wrote {}", json_sibling_path_for_output(&output));
    println!();
    println!("  {:<24} {:<10} {:<8} HF_REPO", "ALIAS", "KIND", "VRAM_GB");
    for entry in &registry.models {
        println!(
            "  {:<24} {:<10} {:<8} {}",
            entry.alias,
            entry.effective_kind(),
            entry.min_vram_gb.map(|v| v.to_string()).unwrap_or_default(),
            entry.hf_repo.as_deref().unwrap_or("")
        );
    }
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

async fn run_models_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let sub = args
        .get(2)
        .ok_or("models: missing subcommand (search, files, pull)")?;

    let sub_args: Vec<String> = args[2..].to_vec();
    match sub.as_str() {
        "help" | "--help" | "-h" => {
            print!("{CLI_HELP}");
            Ok(())
        }
        "search" => cmd_search(&sub_args).await,
        "files" => cmd_files(&sub_args).await,
        "pull" => cmd_pull(&sub_args).await,
        "list" => cmd_list_local(&sub_args).await,
        "delete" => cmd_delete_local(&sub_args).await,
        other => Err(format!(
            "models: unknown subcommand '{other}'\n\nUsage:\n  gguf-switchboard models search <query> [--limit N]\n  gguf-switchboard models search vllm <query> [--limit N]\n  gguf-switchboard models files <repo-id>\n  gguf-switchboard models list [--dir PATH] [--registry models.toml] [--json]\n  gguf-switchboard models delete <name|#> [--yes] [--dir PATH] [--registry models.toml]\n  gguf-switchboard models pull <repo-id> [--quant QUANT] [--dir PATH] [--connections N] [--no-bench]\n  gguf-switchboard models pull vllm <repo-id> [--dir PATH] [--draft <repo>] [--num-speculative-tokens N] [--attention-backend NAME] [--tensor-parallel-size N] [--gpu-memory-utilization F] [--served-model-name NAME] [--force]"
        ).into()),
    }
}

fn json_sibling_path_for_output(toml_path: &str) -> String {
    if let Some(idx) = toml_path.rfind(".toml") {
        format!("{}.json{}", &toml_path[..idx], &toml_path[idx + 5..])
    } else {
        format!("{toml_path}.json")
    }
}

fn cli_error(message: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    Box::new(CliUsageError(format!("{message}\n\n{CLI_HELP}")))
}

fn looks_like_config_path(arg: &str) -> bool {
    arg.ends_with(".toml")
        || arg.contains(std::path::MAIN_SEPARATOR)
        || std::path::Path::new(arg).is_file()
}

fn corrected_models_command(args: &[String]) -> String {
    std::iter::once("ggs".to_string())
        .chain(std::iter::once("models".to_string()))
        .chain(args.iter().skip(2).map(|arg| shell_display_arg(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_display_arg(arg: &str) -> String {
    if arg
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-._/:".contains(ch))
    {
        arg.to_string()
    } else {
        format!("\"{}\"", arg.replace('"', "\\\""))
    }
}

fn add_help_to_cli_usage_error(error: Box<dyn std::error::Error>) -> Box<dyn std::error::Error> {
    let message = error.to_string();
    if is_cli_usage_error(&message) {
        cli_error(message)
    } else {
        error
    }
}

fn is_cli_usage_error(message: &str) -> bool {
    message.starts_with("discover-models: unknown flag")
        || message.starts_with("discover-models: missing value")
        || message.starts_with("sync-hf-metadata: unknown flag")
        || message.starts_with("sync-hf-metadata: missing value")
        || message.starts_with("export-registry: unknown argument")
        || message.starts_with("export-registry: missing input path")
        || message.starts_with("export-registry: missing value")
        || message.starts_with("models: ")
        || (message.starts_with("models search")
            && (message.contains("missing")
                || message.contains("invalid value")
                || message.contains("unknown flag")))
        || (message.starts_with("models files") && message.contains("missing"))
        || message.starts_with("models list:")
        || message.starts_with("models delete:")
        || (message.starts_with("models pull")
            && (message.contains("missing")
                || message.contains("unknown flag")
                || message.contains("must be")
                || message.contains("is required")))
}

#[cfg(test)]
mod service_command_tests {
    use super::*;

    #[test]
    fn logs_defaults_to_the_latest_hundred_entries() {
        let command = parse_logs_command(&[]).expect("default logs command should parse");
        assert_eq!(
            command.command_line_for(true),
            [
                "journalctl",
                "-u",
                "gguf-switchboard",
                "-n",
                "100",
                "--no-pager"
            ]
        );
    }

    #[test]
    fn logs_watch_follows_the_service_journal() {
        let command =
            parse_logs_command(&["watch".to_string()]).expect("watch logs command should parse");
        assert_eq!(
            command.command_line_for(true),
            ["journalctl", "-u", "gguf-switchboard", "-f"]
        );
    }

    #[test]
    fn logs_use_sudo_for_non_root_callers() {
        let command = parse_logs_command(&[]).expect("default logs command should parse");
        assert_eq!(
            command.command_line_for(false),
            [
                "sudo",
                "journalctl",
                "-u",
                "gguf-switchboard",
                "-n",
                "100",
                "--no-pager"
            ]
        );
        assert_eq!(
            command.command_line_for(true),
            [
                "journalctl",
                "-u",
                "gguf-switchboard",
                "-n",
                "100",
                "--no-pager"
            ]
        );
    }

    #[test]
    fn logs_tail_uses_the_requested_positive_line_count() {
        let command = parse_logs_command(&["--tail".to_string(), "250".to_string()])
            .expect("tail logs command should parse");
        assert_eq!(
            command.command_line_for(true),
            [
                "journalctl",
                "-u",
                "gguf-switchboard",
                "-n",
                "250",
                "--no-pager"
            ]
        );
    }

    #[test]
    fn logs_tail_rejects_missing_zero_non_numeric_and_unknown_arguments() {
        for args in [
            vec!["--tail"],
            vec!["--tail", "0"],
            vec!["--tail", "abc"],
            vec!["follow"],
        ] {
            let args = args.into_iter().map(str::to_string).collect::<Vec<_>>();
            let error = parse_logs_command(&args)
                .expect_err("invalid logs arguments should be rejected")
                .to_string();
            assert!(error.contains("Examples:"), "{error}");
            assert!(error.contains("ggs logs watch"), "{error}");
            assert!(error.contains("ggs logs --tail 100"), "{error}");
        }
    }
}
