mod support;

use std::io::Write;
use std::sync::Arc;

use gguf_switchboard::config::Config;
use gguf_switchboard::scheduler::Scheduler;
use support::FakeLlamaServer;

/// A GGUF whose header promises 4 KiB of tensor data that is not there.
fn truncated_gguf() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(0x4655_4747u32.to_le_bytes()); // magic
    b.extend(3u32.to_le_bytes()); // version
    b.extend(1u64.to_le_bytes()); // tensors
    b.extend(0u64.to_le_bytes()); // kv pairs
    b.extend(1u64.to_le_bytes()); // name length
    b.extend(b"t");
    b.extend(1u32.to_le_bytes()); // dims
    b.extend(1024u64.to_le_bytes()); // 1024 x F32
    b.extend(0u32.to_le_bytes()); // type F32
    b.extend(0u64.to_le_bytes()); // offset
    b
}

#[tokio::test]
async fn truncated_gguf_is_refused_without_unloading_the_resident_model() {
    let fake_a = FakeLlamaServer::start().await;
    let dir = tempfile::TempDir::new().expect("temp dir");
    let gguf = dir.path().join("broken.gguf");
    std::fs::File::create(&gguf)
        .expect("create gguf")
        .write_all(&truncated_gguf())
        .expect("write gguf");

    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!(
            r#"
bind = "127.0.0.1:9090"
startup_timeout = 10
default_backend = "llama.cpp"

[models.model-a]
backend = "llama.cpp"
display_name = "Model A"
command = "sleep"
args = ["3600"]
backend_url = "{a_backend}"
health_url = "{a_health}"

[models.broken]
backend = "llama.cpp"
display_name = "Broken"
command = "sleep"
args = ["-m", "{gguf}"]
backend_url = "{a_backend}"
health_url = "{a_health}"
"#,
            a_backend = fake_a.backend_url,
            a_health = fake_a.health_url,
            gguf = gguf
                .display()
                .to_string()
                .replace(std::path::MAIN_SEPARATOR, "/"),
        ),
    )
    .expect("write config");

    let config = Config::load(config_path.to_str().expect("utf8")).expect("load config");
    let scheduler = Arc::new(Scheduler::new(config).await.expect("scheduler"));

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should load");
    let before = scheduler.last_switch().await.expect("first load recorded");

    let Err(err) = scheduler.ensure_loaded("broken").await else {
        panic!("truncated GGUF must be refused");
    };
    let message = err.to_string();
    assert!(
        message.contains("not a complete GGUF"),
        "message: {message}"
    );
    assert!(message.contains("not touched"), "message: {message}");

    // The resident model was never drained, unloaded or reloaded.
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));
    let after = scheduler.last_switch().await.expect("switch report");
    assert_eq!(after.to, before.to);
    assert_eq!(after.finished_at, before.finished_at);
    assert_eq!(after.rollback_ms, 0);

    scheduler.shutdown().await.expect("shutdown");
}
