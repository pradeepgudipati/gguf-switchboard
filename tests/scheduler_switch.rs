mod support;

use std::sync::Arc;
use std::time::Duration;

use support::{
    FakeLlamaServer, scheduler_from_config, write_scheduler_config, write_scheduler_config_with,
};

#[tokio::test]
async fn ensure_loaded_reloads_unhealthy_backend() {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config = write_scheduler_config(&fake_a, &fake_b);
    let scheduler = Arc::new(scheduler_from_config(&config).await);

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should load");
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    fake_a.set_healthy(false);
    let reload = {
        let scheduler = Arc::clone(&scheduler);
        tokio::spawn(async move { scheduler.ensure_loaded("model-a").await })
    };

    tokio::time::sleep(Duration::from_millis(300)).await;
    fake_a.set_healthy(true);

    let result = reload.await.expect("join reload");
    assert!(
        result.is_ok(),
        "should reload once health returns: {}",
        result.err().map(|e| e.to_string()).unwrap_or_default()
    );
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    scheduler.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn failed_switch_keeps_previous_model_loaded() {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config = write_scheduler_config(&fake_a, &fake_b);
    let scheduler = Arc::new(scheduler_from_config(&config).await);

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should load");
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    let result = scheduler.ensure_loaded("model-b").await;
    assert!(result.is_err());

    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should still serve requests");
    scheduler.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn active_request_blocks_model_switch() {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config = write_scheduler_config(&fake_a, &fake_b);
    let scheduler = Arc::new(scheduler_from_config(&config).await);

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should load");

    let _guard = scheduler.track_request("model-a");

    let switch = {
        let scheduler = Arc::clone(&scheduler);
        tokio::spawn(async move { scheduler.ensure_loaded("model-b").await })
    };

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !switch.is_finished(),
        "switch should wait while model-a has an active request"
    );

    drop(_guard);

    let result = switch.await.expect("join switch task");
    assert!(result.is_err());
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    scheduler.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn unload_first_switch_reports_rollback_after_failed_target() {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    // Default strategy is unload_first: model-a is stopped before model-b starts,
    // so a failed model-b load must re-load model-a (rollback).
    let config = write_scheduler_config(&fake_a, &fake_b);
    let scheduler = Arc::new(scheduler_from_config(&config).await);

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should load");

    let first = scheduler.last_switch().await.expect("first load recorded");
    assert!(first.ok);
    assert_eq!(first.to, "model-a");
    assert_eq!(first.from, None);
    assert_eq!(first.trigger, "request");
    assert_eq!(first.unload_previous_ms, 0);
    assert_eq!(first.rollback_ms, 0);

    assert!(scheduler.ensure_loaded("model-b").await.is_err());
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    let failed = scheduler
        .last_switch()
        .await
        .expect("failed switch recorded");
    assert!(!failed.ok);
    assert_eq!(failed.from.as_deref(), Some("model-a"));
    assert_eq!(failed.to, "model-b");
    assert!(
        failed.total_ms >= failed.load_ms,
        "total must cover the load phase"
    );

    // Metrics: the switch counter and unload reason should be exported with labels.
    gguf_switchboard::metrics::register_all();
    let body = gguf_switchboard::metrics::gather();
    assert!(
        body.contains("gguf_switchboard_model_switches_total{"),
        "switch counter missing from /metrics output:\n{body}"
    );
    assert!(
        body.contains("to=\"model-b\"") && body.contains("result=\"error\""),
        "failed switch to model-b not labelled:\n{body}"
    );
    assert!(
        body.contains("gguf_switchboard_model_load_seconds_bucket{"),
        "per-model load histogram missing:\n{body}"
    );
    assert!(
        body.contains("gguf_switchboard_loaded_model_info{model=\"model-a\"}"),
        "resident model gauge series for model-a missing:\n{body}"
    );

    scheduler.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn load_first_switch_keeps_previous_resident_without_rollback() {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config = write_scheduler_config_with(&fake_a, &fake_b, "switch_strategy = \"load_first\"");
    let scheduler = Arc::new(scheduler_from_config(&config).await);

    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a should load");
    assert!(scheduler.ensure_loaded("model-b").await.is_err());
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    let failed = scheduler
        .last_switch()
        .await
        .expect("failed switch recorded");
    assert!(!failed.ok);
    assert_eq!(
        failed.rollback_ms, 0,
        "load_first never unloads the previous model, so nothing to roll back"
    );

    scheduler.shutdown().await.expect("shutdown");
}
