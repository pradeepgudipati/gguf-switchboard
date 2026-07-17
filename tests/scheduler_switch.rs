mod support;

use std::sync::Arc;
use std::time::Duration;

use support::{FakeLlamaServer, scheduler_from_config, write_scheduler_config};

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
