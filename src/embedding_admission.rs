use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

use crate::errors::RuntimeError;
use crate::metrics::{
    EMBEDDING_QUEUE_DEPTH, EMBEDDING_QUEUE_REJECTED_TOTAL, EMBEDDING_QUEUE_WAIT_SECONDS,
};

pub struct EmbeddingAdmission {
    gates: Mutex<HashMap<String, (usize, Arc<Semaphore>)>>,
    timeout: Duration,
}

impl EmbeddingAdmission {
    pub fn new(timeout: Duration) -> Self {
        Self {
            gates: Mutex::new(HashMap::new()),
            timeout,
        }
    }

    pub async fn acquire(
        &self,
        model_id: &str,
        concurrency: usize,
    ) -> Result<OwnedSemaphorePermit, RuntimeError> {
        let semaphore = {
            let mut gates = self.gates.lock().await;
            let entry = gates.entry(model_id.to_string()).or_insert_with(|| {
                (
                    concurrency.max(1),
                    Arc::new(Semaphore::new(concurrency.max(1))),
                )
            });
            if entry.0 != concurrency.max(1) && entry.1.available_permits() == entry.0 {
                *entry = (
                    concurrency.max(1),
                    Arc::new(Semaphore::new(concurrency.max(1))),
                );
            }
            Arc::clone(&entry.1)
        };

        let started = Instant::now();
        EMBEDDING_QUEUE_DEPTH.with_label_values(&[model_id]).inc();
        let result = tokio::time::timeout(self.timeout, semaphore.acquire_owned()).await;
        EMBEDDING_QUEUE_DEPTH.with_label_values(&[model_id]).dec();
        EMBEDDING_QUEUE_WAIT_SECONDS
            .with_label_values(&[model_id])
            .observe(started.elapsed().as_secs_f64());

        match result {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(RuntimeError::InternalError(
                "embedding admission semaphore closed".to_string(),
            )),
            Err(_) => {
                EMBEDDING_QUEUE_REJECTED_TOTAL
                    .with_label_values(&[model_id])
                    .inc();
                Err(RuntimeError::EmbeddingQueueTimeout {
                    retry_after_secs: self.timeout.as_secs().max(1),
                })
            }
        }
    }
}

pub fn balanced_concurrency(batch_size: Option<u32>) -> usize {
    if batch_size.is_some_and(|batch| batch >= 2048) {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn second_request_times_out_when_single_permit_is_held() {
        let admission = EmbeddingAdmission::new(std::time::Duration::from_millis(10));
        let _first = admission.acquire("embed", 1).await.expect("first permit");

        let error = admission.acquire("embed", 1).await.unwrap_err();
        assert!(matches!(
            error,
            crate::errors::RuntimeError::EmbeddingQueueTimeout { .. }
        ));
    }

    #[test]
    fn balanced_concurrency_is_one_below_large_batch_tier() {
        assert_eq!(balanced_concurrency(Some(1024)), 1);
        assert_eq!(balanced_concurrency(Some(2048)), 2);
    }
}
