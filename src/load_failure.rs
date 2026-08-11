//! Classify model startup failures to decide whether context reduction applies.

/// Kind of failure observed while loading a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadFailureKind {
    /// GPU OOM during weight allocation (model too large for VRAM).
    GpuOomWeights,
    /// GPU OOM during KV cache allocation (context too large).
    GpuOomKvCache,
    /// Generic GPU OOM that cannot be sub-classified.
    GpuOomGeneric,
    /// Catch-all for backward compat; prefer the specific variants above.
    Oom,
    PortConflict,
    MissingFile,
    InvalidArgs,
    ProcessExit,
    HealthTimeout,
    Unknown,
}

impl LoadFailureKind {
    /// True when this is any flavour of OOM.
    pub fn is_oom(self) -> bool {
        matches!(
            self,
            Self::GpuOomWeights | Self::GpuOomKvCache | Self::GpuOomGeneric | Self::Oom
        )
    }
}

/// Classify a load failure from the error message and captured stderr.
pub fn classify_load_failure(message: &str, stderr: &str) -> LoadFailureKind {
    let haystack = format!("{message}\n{stderr}").to_lowercase();

    if matches_oom(&haystack) {
        return classify_oom_detail(&haystack);
    }
    if haystack.contains("address already in use")
        || haystack.contains("eaddrinuse")
        || haystack.contains("bind failed")
    {
        return LoadFailureKind::PortConflict;
    }
    if haystack.contains("not found")
        || haystack.contains("no such file")
        || haystack.contains("gguf file not found")
        || haystack.contains("backend binary not found")
    {
        return LoadFailureKind::MissingFile;
    }
    if haystack.contains("invalid argument")
        || haystack.contains("unknown argument")
        || haystack.contains("unrecognized")
    {
        return LoadFailureKind::InvalidArgs;
    }
    if haystack.contains("did not become healthy") || haystack.contains("loading timeout") {
        return LoadFailureKind::HealthTimeout;
    }
    if haystack.contains("process exited") || haystack.contains("backend process exited") {
        return LoadFailureKind::ProcessExit;
    }

    LoadFailureKind::Unknown
}

fn matches_oom(haystack: &str) -> bool {
    const PATTERNS: &[&str] = &[
        "out of memory",
        "cuda error",
        "cudamalloc",
        "failed to allocate",
        "cannot allocate",
        "insufficient memory",
        "insufficient device memory",
        "vk_error_out_of_device_memory",
        "metal: insufficient",
        "ggml_alloc",
        "alloc tensor",
        "oom",
    ];
    PATTERNS.iter().any(|p| haystack.contains(p))
}

/// Sub-classify an OOM into weights vs KV cache vs generic.
fn classify_oom_detail(haystack: &str) -> LoadFailureKind {
    // KV cache OOM patterns — these are safe to retry with smaller context / quantized KV.
    const KV_OOM: &[&str] = &[
        "kv cache",
        "kv_init",
        "failed to allocate buffer for kv",
        "ggml_backend_sched_alloc_splits: failed to allocate",
    ];
    if KV_OOM.iter().any(|p| haystack.contains(p)) {
        return LoadFailureKind::GpuOomKvCache;
    }

    // Weight allocation OOM patterns — model tensors don't fit.
    const WEIGHTS_OOM: &[&str] = &[
        "failed to allocate cuda",
        "cudamalloc failed",
        "alloc tensor",
        "ggml_alloc",
        "failed to allocate buffer for",
    ];
    if WEIGHTS_OOM.iter().any(|p| haystack.contains(p)) {
        return LoadFailureKind::GpuOomWeights;
    }

    LoadFailureKind::GpuOomGeneric
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_oom_generic() {
        let kind = classify_load_failure("load failed", "CUDA error: out of memory");
        assert!(kind.is_oom());
    }

    #[test]
    fn classifies_oom_kv_cache() {
        let kind =
            classify_load_failure("backend failed", "failed to allocate buffer for kv cache");
        assert_eq!(kind, LoadFailureKind::GpuOomKvCache);
        assert!(kind.is_oom());
    }

    #[test]
    fn classifies_oom_weights() {
        let kind = classify_load_failure("backend failed", "cudamalloc failed: out of memory");
        assert_eq!(kind, LoadFailureKind::GpuOomWeights);
        assert!(kind.is_oom());
    }

    #[test]
    fn classifies_oom_alloc_tensor() {
        let kind =
            classify_load_failure("model load failed", "ggml_alloc: failed to allocate tensor");
        assert_eq!(kind, LoadFailureKind::GpuOomWeights);
    }

    #[test]
    fn classifies_oom_generic_when_unspecific() {
        let kind = classify_load_failure("", "insufficient device memory");
        assert_eq!(kind, LoadFailureKind::GpuOomGeneric);
        assert!(kind.is_oom());
    }

    #[test]
    fn classifies_port_conflict() {
        assert_eq!(
            classify_load_failure("Failed to spawn backend: Address already in use", ""),
            LoadFailureKind::PortConflict
        );
    }

    #[test]
    fn classifies_missing_file() {
        assert_eq!(
            classify_load_failure("Model GGUF file not found: '/tmp/missing.gguf'", ""),
            LoadFailureKind::MissingFile
        );
    }

    #[test]
    fn unknown_does_not_reduce_context() {
        assert_eq!(
            classify_load_failure("some random failure", "bad flag"),
            LoadFailureKind::Unknown
        );
    }
}
