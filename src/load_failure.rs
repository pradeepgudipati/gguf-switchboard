//! Classify model startup failures to decide whether context reduction applies.

/// Kind of failure observed while loading a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadFailureKind {
    Oom,
    PortConflict,
    MissingFile,
    InvalidArgs,
    ProcessExit,
    HealthTimeout,
    Unknown,
}

/// Classify a load failure from the error message and captured stderr.
pub fn classify_load_failure(message: &str, stderr: &str) -> LoadFailureKind {
    let haystack = format!("{message}\n{stderr}").to_lowercase();

    if matches_oom(&haystack) {
        return LoadFailureKind::Oom;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_oom() {
        assert_eq!(
            classify_load_failure("load failed", "CUDA error: out of memory"),
            LoadFailureKind::Oom
        );
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
