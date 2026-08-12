//! Helpers for reading and adjusting llama-server batch size (`-b`) and micro-batch size (`-ub`).

const BATCH_FLAGS: &[&str] = &["-b", "--batch-size", "--batch"];
const UBATCH_FLAGS: &[&str] = &["-ub", "--ubatch-size", "--ubatch"];

/// Read the configured batch size from backend args, if present.
pub fn get_batch_size(args: &[String]) -> Option<u32> {
    let (_, value_idx) = batch_value_index(args)?;
    args.get(value_idx)?.parse().ok().filter(|&n| n > 0)
}

/// Return a copy of `args` with the batch size flag value set to `size`.
pub fn with_batch_size(args: &[String], size: u32) -> Vec<String> {
    let mut updated = args.to_vec();
    if let Some((_, value_idx)) = batch_value_index(&updated) {
        updated[value_idx] = size.to_string();
        return updated;
    }

    updated.push("-b".to_string());
    updated.push(size.to_string());
    updated
}

/// Read the configured micro-batch size from backend args, if present.
pub fn get_ubatch_size(args: &[String]) -> Option<u32> {
    let (_, value_idx) = ubatch_value_index(args)?;
    args.get(value_idx)?.parse().ok().filter(|&n| n > 0)
}

/// Return a copy of `args` with the micro-batch size flag value set to `size`.
pub fn with_ubatch_size(args: &[String], size: u32) -> Vec<String> {
    let mut updated = args.to_vec();
    if let Some((_, value_idx)) = ubatch_value_index(&updated) {
        updated[value_idx] = size.to_string();
        return updated;
    }

    updated.push("-ub".to_string());
    updated.push(size.to_string());
    updated
}

/// Compute default batch and micro-batch sizes for embedding models.
///
/// Returns `(batch_size, ubatch_size)` suitable for Nomic and similar embedding models.
/// The defaults are chosen to handle large inputs without exceeding physical limits:
/// - batch_size: 2048 (logical batch size for token processing)
/// - ubatch_size: 2048 (physical micro-batch size for prompt processing)
pub fn embedding_batch_defaults() -> (u32, u32) {
    (2048, 2048)
}

/// Check if the args already have batch/ubatch flags configured.
pub fn has_batch_flags(args: &[String]) -> bool {
    get_batch_size(args).is_some() || get_ubatch_size(args).is_some()
}

fn batch_value_index(args: &[String]) -> Option<(usize, usize)> {
    for (idx, arg) in args.iter().enumerate() {
        if BATCH_FLAGS.contains(&arg.as_str()) && idx + 1 < args.len() {
            return Some((idx, idx + 1));
        }
    }
    None
}

fn ubatch_value_index(args: &[String]) -> Option<(usize, usize)> {
    for (idx, arg) in args.iter().enumerate() {
        if UBATCH_FLAGS.contains(&arg.as_str()) && idx + 1 < args.len() {
            return Some((idx, idx + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_batch_flag() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-b".to_string(),
            "2048".to_string(),
        ];
        assert_eq!(get_batch_size(&args), Some(2048));
    }

    #[test]
    fn reads_batch_size_flag() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "--batch-size".to_string(),
            "1024".to_string(),
        ];
        assert_eq!(get_batch_size(&args), Some(1024));
    }

    #[test]
    fn reads_ubatch_flag() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-ub".to_string(),
            "512".to_string(),
        ];
        assert_eq!(get_ubatch_size(&args), Some(512));
    }

    #[test]
    fn reads_ubatch_size_flag() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "--ubatch-size".to_string(),
            "256".to_string(),
        ];
        assert_eq!(get_ubatch_size(&args), Some(256));
    }

    #[test]
    fn updates_existing_batch_flag() {
        let args = vec!["-b".to_string(), "1024".to_string()];
        let updated = with_batch_size(&args, 2048);
        assert_eq!(get_batch_size(&updated), Some(2048));
    }

    #[test]
    fn updates_existing_ubatch_flag() {
        let args = vec!["-ub".to_string(), "512".to_string()];
        let updated = with_ubatch_size(&args, 1024);
        assert_eq!(get_ubatch_size(&updated), Some(1024));
    }

    #[test]
    fn appends_batch_flag_when_missing() {
        let args = vec!["-m".to_string(), "model.gguf".to_string()];
        let updated = with_batch_size(&args, 2048);
        assert_eq!(get_batch_size(&updated), Some(2048));
    }

    #[test]
    fn appends_ubatch_flag_when_missing() {
        let args = vec!["-m".to_string(), "model.gguf".to_string()];
        let updated = with_ubatch_size(&args, 1024);
        assert_eq!(get_ubatch_size(&updated), Some(1024));
    }

    #[test]
    fn embedding_defaults_are_reasonable() {
        let (batch, ubatch) = embedding_batch_defaults();
        assert_eq!(batch, 2048);
        assert_eq!(ubatch, 2048);
        assert!(ubatch <= batch);
    }

    #[test]
    fn detects_batch_flags_present() {
        let with_batch = vec!["-b".to_string(), "2048".to_string()];
        assert!(has_batch_flags(&with_batch));

        let with_ubatch = vec!["-ub".to_string(), "1024".to_string()];
        assert!(has_batch_flags(&with_ubatch));

        let neither = vec!["-c".to_string(), "8192".to_string()];
        assert!(!has_batch_flags(&neither));
    }

    #[test]
    fn returns_none_when_no_batch_flag() {
        let args = vec!["-m".to_string(), "model.gguf".to_string()];
        assert_eq!(get_batch_size(&args), None);
        assert_eq!(get_ubatch_size(&args), None);
    }

    #[test]
    fn rejects_zero_batch_size() {
        let args = vec!["-b".to_string(), "0".to_string()];
        assert_eq!(get_batch_size(&args), None);
    }
}
