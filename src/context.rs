//! Helpers for reading and adjusting llama-server context size (`-c`) at runtime.

const CONTEXT_FLAGS: &[&str] = &["-c", "--ctx-size", "--context-size"];

/// Read the configured context size from backend args, if present.
pub fn get_context_size(args: &[String]) -> Option<u32> {
    let (_, value_idx) = context_value_index(args)?;
    args.get(value_idx)?.parse().ok().filter(|&n| n > 0)
}

/// Return a copy of `args` with the context flag value set to `size`.
pub fn with_context_size(args: &[String], size: u32) -> Vec<String> {
    let mut updated = args.to_vec();
    if let Some((_, value_idx)) = context_value_index(&updated) {
        updated[value_idx] = size.to_string();
        return updated;
    }

    updated.push("-c".to_string());
    updated.push(size.to_string());
    updated
}

/// Halve the context size for the next load attempt, stopping at `min`.
pub fn next_lower_context(current: u32, min: u32) -> Option<u32> {
    if current <= min {
        return None;
    }

    let halved = current / 2;
    let next = halved.max(min);
    if next >= current {
        return None;
    }

    Some(next)
}

/// Compute the context size for a given attempt using configured step ratios.
///
/// `steps` are descending ratios (e.g. `[1.0, 0.75, 0.5, 0.25]`).  The
/// `attempt` is zero-indexed.  The result is clamped to `minimum`.
///
/// Returns `None` when `attempt` exceeds the available steps.
pub fn context_for_attempt(
    requested: u32,
    minimum: u32,
    steps: &[f64],
    attempt: usize,
) -> Option<u32> {
    let ratio = steps.get(attempt)?;
    let ctx = (requested as f64 * ratio).floor() as u32;
    Some(ctx.max(minimum.max(512)))
}

fn context_value_index(args: &[String]) -> Option<(usize, usize)> {
    for (idx, arg) in args.iter().enumerate() {
        if CONTEXT_FLAGS.contains(&arg.as_str()) {
            return Some((idx, idx + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_context_flag() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-c".to_string(),
            "65536".to_string(),
        ];
        assert_eq!(get_context_size(&args), Some(65536));
    }

    #[test]
    fn updates_existing_context_flag() {
        let args = vec!["-c".to_string(), "65536".to_string()];
        let updated = with_context_size(&args, 32768);
        assert_eq!(get_context_size(&updated), Some(32768));
    }

    #[test]
    fn appends_context_flag_when_missing() {
        let args = vec!["-m".to_string(), "model.gguf".to_string()];
        let updated = with_context_size(&args, 16384);
        assert_eq!(get_context_size(&updated), Some(16384));
    }

    #[test]
    fn halves_context_until_min() {
        assert_eq!(next_lower_context(65536, 8192), Some(32768));
        assert_eq!(next_lower_context(32768, 8192), Some(16384));
        assert_eq!(next_lower_context(16384, 8192), Some(8192));
        assert_eq!(next_lower_context(8192, 8192), None);
    }

    #[test]
    fn context_for_attempt_stepped() {
        let steps = [1.0, 0.75, 0.5, 0.25];
        assert_eq!(context_for_attempt(32768, 4096, &steps, 0), Some(32768));
        assert_eq!(context_for_attempt(32768, 4096, &steps, 1), Some(24576));
        assert_eq!(context_for_attempt(32768, 4096, &steps, 2), Some(16384));
        assert_eq!(context_for_attempt(32768, 4096, &steps, 3), Some(8192));
        assert_eq!(context_for_attempt(32768, 4096, &steps, 4), None);
    }

    #[test]
    fn context_for_attempt_clamped_to_minimum() {
        let steps = [1.0, 0.25];
        // 0.25 * 4096 = 1024, but minimum is 4096
        assert_eq!(context_for_attempt(4096, 4096, &steps, 1), Some(4096));
    }

    #[test]
    fn context_for_attempt_minimum_floor() {
        let steps = [1.0, 0.1];
        // 0.1 * 8192 = 819, which is above both minimum=256 and absolute floor=512
        assert_eq!(context_for_attempt(8192, 256, &steps, 1), Some(819));
        // But with a very small ratio: 0.01 * 8192 = 81, clamped to max(256, 512) = 512
        let tiny_steps = [1.0, 0.01];
        assert_eq!(context_for_attempt(8192, 256, &tiny_steps, 1), Some(512));
    }
}
