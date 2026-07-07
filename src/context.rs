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
}
