//! Best-effort free VRAM probe (NVIDIA via nvidia-smi).

use tracing::debug;

/// Free VRAM on the first GPU in megabytes, if queryable.
pub fn free_vram_mb() -> Option<u64> {
    free_vram_mb_from_nvidia_smi().or_else(|| {
        debug!("nvidia-smi free VRAM unavailable; caller should fall back to vram_gb");
        None
    })
}

/// Total VRAM across all NVIDIA GPUs in megabytes, if queryable.
pub fn total_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_nvidia_smi_total_mb(&String::from_utf8_lossy(&output.stdout))
}

fn free_vram_mb_from_nvidia_smi() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_nvidia_smi_free_mb(&String::from_utf8_lossy(&output.stdout))
}

/// Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` stdout.
/// Uses the first GPU line only.
pub fn parse_nvidia_smi_free_mb(stdout: &str) -> Option<u64> {
    let line = stdout.lines().map(str::trim).find(|l| !l.is_empty())?;
    // Tolerate trailing units if a driver variant includes them.
    let token = line.split_whitespace().next()?;
    token.parse::<u64>().ok().filter(|&n| n > 0).or_else(|| {
        // Some outputs are "12345 MiB"
        line.split_whitespace()
            .next()
            .and_then(|t| t.parse::<u64>().ok())
    })
}

/// Parse total VRAM rows from `nvidia-smi`, summing all valid GPUs.
pub fn parse_nvidia_smi_total_mb(stdout: &str) -> Option<u64> {
    let mut total = 0_u64;
    let mut found = false;
    for line in stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let Some(value) = line
            .split_whitespace()
            .next()
            .and_then(|token| token.parse::<u64>().ok())
            .filter(|value| *value > 0)
        else {
            continue;
        };
        total = total.checked_add(value)?;
        found = true;
    }
    found.then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_total_single_gpu() {
        assert_eq!(parse_nvidia_smi_total_mb("8192\n"), Some(8192));
    }

    #[test]
    fn parse_nvidia_smi_total_sums_multiple_gpus() {
        assert_eq!(parse_nvidia_smi_total_mb("8192\n24576\n"), Some(32768));
    }

    #[test]
    fn parse_nvidia_smi_total_ignores_malformed_lines() {
        assert_eq!(
            parse_nvidia_smi_total_mb("not supported\n16384 MiB\n"),
            Some(16384)
        );
        assert_eq!(parse_nvidia_smi_total_mb("unknown\n0\n"), None);
    }

    #[test]
    fn parse_nvidia_smi_single_gpu() {
        assert_eq!(parse_nvidia_smi_free_mb("10240\n"), Some(10240));
        assert_eq!(parse_nvidia_smi_free_mb("  8192  \n"), Some(8192));
    }

    #[test]
    fn parse_nvidia_smi_multi_gpu_uses_first() {
        assert_eq!(parse_nvidia_smi_free_mb("4096\n2048\n"), Some(4096));
    }

    #[test]
    fn parse_nvidia_smi_empty() {
        assert_eq!(parse_nvidia_smi_free_mb(""), None);
        assert_eq!(parse_nvidia_smi_free_mb("\n\n"), None);
    }
}
