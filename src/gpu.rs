//! Best-effort free VRAM probe (NVIDIA via nvidia-smi).

use tracing::debug;

/// Per-GPU device information from nvidia-smi.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub total_mb: u64,
    pub free_mb: u64,
    pub used_mb: u64,
}

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

/// Probe all NVIDIA GPUs for detailed per-device info.
/// Returns an empty vec when nvidia-smi is unavailable.
pub fn probe_all_gpus() -> Vec<GpuDeviceInfo> {
    probe_all_gpus_from_nvidia_smi().unwrap_or_default()
}

fn probe_all_gpus_from_nvidia_smi() -> Option<Vec<GpuDeviceInfo>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_nvidia_smi_all_gpus(&stdout)
}

fn parse_nvidia_smi_all_gpus(stdout: &str) -> Option<Vec<GpuDeviceInfo>> {
    let mut gpus = Vec::new();
    for line in stdout.lines().map(str::trim).filter(|l| !l.is_empty()) {
        // Expected format: "0, NVIDIA GeForce RTX 4090, 24564, 22000, 2564"
        // Tolerate spaces around commas and extra whitespace.
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 5 {
            continue;
        }
        let index = parts[0].parse::<usize>().ok()?;
        let name = parts[1].to_string();
        let total_mb = parse_csv_u64(parts[2])?;
        let free_mb = parse_csv_u64(parts[3]).unwrap_or(0);
        let used_mb = parse_csv_u64(parts[4]).unwrap_or(0);
        gpus.push(GpuDeviceInfo {
            index,
            name,
            total_mb,
            free_mb,
            used_mb,
        });
    }
    if gpus.is_empty() {
        return None;
    }
    Some(gpus)
}

/// Parse a single CSV token that may contain trailing units (e.g. "24564 MiB" → 24564).
fn parse_csv_u64(token: &str) -> Option<u64> {
    token
        .split_whitespace()
        .next()
        .and_then(|t| t.parse::<u64>().ok())
        .filter(|&n| n > 0)
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

    #[test]
    fn parse_all_gpus_single() {
        let stdout = "0, NVIDIA GeForce RTX 4090, 24564, 22000, 2564\n";
        let gpus = parse_nvidia_smi_all_gpus(stdout).unwrap();
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].total_mb, 24564);
        assert_eq!(gpus[0].free_mb, 22000);
        assert_eq!(gpus[0].used_mb, 2564);
    }

    #[test]
    fn parse_all_gpus_multi() {
        let stdout = "0, RTX 4090, 24564, 22000, 2564\n1, RTX 4090, 24564, 11000, 13564\n";
        let gpus = parse_nvidia_smi_all_gpus(stdout).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].free_mb, 11000);
    }

    #[test]
    fn parse_all_gpus_with_units() {
        let stdout = "0, RTX 3090, 24576 MiB, 20480 MiB, 4096 MiB\n";
        let gpus = parse_nvidia_smi_all_gpus(stdout).unwrap();
        assert_eq!(gpus[0].total_mb, 24576);
        assert_eq!(gpus[0].free_mb, 20480);
    }

    #[test]
    fn parse_all_gpus_empty() {
        assert!(parse_nvidia_smi_all_gpus("").is_none());
        assert!(parse_nvidia_smi_all_gpus("\n\n").is_none());
    }

    #[test]
    fn parse_all_gpus_malformed_line_skipped() {
        let stdout = "bad line\n0, RTX 4090, 24564, 22000, 2564\n";
        let gpus = parse_nvidia_smi_all_gpus(stdout).unwrap();
        assert_eq!(gpus.len(), 1);
    }
}
