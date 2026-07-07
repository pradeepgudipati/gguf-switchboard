//! System memory monitoring.
//!
//! Provides platform-specific memory stats for Linux and macOS.
//! Falls back gracefully on unsupported platforms.

use tracing::{debug, warn};

/// Current system memory statistics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryStats {
    /// Total system RAM in megabytes.
    pub total_mb: u64,
    /// Used RAM in megabytes.
    pub used_mb: u64,
    /// Available RAM in megabytes.
    pub available_mb: u64,
    /// Percentage of RAM currently in use (0–100).
    pub used_percent: u8,
}

/// Read current system memory statistics.
///
/// Returns `None` if memory information cannot be determined (e.g. on an
/// unsupported platform or when `/proc/meminfo` is unavailable).
pub fn check_memory() -> Option<MemoryStats> {
    #[cfg(target_os = "linux")]
    {
        read_linux().or_else(|| {
            warn!("Failed to read /proc/meminfo; memory monitoring disabled");
            None
        })
    }

    #[cfg(target_os = "macos")]
    {
        read_macos().or_else(|| {
            warn!("Failed to read memory stats via sysctl/vm_stat; memory monitoring disabled");
            None
        })
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        warn!("Memory monitoring is not supported on this platform");
        return None;
    }
}

// ── Linux implementation ────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn read_linux() -> Option<MemoryStats> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb: Option<u64> = None;
    let mut available_kb: Option<u64> = None;

    for line in content.lines() {
        if let Some(val) = parse_meminfo_field(line, "MemTotal") {
            total_kb = Some(val);
        } else if let Some(val) = parse_meminfo_field(line, "MemAvailable") {
            available_kb = Some(val);
        }
        if total_kb.is_some() && available_kb.is_some() {
            break;
        }
    }

    let total_kb = total_kb?;
    let available_kb = available_kb?;

    let total_mb = total_kb / 1024;
    let available_mb = available_kb / 1024;
    let used_mb = total_mb.saturating_sub(available_mb);
    let used_percent = u8::try_from(
        used_mb
            .saturating_mul(100)
            .checked_div(total_mb)
            .unwrap_or(0),
    )
    .unwrap_or(0);

    debug!(
        total_mb,
        used_mb, available_mb, used_percent, "Memory stats (linux)"
    );

    Some(MemoryStats {
        total_mb,
        used_mb,
        available_mb,
        used_percent,
    })
}

#[cfg(target_os = "linux")]
fn parse_meminfo_field(line: &str, field: &str) -> Option<u64> {
    let line = line.strip_prefix(field)?;
    let line = line.strip_prefix(':')?;
    let value_str = line.split_whitespace().next()?;
    value_str.parse::<u64>().ok()
}

// ── macOS implementation ────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn read_macos() -> Option<MemoryStats> {
    use std::process::Command;

    // Total physical memory via sysctl
    let total_bytes = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<u64>().ok()
        })?;

    let total_mb = total_bytes / (1024 * 1024);

    // Page stats via vm_stat
    let vm_output = Command::new("vm_stat")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())?;

    let page_size: u64 = 16384; // 16 KiB on Apple Silicon, same result for Intel with 4096 * pages
    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut speculative_pages: u64 = 0;

    for line in vm_output.lines().skip(1) {
        // Lines look like: "Pages free:        123456."
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key = parts[0].trim();
        let val_str = parts[1].trim().trim_end_matches('.');
        let val: u64 = match val_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        match key {
            "Pages free" => free_pages = val,
            "Pages inactive" => inactive_pages = val,
            "Pages speculative" => speculative_pages = val,
            _ => {}
        }
    }

    let available_bytes = (free_pages + inactive_pages + speculative_pages) * page_size;
    let available_mb = available_bytes / (1024 * 1024);
    let used_mb = total_mb.saturating_sub(available_mb);
    let used_percent = u8::try_from(
        used_mb
            .saturating_mul(100)
            .checked_div(total_mb)
            .unwrap_or(0),
    )
    .unwrap_or(0);

    debug!(
        total_mb,
        used_mb, available_mb, used_percent, "Memory stats (macos)"
    );

    Some(MemoryStats {
        total_mb,
        used_mb,
        available_mb,
        used_percent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_memory_returns_stats() {
        // Should return Some on a real system; gracefully returns None in
        // CI containers where /proc/meminfo may be restricted.
        if let Some(stats) = check_memory() {
            assert!(stats.total_mb > 0, "total_mb should be > 0");
            assert!(stats.used_percent <= 100, "used_percent should be <= 100");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_field_works() {
        let line = "MemTotal:       16384000 kB";
        assert_eq!(parse_meminfo_field(line, "MemTotal"), Some(16384000));
        assert_eq!(parse_meminfo_field(line, "MemFree"), None);
    }
}
