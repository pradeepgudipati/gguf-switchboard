//! Best-effort host CPU / RAM telemetry for the `/status` endpoint.
//!
//! Backed by [`sysinfo`]. A single [`System`] is kept alive behind a mutex so
//! that CPU usage is measured as a delta between refreshes; results are cached
//! for a short TTL so Swagger UI polling does not resample on every request.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use sysinfo::{MemoryRefreshKind, RefreshKind, System};

/// Host CPU / memory sample.
#[derive(Debug, Clone, Default)]
pub struct HostStats {
    /// Aggregate CPU utilization across all logical cores, 0-100.
    pub cpu_util_pct: u32,
    /// Logical CPU count.
    pub cpu_count: usize,
    pub mem_total_mb: u64,
    pub mem_used_mb: u64,
    /// `mem_used_mb / mem_total_mb`, rounded, 0-100.
    pub mem_used_pct: u32,
}

fn system() -> &'static Mutex<System> {
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    SYS.get_or_init(|| {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        // Prime CPU usage so the first real sample has a baseline to diff against.
        sys.refresh_cpu_usage();
        Mutex::new(sys)
    })
}

/// Sample host CPU / RAM, memoised for `ttl`.
pub fn host_stats_cached(ttl: Duration) -> HostStats {
    static CACHE: OnceLock<Mutex<Option<(Instant, HostStats)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));

    if let Ok(guard) = cache.lock() {
        if let Some((at, stats)) = guard.as_ref() {
            if at.elapsed() < ttl {
                return stats.clone();
            }
        }
    }

    let stats = sample();
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((Instant::now(), stats.clone()));
    }
    stats
}

fn sample() -> HostStats {
    let Ok(mut sys) = system().lock() else {
        return HostStats::default();
    };
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    let cpu_count = sys.cpus().len();
    let cpu_util_pct = sys.global_cpu_usage().round().clamp(0.0, 100.0) as u32;
    let mem_total_mb = sys.total_memory() / (1024 * 1024);
    let mem_used_mb = sys.used_memory() / (1024 * 1024);
    let mem_used_pct = if mem_total_mb > 0 {
        ((mem_used_mb as f64 / mem_total_mb as f64) * 100.0).round() as u32
    } else {
        0
    };

    HostStats {
        cpu_util_pct,
        cpu_count,
        mem_total_mb,
        mem_used_mb,
        mem_used_pct,
    }
}
