//! Detect truncated or damaged GGUF files before a backend is asked to load them.
//!
//! llama-server reports a short file as `tensor '...' data is not within the file
//! bounds` only *after* the scheduler has already unloaded the resident model, so
//! every request against a broken file costs a full unload → fail → reload cycle
//! of a healthy model. Reading the header is cheap (tens of ms, no weights), and
//! the tensor table tells us exactly how many bytes the file must contain.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

// Sanity ceilings so a damaged header cannot drive a huge allocation or loop.
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KV_PAIRS: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIMS: u32 = 8;

/// Why a GGUF file failed the integrity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufIntegrityError {
    Unreadable(String),
    NotGguf,
    /// The header or tensor table is cut off or internally inconsistent.
    HeaderDamaged(String),
    /// The tensor data extends past the end of the file.
    Truncated {
        file_bytes: u64,
        required_bytes: u64,
    },
}

impl fmt::Display for GgufIntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable(e) => write!(f, "cannot read file: {e}"),
            Self::NotGguf => write!(f, "not a GGUF file (bad magic)"),
            Self::HeaderDamaged(what) => write!(f, "GGUF header is damaged ({what})"),
            Self::Truncated {
                file_bytes,
                required_bytes,
            } => write!(
                f,
                "file is {} but its tensors need at least {} (short by {}) — \
                 incomplete download or damaged file",
                human(*file_bytes),
                human(*required_bytes),
                human(required_bytes.saturating_sub(*file_bytes)),
            ),
        }
    }
}

impl std::error::Error for GgufIntegrityError {}

fn human(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.1} MiB", bytes as f64 / MIB)
}

/// `(elements per block, bytes per block)` for a ggml tensor type, when known.
fn type_layout(ggml_type: u32) -> Option<(u64, u64)> {
    Some(match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        _ => return None,
    })
}

struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self, what: &str) -> Result<[u8; N], GgufIntegrityError> {
        let mut buf = [0u8; N];
        self.inner
            .read_exact(&mut buf)
            .map_err(|_| GgufIntegrityError::HeaderDamaged(format!("ends inside {what}")))?;
        Ok(buf)
    }

    fn u32(&mut self, what: &str) -> Result<u32, GgufIntegrityError> {
        Ok(u32::from_le_bytes(self.bytes::<4>(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64, GgufIntegrityError> {
        Ok(u64::from_le_bytes(self.bytes::<8>(what)?))
    }

    fn skip(&mut self, len: u64, what: &str) -> Result<(), GgufIntegrityError> {
        let copied = std::io::copy(&mut (&mut self.inner).take(len), &mut std::io::sink())
            .map_err(|e| GgufIntegrityError::Unreadable(e.to_string()))?;
        if copied != len {
            return Err(GgufIntegrityError::HeaderDamaged(format!(
                "ends inside {what}"
            )));
        }
        Ok(())
    }

    fn string_len(&mut self, what: &str) -> Result<u64, GgufIntegrityError> {
        let len = self.u64(what)?;
        if len > MAX_STRING_BYTES {
            return Err(GgufIntegrityError::HeaderDamaged(format!(
                "implausible string length in {what}"
            )));
        }
        Ok(len)
    }

    fn string(&mut self, what: &str) -> Result<Vec<u8>, GgufIntegrityError> {
        let len = self.string_len(what)?;
        let mut buf = vec![0u8; len as usize];
        self.inner
            .read_exact(&mut buf)
            .map_err(|_| GgufIntegrityError::HeaderDamaged(format!("ends inside {what}")))?;
        Ok(buf)
    }

    /// Skip one metadata value; returns it as `u64` when it is a small integer.
    fn value(&mut self, value_type: u32, depth: u8) -> Result<Option<u64>, GgufIntegrityError> {
        const W: &str = "a metadata value";
        Ok(match value_type {
            0 | 1 | 7 => {
                self.skip(1, W)?;
                None
            }
            2 | 3 => {
                self.skip(2, W)?;
                None
            }
            4 => Some(u64::from(self.u32(W)?)),
            5 | 6 => {
                self.skip(4, W)?;
                None
            }
            10..=12 => {
                self.skip(8, W)?;
                None
            }
            8 => {
                let len = self.string_len(W)?;
                self.skip(len, W)?;
                None
            }
            9 => {
                if depth > 2 {
                    return Err(GgufIntegrityError::HeaderDamaged(
                        "array nesting too deep".into(),
                    ));
                }
                let element_type = self.u32(W)?;
                let count = self.u64(W)?;
                // Fixed-width element types can be skipped in one read.
                let fixed = match element_type {
                    0 | 1 | 7 => Some(1),
                    2 | 3 => Some(2),
                    4..=6 => Some(4),
                    10..=12 => Some(8),
                    _ => None,
                };
                if let Some(width) = fixed {
                    let total = count.checked_mul(width).ok_or_else(|| {
                        GgufIntegrityError::HeaderDamaged("implausible array length".into())
                    })?;
                    self.skip(total, W)?;
                } else {
                    for _ in 0..count {
                        self.value(element_type, depth + 1)?;
                    }
                }
                None
            }
            other => {
                return Err(GgufIntegrityError::HeaderDamaged(format!(
                    "unknown metadata type {other}"
                )));
            }
        })
    }
}

/// Verify that `path` is a complete GGUF file: readable header and tensor table,
/// and a file long enough to hold every tensor the table describes.
///
/// Tensor types this build does not know about are counted as one byte, so an
/// unfamiliar quantization can only make the check more lenient, never reject a
/// good file.
pub fn check_gguf_integrity(path: &Path) -> Result<(), GgufIntegrityError> {
    let file = File::open(path).map_err(|e| GgufIntegrityError::Unreadable(e.to_string()))?;
    let file_len = file
        .metadata()
        .map_err(|e| GgufIntegrityError::Unreadable(e.to_string()))?
        .len();
    let mut reader = Reader {
        inner: BufReader::with_capacity(256 * 1024, file),
    };

    if reader.u32("the magic")? != GGUF_MAGIC {
        return Err(GgufIntegrityError::NotGguf);
    }
    let version = reader.u32("the version")?;
    if !(2..=3).contains(&version) {
        return Err(GgufIntegrityError::HeaderDamaged(format!(
            "unsupported version {version}"
        )));
    }
    let tensor_count = reader.u64("the tensor count")?;
    let kv_count = reader.u64("the metadata count")?;
    if tensor_count > MAX_TENSORS || kv_count > MAX_KV_PAIRS {
        return Err(GgufIntegrityError::HeaderDamaged(
            "implausible tensor or metadata count".into(),
        ));
    }

    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..kv_count {
        let key = reader.string("a metadata key")?;
        let value_type = reader.u32("a metadata type")?;
        let value = reader.value(value_type, 0)?;
        if key == b"general.alignment"
            && let Some(a) = value.filter(|a| *a > 0)
        {
            alignment = a;
        }
    }

    // Tensor table: the furthest byte any tensor reaches, relative to data start.
    let mut furthest: u64 = 0;
    for _ in 0..tensor_count {
        let name_len = reader.string_len("a tensor name")?;
        reader.skip(name_len, "a tensor name")?;
        let n_dims = reader.u32("a tensor shape")?;
        if n_dims > MAX_DIMS {
            return Err(GgufIntegrityError::HeaderDamaged(
                "implausible tensor rank".into(),
            ));
        }
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = reader.u64("a tensor shape")?;
            elements = elements.checked_mul(dim).ok_or_else(|| {
                GgufIntegrityError::HeaderDamaged("implausible tensor size".into())
            })?;
        }
        let ggml_type = reader.u32("a tensor type")?;
        let offset = reader.u64("a tensor offset")?;
        let bytes = match type_layout(ggml_type) {
            Some((block, size)) => elements.div_ceil(block).saturating_mul(size),
            None => 1,
        };
        furthest = furthest.max(offset.saturating_add(bytes));
    }

    // Position after the tensor table is where the (aligned) data section begins.
    let table_end = reader
        .inner
        .stream_position()
        .map_err(|e| GgufIntegrityError::Unreadable(e.to_string()))?;
    let data_start = table_end.div_ceil(alignment) * alignment;
    let required = data_start.saturating_add(furthest);

    if file_len < required {
        return Err(GgufIntegrityError::Truncated {
            file_bytes: file_len,
            required_bytes: required,
        });
    }
    Ok(())
}

/// The `-m` / `--model` path in a llama-server argument list.
pub fn model_path_from_args(args: &[String]) -> Option<&str> {
    args.windows(2)
        .find(|w| w[0] == "-m" || w[0] == "--model")
        .map(|w| w[1].as_str())
}

/// Recursively check every `.gguf` under `dir`. Returns `(path, result)` sorted by path.
pub fn scan_dir(dir: &Path) -> Vec<(PathBuf, Result<(), GgufIntegrityError>)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(dir, &mut files);
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let result = check_gguf_integrity(&p);
            (p, result)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn put_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend((s.len() as u64).to_le_bytes());
        buf.extend(s.as_bytes());
    }

    /// One F32 tensor of `elements` values plus a Q4_K tensor of 256 values.
    fn gguf_bytes(alignment_kv: Option<u32>, with_data: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(GGUF_MAGIC.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // tensors
        let kvs = 1 + u64::from(alignment_kv.is_some());
        b.extend(kvs.to_le_bytes());
        put_string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        put_string(&mut b, "qwen3");
        if let Some(a) = alignment_kv {
            put_string(&mut b, "general.alignment");
            b.extend(4u32.to_le_bytes());
            b.extend(a.to_le_bytes());
        }
        // tensor 0: 64 x F32 = 256 bytes @ 0
        put_string(&mut b, "a");
        b.extend(1u32.to_le_bytes());
        b.extend(64u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        // tensor 1: 256 x Q4_K = 144 bytes @ 256
        put_string(&mut b, "b");
        b.extend(1u32.to_le_bytes());
        b.extend(256u64.to_le_bytes());
        b.extend(12u32.to_le_bytes());
        b.extend(256u64.to_le_bytes());
        let align = u64::from(alignment_kv.unwrap_or(32));
        let padded = (b.len() as u64).div_ceil(align) * align;
        b.resize(padded as usize, 0);
        if with_data {
            b.extend(vec![0u8; 256 + 144]);
        }
        b
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        path
    }

    #[test]
    fn complete_file_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "ok.gguf", &gguf_bytes(None, true));
        assert_eq!(check_gguf_integrity(&path), Ok(()));
    }

    #[test]
    fn honours_custom_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "a.gguf", &gguf_bytes(Some(64), true));
        assert_eq!(check_gguf_integrity(&path), Ok(()));
    }

    #[test]
    fn file_missing_its_tail_is_truncated() {
        // The production failure: header intact, weights cut short.
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = gguf_bytes(None, true);
        bytes.truncate(bytes.len() - 100);
        let path = write(dir.path(), "short.gguf", &bytes);
        match check_gguf_integrity(&path) {
            Err(GgufIntegrityError::Truncated {
                file_bytes,
                required_bytes,
            }) => assert_eq!(required_bytes - file_bytes, 100),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn header_only_file_is_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "hdr.gguf", &gguf_bytes(None, false));
        assert!(matches!(
            check_gguf_integrity(&path),
            Err(GgufIntegrityError::Truncated { .. })
        ));
    }

    #[test]
    fn cut_inside_the_tensor_table_is_header_damage() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = gguf_bytes(None, false);
        bytes.truncate(90);
        let path = write(dir.path(), "cut.gguf", &bytes);
        assert!(matches!(
            check_gguf_integrity(&path),
            Err(GgufIntegrityError::HeaderDamaged(_))
        ));
    }

    #[test]
    fn non_gguf_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "x.gguf", b"not a gguf file at all, sorry");
        assert_eq!(
            check_gguf_integrity(&path),
            Err(GgufIntegrityError::NotGguf)
        );
    }

    #[test]
    fn absurd_counts_do_not_allocate() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = Vec::new();
        b.extend(GGUF_MAGIC.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(u64::MAX.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        let path = write(dir.path(), "big.gguf", &b);
        assert!(matches!(
            check_gguf_integrity(&path),
            Err(GgufIntegrityError::HeaderDamaged(_))
        ));
    }

    #[test]
    fn missing_file_is_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            check_gguf_integrity(&dir.path().join("nope.gguf")),
            Err(GgufIntegrityError::Unreadable(_))
        ));
    }

    #[test]
    fn model_path_is_read_from_args() {
        let args: Vec<String> = ["--host", "x", "-m", "/m/a.gguf", "-c", "1"]
            .map(String::from)
            .to_vec();
        assert_eq!(model_path_from_args(&args), Some("/m/a.gguf"));
        assert_eq!(model_path_from_args(&[]), None);
    }

    #[test]
    fn scan_dir_reports_each_file() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "good.gguf", &gguf_bytes(None, true));
        let mut bad = gguf_bytes(None, true);
        bad.truncate(bad.len() - 1);
        write(dir.path(), "bad.gguf", &bad);
        write(dir.path(), "readme.txt", b"ignored");
        let results = scan_dir(dir.path());
        assert_eq!(results.len(), 2);
        assert!(results[0].0.ends_with("bad.gguf") && results[0].1.is_err());
        assert!(results[1].0.ends_with("good.gguf") && results[1].1.is_ok());
    }
}
