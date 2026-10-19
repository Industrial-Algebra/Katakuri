// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Validated checkpoint loading with an explicit trust policy (S5, wishlist
//! W2 — Amari Task 43).
//!
//! 0.1.0's [`SafeTensors::parse`] takes `&[u8]` and is deliberately
//! value-blind: it validates layout, never values. This module is the
//! **scoped and named reversal**: [`load_validated`] opens a file, applies an
//! explicit [`TrustPolicy`], parses through S1 unchanged, and finally may scan
//! values for nonfinite elements. `parse` itself stays value-blind.
//!
//! Feature gates: `std` (file I/O) **and** `digest` (schema/model hash
//! verification). Spec and manifest *types* build without either
//! ([`crate::model::MlpSpec`], [`crate::manifest::TensorManifest`]).
//!
//! # Pipeline (ordered, failure is total)
//!
//! 1. **Regular-file check** (`regular_file_only`): the path must resolve to a
//!    regular file. Symlinks, FIFOs, devices, and directories reject;
//!    hardlinks are regular files and are accepted. Enforced with
//!    `symlink_metadata` + `O_NOFOLLOW` (Unix) + `fstat` on the opened
//!    descriptor.
//! 2. **Byte cap** (`byte_cap`): a file larger than the cap rejects before any
//!    parse work.
//! 3. **Read** the bytes.
//! 4. **S1 parse** — value-blind, full safetensors validation.
//! 5. **Metadata convention** (§5.5): `ia.spec` must be present; the
//!    `ia.provenance.training-data` length bound applies; unknown `ia.*` keys
//!    reject.
//! 6. **Spec** — `ia.spec` parses as an [`MlpSpec`] (strict `ia-mlp`
//!    grammar).
//! 7. **Shape/schema agreement** — the tensor set must equal
//!    [`MlpSpec::expected_tensors`] (names, shapes, dtype F32), with no
//!    unexpected tensors.
//! 8. **Hashes** (`verify_schema_hash`, `verify_model_hash`), before the value
//!    scan: a tampered file reports a hash mismatch, not a value diagnostic.
//! 9. **Nonfinite scan** (`reject_nonfinite`): NaN / ±Infinity in any F32
//!    tensor rejects. Subnormals are finite and are kept — they are legal f32
//!    values; whether they are acceptable is the compute layer's question.
//! 10. **Manifest assembly** in canonical order.
//!
//! No partial load is ever returned: every step either yields the full
//! [`ValidatedCheckpoint`] or an error.
//!
//! [`SafeTensors::parse`]: crate::safetensors::SafeTensors::parse
//! [`MlpSpec::expected_tensors`]: crate::model::MlpSpec::expected_tensors

use crate::digest::artifact_sha256;
use crate::error::{KatakuriError, NonfiniteKind};
use crate::manifest::{
    DeterminismPolicy, ManifestDtype, ManifestEntry, ManifestLayout, TensorManifest,
};
use crate::model::MlpSpec;
use crate::safetensors::{Dtype, SafeTensors};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// `__metadata__` key carrying the [`MlpSpec`] JSON itself.
pub const META_SPEC: &str = "ia.spec";

/// `__metadata__` key carrying the sha256 of the spec's canonical JSON
/// ([`MlpSpec::canonical_json`]).
pub const META_SCHEMA_HASH: &str = "ia.schema-hash";

/// `__metadata__` key carrying the sha256 of the safetensors data section.
pub const META_MODEL_HASH: &str = "ia.model-hash";

/// `__metadata__` key Amari binds its replay/training-data hashes to
/// (free-form string, bounded to `1 ..= 4096` bytes).
pub const META_TRAINING_DATA: &str = "ia.provenance.training-data";

/// The `ia.` metadata namespace: keys outside the S5 convention reject.
const IA_PREFIX: &str = "ia.";

/// Largest accepted `ia.provenance.training-data` value, in bytes.
const MAX_PROVENANCE_BYTES: usize = 4096;

/// Trust policy for checkpoint loading (Amari Task 43). Every field is an
/// explicit choice; nothing is silently on.
#[derive(Clone, Debug)]
pub struct TrustPolicy {
    /// Reject unless the path resolves to a regular file (no symlinks, no
    /// FIFOs, no devices). Implemented via `symlink_metadata` + `O_NOFOLLOW`
    /// on Unix (plus `fstat` on the opened descriptor), and plain metadata
    /// elsewhere. Hardlinks are regular files and are accepted.
    pub regular_file_only: bool,
    /// Reject files larger than this many bytes. Checked before reading; the
    /// read itself is bounded by the same cap.
    pub byte_cap: u64,
    /// Verify `__metadata__["ia.schema-hash"]` (the sha256 of the parsed
    /// spec's canonical JSON) against the parsed spec.
    pub verify_schema_hash: bool,
    /// Verify `__metadata__["ia.model-hash"]` (the sha256 of the safetensors
    /// data section) against the file's data bytes.
    pub verify_model_hash: bool,
    /// **Scan values**: reject NaN / ±Infinity in any F32 tensor.
    ///
    /// This is the deliberate, scoped reversal of S1's "values never
    /// inspected": `parse` stays value-blind; this seam scans, because
    /// Amari's ordering decisions must never see nonfinite scores propagate
    /// (wishlist W3). Subnormals are finite and are KEPT — they are legal
    /// f32 values, and whether the compute layer tolerates them is its
    /// question, not the loader's.
    pub reject_nonfinite: bool,
}

/// A tensor's canonical name, dtype, shape, and raw little-endian element
/// bytes — an owned counterpart of S1's borrowed
/// [`Tensor`](crate::safetensors::Tensor) view, so a [`ValidatedCheckpoint`]
/// owns its data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedTensor {
    /// Canonical parameter name (`"hidden.0.weight"`, `"head.bias"`, …).
    pub name: String,
    /// Element type (F32 for every tensor the loader accepts).
    pub dtype: Dtype,
    /// Shape in row-major dimension order.
    pub shape: Vec<usize>,
    /// Raw element bytes, little-endian, no padding.
    pub data: Vec<u8>,
}

/// A fully validated checkpoint: spec, layout manifest, metadata, and tensors
/// in canonical order.
#[derive(Clone, Debug)]
pub struct ValidatedCheckpoint {
    /// The architecture spec parsed from `__metadata__["ia.spec"]`.
    pub spec: MlpSpec,
    /// The typed layout manifest, entries in canonical parameter order.
    pub manifest: TensorManifest,
    /// The full `__metadata__` string map, provenance included.
    pub metadata: BTreeMap<String, String>,
    /// Parsed tensors in canonical parameter order (the manifest's order).
    pub tensors: Vec<ParsedTensor>,
}

/// Load a safetensors checkpoint and validate it against `policy`.
///
/// The ordered pipeline is documented at the module level; failure is total
/// (no partial loads). The returned tensors are in canonical parameter order,
/// matching [`TensorManifest::entries`].
///
/// # Errors
///
/// - [`KatakuriError::CheckpointIo`] — the path could not be opened or read.
/// - [`KatakuriError::CheckpointNotRegularFile`] — `regular_file_only` is set
///   and the path is a symlink, FIFO, device, or directory.
/// - [`KatakuriError::CheckpointTooLarge`] — the file exceeds `byte_cap`.
/// - Any [`KatakuriError`] from S1's parse (truncated file, bad header, …).
/// - [`KatakuriError::ProvenanceMalformed`] — `ia.spec` absent, or
///   `ia.provenance.training-data` outside `1 ..= 4096` bytes.
/// - [`KatakuriError::UnknownIaMetadata`] — an unrecognized `ia.*` key.
/// - [`KatakuriError::MlpSpecParse`] — `ia.spec` is not a valid `ia-mlp` spec.
/// - [`KatakuriError::WeightLayoutMismatch`] — the tensor set diverges from
///   `expected_tensors()` (missing / unexpected / wrong shape / wrong dtype).
/// - [`KatakuriError::SchemaHashMismatch`] / [`KatakuriError::ModelHashMismatch`]
///   — a declared digest is absent or does not match.
/// - [`KatakuriError::NonfiniteTensorValue`] — `reject_nonfinite` is set and
///   an F32 tensor holds NaN or ±Infinity.
pub fn load_validated(
    path: &Path,
    policy: &TrustPolicy,
) -> Result<ValidatedCheckpoint, KatakuriError> {
    let bytes = read_checkpoint(path, policy)?;
    let parsed = SafeTensors::parse(&bytes)?;
    let metadata = parsed.metadata().clone();

    check_metadata_convention(&metadata)?;

    let spec_json = metadata
        .get(META_SPEC)
        .ok_or_else(|| KatakuriError::ProvenanceMalformed {
            key: META_SPEC.to_string(),
        })?;
    let spec = MlpSpec::from_json(spec_json)?;

    validate_layout(&spec, &parsed)?;

    if policy.verify_schema_hash {
        let actual = artifact_sha256(spec.canonical_json().as_bytes());
        let expected = metadata
            .get(META_SCHEMA_HASH)
            .map_or("<absent>", String::as_str);
        if expected != actual {
            return Err(KatakuriError::SchemaHashMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
    }
    if policy.verify_model_hash {
        let actual = artifact_sha256(data_section(&bytes)?);
        let expected = metadata
            .get(META_MODEL_HASH)
            .map_or("<absent>", String::as_str);
        if expected != actual {
            return Err(KatakuriError::ModelHashMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
    }

    if policy.reject_nonfinite {
        scan_nonfinite(&spec, &parsed)?;
    }

    let mut entries = Vec::with_capacity(2 + 2 * spec.hidden.len());
    let mut tensors = Vec::with_capacity(2 + 2 * spec.hidden.len());
    for (name, shape) in spec.expected_tensors() {
        let tensor = parsed.tensor(&name)?;
        let (start, end) = tensor.data_offsets();
        entries.push(ManifestEntry {
            name: name.clone(),
            // Parse-valid widths are `<= 16384`, so the narrowing cannot fail.
            shape: shape.iter().map(|&dim| dim as u32).collect(),
            byte_offset: start as u64,
            byte_len: (end - start) as u64,
        });
        tensors.push(ParsedTensor {
            name,
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            data: tensor.bytes().to_vec(),
        });
    }

    Ok(ValidatedCheckpoint {
        spec,
        manifest: TensorManifest {
            entries,
            dtype: ManifestDtype::F32,
            layout: ManifestLayout::RowMajorContiguous,
            determinism: DeterminismPolicy::pinned(),
        },
        metadata,
        tensors,
    })
}

/// Apply the trust policy's file checks and read the bytes.
fn read_checkpoint(path: &Path, policy: &TrustPolicy) -> Result<Vec<u8>, KatakuriError> {
    if policy.regular_file_only {
        ensure_regular_file(path)?;
    }
    let file = open_checkpoint(path, policy.regular_file_only)?;
    let meta = file
        .metadata()
        .map_err(|e| io_error(path, &e.to_string()))?;
    if policy.regular_file_only {
        // Re-check on the opened descriptor: the path could have been swapped
        // between the metadata check and the open.
        if !meta.file_type().is_file() {
            return Err(KatakuriError::CheckpointNotRegularFile {
                path: path.display().to_string(),
            });
        }
    }
    let size = meta.len();
    if size > policy.byte_cap {
        return Err(KatakuriError::CheckpointTooLarge {
            size,
            cap: policy.byte_cap,
        });
    }

    let mut bytes = Vec::new();
    // Read at most one byte past the cap: a file that grew after the metadata
    // check is caught by the length check below rather than silently
    // truncated into a possibly-valid shorter file.
    let read = file
        .take(policy.byte_cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| io_error(path, &e.to_string()))?;
    if read as u64 > policy.byte_cap {
        return Err(KatakuriError::CheckpointTooLarge {
            size: read as u64,
            cap: policy.byte_cap,
        });
    }
    Ok(bytes)
}

fn io_error(path: &Path, detail: &str) -> KatakuriError {
    KatakuriError::CheckpointIo {
        path: path.display().to_string(),
        detail: detail.to_string(),
    }
}

/// `regular_file_only` pre-check: `symlink_metadata` does not follow the final
/// component, so a symlink, FIFO, device, or directory is visible here.
/// Hardlinks resolve to a regular file and pass.
fn ensure_regular_file(path: &Path) -> Result<(), KatakuriError> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| io_error(path, &e.to_string()))?;
    if !meta.file_type().is_file() {
        return Err(KatakuriError::CheckpointNotRegularFile {
            path: path.display().to_string(),
        });
    }
    Ok(())
}

/// Open the checkpoint, adding `O_NOFOLLOW` on Unix when the policy demands a
/// regular file (closing the symlink race between the metadata check and the
/// open). Platforms without a known `O_NOFOLLOW` value fall back to the
/// metadata + `fstat` checks.
fn open_checkpoint(path: &Path, nofollow: bool) -> Result<File, KatakuriError> {
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    if nofollow {
        use std::os::unix::fs::OpenOptionsExt;
        if let Some(flag) = unix_nofollow_flag() {
            options.custom_flags(flag);
        }
    }
    #[cfg(not(unix))]
    let _ = nofollow;

    match options.open(path) {
        Ok(file) => Ok(file),
        Err(e) => {
            // `O_NOFOLLOW` surfaces as `ELOOP`; re-state it as the policy
            // failure it is. Any other error stays an I/O error.
            if nofollow {
                if let Ok(meta) = std::fs::symlink_metadata(path) {
                    if !meta.file_type().is_file() {
                        return Err(KatakuriError::CheckpointNotRegularFile {
                            path: path.display().to_string(),
                        });
                    }
                }
            }
            Err(io_error(path, &e.to_string()))
        }
    }
}

/// `O_NOFOLLOW` for the target ABI, where a known value exists.
///
/// The value is not exposed by `std`, and pulling in `libc` for one constant
/// would add a dependency for a platform table. Values match the `libc`
/// crate (each OS's `fcntl.h`):
///
/// - Linux/Android (asm-generic): `0o400000` = 0x20000
/// - macOS/iOS (Darwin): `0x10000`
/// - FreeBSD: `0x20000`
/// - NetBSD/OpenBSD: `0x400000`
///
/// DragonFly is deliberately NOT mapped — its value was not verified against
/// its `fcntl.h`, so it takes the documented fallback (`symlink_metadata`
/// pre-check + post-open `fstat`) rather than a guessed flag. A wrong value
/// here is worse than none: it both fails to reject symlinks AND sets an
/// unrelated open flag on that ABI.
#[cfg(unix)]
fn unix_nofollow_flag() -> Option<i32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        Some(0o400000)
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(0x0001_0000)
    }
    #[cfg(target_os = "freebsd")]
    {
        Some(0x0002_0000)
    }
    #[cfg(any(target_os = "netbsd", target_os = "openbsd"))]
    {
        Some(0x0040_0000)
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd"
    )))]
    {
        None
    }
}

/// Validate the `ia.*` metadata convention (S5 §5.5). Only `ia.*` keys are
/// governed: metadata outside the namespace is the writer's business.
fn check_metadata_convention(metadata: &BTreeMap<String, String>) -> Result<(), KatakuriError> {
    for (key, value) in metadata {
        match key.as_str() {
            META_SPEC | META_SCHEMA_HASH | META_MODEL_HASH => {}
            META_TRAINING_DATA => {
                let len = value.len();
                // Comparisons only: `len == 0 || len > MAX`.
                if len == 0 || len > MAX_PROVENANCE_BYTES {
                    return Err(KatakuriError::ProvenanceMalformed { key: key.clone() });
                }
            }
            _ if key.starts_with(IA_PREFIX) => {
                return Err(KatakuriError::UnknownIaMetadata { key: key.clone() });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Check the tensor set against [`MlpSpec::expected_tensors`], collecting
/// every divergence (the S3 diagnostics-first style).
fn validate_layout(spec: &MlpSpec, parsed: &SafeTensors<'_>) -> Result<(), KatakuriError> {
    let expected = spec.expected_tensors();
    let expected_names: BTreeSet<&str> = expected.iter().map(|(name, _)| name.as_str()).collect();
    let mut problems: Vec<String> = Vec::new();

    for (name, shape) in &expected {
        match parsed.tensor(name) {
            Ok(tensor) => {
                if tensor.dtype() != Dtype::F32 {
                    problems.push(format!(
                        "tensor '{name}': wrong dtype (expected F32, found {:?})",
                        tensor.dtype()
                    ));
                }
                if tensor.shape() != shape.as_slice() {
                    problems.push(format!(
                        "tensor '{name}': wrong shape (expected {shape:?}, found {:?})",
                        tensor.shape()
                    ));
                }
            }
            Err(_) => problems.push(format!("missing tensor '{name}' (expected {shape:?})")),
        }
    }

    for name in parsed.names() {
        if !expected_names.contains(name) {
            problems.push(format!("unexpected tensor '{name}'"));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(KatakuriError::WeightLayoutMismatch {
            detail: problems.join("\n"),
        })
    }
}

/// Scan every expected F32 tensor for NaN / ±Infinity, in canonical order.
/// Subnormals are finite and pass.
fn scan_nonfinite(spec: &MlpSpec, parsed: &SafeTensors<'_>) -> Result<(), KatakuriError> {
    for (name, _shape) in spec.expected_tensors() {
        let tensor = parsed.tensor(&name)?;
        for chunk in tensor.bytes().chunks_exact(4) {
            let Ok(raw) = <[u8; 4]>::try_from(chunk) else {
                // `chunks_exact(4)` yields exactly four bytes; unreachable.
                continue;
            };
            let value = f32::from_le_bytes(raw);
            let which = if value.is_nan() {
                NonfiniteKind::Nan
            } else if value == f32::INFINITY {
                NonfiniteKind::PositiveInfinity
            } else if value == f32::NEG_INFINITY {
                NonfiniteKind::NegativeInfinity
            } else {
                continue;
            };
            return Err(KatakuriError::NonfiniteTensorValue {
                tensor: name,
                which,
            });
        }
    }
    Ok(())
}

/// The safetensors data section of a file that S1 has already parsed.
///
/// S1 does not expose the raw data slice, and the format's prefix is fixed:
/// 8-byte little-endian header length, then header, then data.
fn data_section(bytes: &[u8]) -> Result<&[u8], KatakuriError> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|slice| slice.try_into().ok())
        .ok_or(KatakuriError::TruncatedFile {
            file_len: bytes.len(),
        })?;
    let header_len = u64::from_le_bytes(prefix);
    let header_len =
        usize::try_from(header_len).map_err(|_| KatakuriError::HeaderLengthExceedsFile {
            header_len,
            available: bytes.len(),
        })?;
    let start = 8usize
        .checked_add(header_len)
        .ok_or(KatakuriError::HeaderLengthExceedsFile {
            header_len: u64::try_from(header_len).unwrap_or(u64::MAX),
            available: bytes.len(),
        })?;
    bytes
        .get(start..)
        .ok_or(KatakuriError::HeaderLengthExceedsFile {
            header_len: u64::try_from(header_len).unwrap_or(u64::MAX),
            available: bytes.len(),
        })
}
