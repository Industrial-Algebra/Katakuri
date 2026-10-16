# Katakuri S1 — Safetensors Parse/Validate → Zunesha

**Date:** 2026-10-16
**Unit:** S1 (safetensors) — first implementation slice
**Depends on:** repo scaffold (PR #1, merged; develop @ c099500)
**Followed by:** S2 (tokenizer), S3 (model specs), S4 (hub)
**Plan contract for generation dispatch** (ia-moment AUTHOR stage)

---

## 0. Context for the implementer

Katakuri is IA's model-ecosystem layer over Zunesha. It **never computes** —
this slice implements safetensors *parsing and validation only*, plus a
device-upload seam. No forward passes, no math on tensor contents.

Consumers (Ijima first) parse a safetensors file, look up named tensors with
dtype/shape metadata, and read the raw bytes (host path — GPU-less hardware is
a first-class requirement, not a fallback). When a Zunesha device exists, the
same bytes can be uploaded into a `zunesha::Buffer`.

The Borsalino verification pattern applies (operator directive 2026-10-16):
a `verify` feature carrying karpal-verify obligation bundles for the parse
invariants, plus `#[cfg(kani)]` proof harnesses for the validation arithmetic.
Both are additive and inert in default builds.

Format reference (huggingface/safetensors): file = 8-byte little-endian u64
header length N, then N bytes of JSON header, then the data byte buffer.
Header keys are tensor names (or `__metadata__`); each tensor entry is
`{"dtype": str, "shape": [int...], "data_offsets": [start, end]}` with offsets
relative to the start of the data region. Spans may have gaps between tensors
but must not overlap (Katakuri's validation contract; accepts every
contiguous real-world file, rejects corrupted ones).

## 1. Unit

Implement `katakuri::safetensors`: `SafeTensors::parse(&[u8]) -> validated
manifest borrowing the input`, per-tensor `Tensor` views, dtype/shape/offset
validation, `__metadata__` extraction, and `upload_to_device` over the
`zunesha::Device` trait. Plus the Borsalino-pattern verification surface
(`verify` feature: obligation bundles; `cfg(kani)`: proof harnesses).

## 2. Files

Create or replace EXACTLY these files (no others; Cargo.lock excepted):

1. Replace `Cargo.toml` — full content in §3.1
2. Replace `src/lib.rs` — full content in §3.2
3. Replace `src/error.rs` — full content in §3.3
4. Replace `src/safetensors.rs` — full content in §3.4
5. Create `src/verify.rs` — full content in §3.5
6. Create `src/kani_harnesses.rs` — full content in §3.6
7. Create `tests/real_file.rs` — full content in §3.7
8. Replace `CHANGELOG.md` — full content in §3.8

Do NOT touch: `src/tokenizer.rs`, `src/model.rs`, `src/hub.rs`, `LICENSE`,
`CONTRIBUTING.md`, `README.md`, `rust-toolchain.toml`, `docs/plans/*`
(this plan file included).

## 3. Contracts

Each fenced block is the **complete final file content**, verbatim. When prose
and a Contract disagree, the Contract wins.

### 3.1 `Cargo.toml`

```toml
[package]
name = "katakuri"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
description = "Model ecosystem layer over Zunesha — safetensors weight loading, WordPiece tokenization, architecture specs, and hub access. Never computes."
license = "Apache-2.0"
repository = "https://github.com/Industrial-Algebra/Katakuri"
keywords = ["safetensors", "tokenization", "wordpiece", "model-weights", "bert"]
categories = ["science", "parser-implementations"]
readme = "README.md"

[features]
default = ["std"]
std = []
# Borsalino-pattern verification obligations (karpal stack).
verify = ["dep:karpal-verify", "dep:karpal-proof"]

[dependencies]
thiserror = "2"
# Header parsing: the safetensors header IS JSON; serde_json is the
# ecosystem-standard, dependency-light parser (serde/itoa/ryu/memchr only).
serde_json = "1"
# The device substrate: Buffer upload + the NoDeviceStub host path.
zunesha = "0.1.2"
# Verification stack, feature-gated per the Borsalino pattern.
karpal-verify = { version = "0.6.1", optional = true }
karpal-proof = { version = "0.6.1", optional = true }

# NOTE: Katakuri never computes — no forward passes, no autodiff, no optimizers,
# no kernels, no shader compilation. That boundary is the architectural contract
# (docs/plans/2026-10-16-katakuri-architecture-handover.md §4.1).
#
# Dependency ledger (no silent growth — every entry justified in the PR):
# - thiserror 2: central error enum (IA standard, same as Zunesha)
# - serde_json 1: safetensors header parse (the format is JSON)
# - zunesha 0.1.2: device upload + NoDevice host path (the substrate)
# - karpal-verify/proof 0.6.1: verify-feature obligations (Borsalino pattern,
#   pinned to Borsalino's own pin)
```

### 3.2 `src/lib.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! # Katakuri — Model Ecosystem Layer
//!
//! > The one who sees ahead — the weights are ready before the compute begins.
//!
//! Katakuri is the model-ecosystem layer over
//! [Zunesha](https://github.com/Industrial-Algebra/Zunesha): safetensors weight
//! loading, BERT-class tokenization, architecture specs, and hub access. It
//! exists so the IA ecosystem does not inherit a heavy ML framework's
//! dependency risk for the parts that are *not* tensor math.
//!
//! ## The hard boundary: Katakuri never computes
//!
//! No forward implementations, no autodiff, no kernels, no shader compilation.
//! Katakuri stops precisely where tensor math starts: it loads, validates, and
//! describes the data consumers compute over. Consumers implement forward
//! passes on their tensor engine of choice — exact CPU math today,
//! Borsalino-class GPU compute later. The same weights feed either path;
//! Katakuri takes no side. This is the structural form of the workhorse /
//! precision doctrine: Borsalino is the workhorse, Amari is the precision.
//!
//! ## Layering over Zunesha
//!
//! Weight data lands in `zunesha::Buffer` when a device exists, and in host
//! memory via Zunesha's `NoDevice` path otherwise — GPU-less hardware (Ijima's
//! deployment class) is a first-class requirement, not a fallback. Katakuri is
//! a *consumer* of Zunesha, never a wrapper of it: buffers flow through, and
//! alignment and lifecycle obligations remain Zunesha's (its `verify` feature
//! bundles the karpal-verify obligations; Katakuri inherits, never restates).
//!
//! ## Modules
//!
//! - [`safetensors`] — weight parse + validate (dtype / shape / layout /
//!   manifest) into borrowed host memory or an uploaded `zunesha::Buffer`.
//!   Replaces candle's `VarBuilder` and safetensors glue. Slice S1.
//! - [`tokenizer`] — WordPiece (BERT-class) tokenization matching Ijima's
//!   exact tokenization contract, with golden-vector parity against HF
//!   tokenizers. Replaces the `tokenizers` crate dependency. Lands with S2.
//! - [`model`] — architecture specs as typed data: BERT config (layers /
//!   heads / dims / hidden act), named-tensor layout manifests. Replaces
//!   candle-transformers `models::bert`. Lands with S3.
//! - [`hub`] — model fetch + content-addressed cache over plain HTTPS,
//!   offline-first. Replaces `hf-hub`. Optional feature; lands with S4.
//!
//! ## Features
//!
//! - `std` (default): standard-library support
//! - `verify`: karpal-verify obligation bundles for the parse invariants
//!   (the Borsalino verification pattern, adapted to a parser crate)
//!
//! ## Kani harnesses
//!
//! `src/kani_harnesses.rs` carries `#[cfg(kani)]` proof harnesses for the
//! validation arithmetic (offset containment, span consistency, overflow
//! rejection). Inert in normal builds; run with
//! `cargo kani --harness <name>`.
//!
//! ## Usage
//!
//! ```rust
//! use katakuri::safetensors::SafeTensors;
//!
//! # fn main() -> Result<(), katakuri::KatakuriError> {
//! // An empty safetensors file: 8-byte LE header length of 2, then "{}".
//! let bytes: &[u8] = &[2, 0, 0, 0, 0, 0, 0, 0, b'{', b'}'];
//! let st = SafeTensors::parse(bytes)?;
//! assert_eq!(st.tensor_count(), 0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Provenance
//!
//! Created after the 2026-10-16 `zip 8.6.0` crates.io yank permanently broke
//! `candle-core 0.11.0` dependency resolution. The full architecture and
//! slice plan live in
//! `docs/plans/2026-10-16-katakuri-architecture-handover.md`.

#![warn(missing_docs)]

pub mod error;
pub mod hub;
pub mod model;
pub mod safetensors;
pub mod tokenizer;

/// Karpal obligation bundles for the parse invariants (Borsalino pattern).
#[cfg(feature = "verify")]
pub mod verify;

/// Kani proof harnesses — compiled only under `cargo kani`.
#[cfg(kani)]
pub mod kani_harnesses;

pub use error::KatakuriError;
```

### 3.3 `src/error.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Central error type for Katakuri.
//!
//! One error enum for the whole crate, per IA conventions. Variants arrive
//! slice-by-slice with TDD: each variant exists because a test names its
//! failure mode. The S1 set covers safetensors validation failures and the
//! device-upload seam.

use thiserror::Error;

/// The error type for every fallible Katakuri operation.
#[derive(Debug, Error)]
pub enum KatakuriError {
    /// File shorter than the 8-byte header-length prefix.
    #[error("file too short for an 8-byte header length prefix: {file_len} bytes")]
    TruncatedFile {
        /// Actual file length observed.
        file_len: usize,
    },

    /// Declared header length exceeds the bytes remaining after the prefix.
    #[error("header length {header_len} exceeds the {available} bytes available after the prefix")]
    HeaderLengthExceedsFile {
        /// Declared header length from the prefix.
        header_len: u64,
        /// Bytes actually available after the 8-byte prefix.
        available: usize,
    },

    /// Declared header length exceeds the safety cap (DoS guard; matches the
    /// reference implementation's 100 MB bound).
    #[error("header length {header_len} exceeds the maximum allowed header size (100000000 bytes)")]
    HeaderTooLarge {
        /// Declared header length from the prefix.
        header_len: u64,
    },

    /// The header region is not valid JSON.
    #[error("header is not valid JSON: {detail}")]
    HeaderParse {
        /// Underlying parse failure detail.
        detail: String,
    },

    /// The header parsed but is not a JSON object.
    #[error("header must be a JSON object")]
    HeaderNotAnObject,

    /// `__metadata__` present but not a string-to-string map.
    #[error("__metadata__ invalid: {reason}")]
    InvalidMetadata {
        /// What was wrong with the metadata entry.
        reason: String,
    },

    /// A tensor entry carries a dtype string outside the safetensors set.
    #[error("tensor '{tensor}': invalid dtype '{dtype}'")]
    InvalidDtype {
        /// Name of the offending tensor.
        tensor: String,
        /// The unrecognized dtype string.
        dtype: String,
    },

    /// A tensor entry is structurally invalid (missing/mistyped field,
    /// unknown field, bad shape dimension, malformed offsets).
    #[error("tensor '{tensor}': {reason}")]
    InvalidTensorInfo {
        /// Name of the offending tensor.
        tensor: String,
        /// What was wrong with the entry.
        reason: String,
    },

    /// Data offsets fall outside the data region, or start > end.
    #[error("tensor '{tensor}': data offsets [{start}, {end}) exceed the {data_len}-byte data region")]
    OffsetOutOfBounds {
        /// Name of the offending tensor.
        tensor: String,
        /// Start offset.
        start: usize,
        /// End offset.
        end: usize,
        /// Length of the data region.
        data_len: usize,
    },

    /// Byte span does not match shape × dtype size.
    #[error("tensor '{tensor}': byte span {found} does not match shape×dtype ({expected})")]
    SizeMismatch {
        /// Name of the offending tensor.
        tensor: String,
        /// Expected span: element count × dtype byte size.
        expected: u64,
        /// Actual span: end − start.
        found: u64,
    },

    /// shape × dtype byte size overflows u64 arithmetic.
    #[error("tensor '{tensor}': shape×dtype overflows u64")]
    SizeOverflow {
        /// Name of the offending tensor.
        tensor: String,
    },

    /// Two tensors claim overlapping byte ranges.
    #[error("tensors '{first}' and '{second}' overlap at byte {offset}")]
    OverlappingTensors {
        /// Tensor with the earlier start offset.
        first: String,
        /// Tensor whose start falls inside `first`'s span.
        second: String,
        /// Byte offset where the overlap begins.
        offset: usize,
    },

    /// Named-tensor lookup missed.
    #[error("tensor '{name}' not found in the manifest")]
    TensorNotFound {
        /// The requested name.
        name: String,
    },

    /// A Zunesha device operation failed (includes `NoDevice` — the typed
    /// signal that no GPU backend is available and weights stay host-side).
    #[error("device error: {0}")]
    Device(
        /// The underlying Zunesha device error.
        #[from] zunesha::DeviceError,
    ),
}
```

### 3.4 `src/safetensors.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Weight parse + validate: dtype, shape, layout, and manifest checks over
//! the borrowed input bytes, plus a device-upload seam over
//! [`zunesha::Device`]. No math — Katakuri never computes.
//!
//! Replaces candle's `VarBuilder` and safetensors glue on Ijima's
//! `embeddings-candle` surface (slice S1).
//!
//! # Format
//!
//! A safetensors file is an 8-byte little-endian u64 header length N, then N
//! bytes of JSON header, then the data byte buffer. Header keys are tensor
//! names (or `__metadata__`); each tensor entry is
//! `{"dtype", "shape", "data_offsets": [start, end]}` with offsets relative
//! to the start of the data region.
//!
//! # Validation contract
//!
//! - header length capped at 100 MB (DoS guard, matching the reference
//!   implementation)
//! - `__metadata__` (if present) must be a string-to-string map
//! - tensor entries are strictly validated: exactly the three known fields,
//!   known dtype, integer shape dims, exactly two integer offsets
//! - `start <= end <= data_len` for every tensor
//! - `end - start == element_count * dtype_size` (checked u64 arithmetic)
//! - tensor byte spans must not overlap; gaps between tensors are allowed
//!   (every contiguous real-world file parses; corrupted ones reject)
//!
//! The parse is zero-copy: [`SafeTensors`] borrows the input slice and hands
//! out `&[u8]` views into it. GPU-less hardware is the first-class path —
//! nothing here requires a device. When one exists,
//! [`SafeTensors::upload_to_device`] moves a tensor's bytes into a
//! `zunesha::Buffer`.

use crate::error::KatakuriError;
use serde_json::Value;
use std::collections::BTreeMap;

/// Maximum accepted header length in bytes (DoS guard; matches the
/// reference implementation's bound).
const MAX_HEADER_BYTES: u64 = 100_000_000;

// ── Dtype ─────────────────────────────────────────────────────────

/// Element types defined by the safetensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// Boolean, one byte per element.
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// 8-bit float, E5M2 encoding.
    F8E5M2,
    /// 8-bit float, E4M3 encoding.
    F8E4M3,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// 16-bit IEEE float.
    F16,
    /// 16-bit brain float.
    Bf16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// 32-bit IEEE float.
    F32,
    /// 64-bit IEEE float.
    F64,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
}

impl Dtype {
    /// Bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 | Dtype::F8E5M2 | Dtype::F8E4M3 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        }
    }

    /// Parse the canonical safetensors dtype string (case-sensitive).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "F8_E5M2" => Dtype::F8E5M2,
            "F8_E4M3" => Dtype::F8E4M3,
            "I16" => Dtype::I16,
            "U16" => Dtype::U16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "I64" => Dtype::I64,
            "U64" => Dtype::U64,
            _ => return None,
        })
    }
}

// ── Span arithmetic (pure; Kani harness target) ───────────────────

/// Byte span for `shape` × `dtype`, or `None` on u64 overflow.
///
/// Pure function separated from parsing so `#[cfg(kani)]` harnesses can
/// prove its arithmetic in isolation. An empty shape (scalar) has element
/// count 1 (empty product).
pub(crate) fn span_for(shape: &[usize], dtype: Dtype) -> Option<u64> {
    let mut product: u64 = 1;
    for &dim in shape {
        product = product.checked_mul(u64::try_from(dim).unwrap_or(u64::MAX))?;
    }
    product.checked_mul(dtype.byte_size() as u64)
}

// ── Metadata ──────────────────────────────────────────────────────

/// Manifest metadata for one named tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    /// Element type of the tensor.
    pub dtype: Dtype,
    /// Shape, in row-major dimension order. Empty means scalar.
    pub shape: Vec<usize>,
    /// `[start, end)` byte range within the data region.
    pub data_offsets: (usize, usize),
}

// ── Tensor view ───────────────────────────────────────────────────

/// A validated, borrowed view of one tensor: manifest metadata plus the
/// tensor's raw bytes (a slice of the parsed input).
#[derive(Debug)]
pub struct Tensor<'meta, 'data> {
    meta: &'meta TensorMeta,
    bytes: &'data [u8],
}

impl Tensor<'_, '_> {
    /// Element type of this tensor.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.meta.dtype
    }

    /// Shape, in row-major dimension order.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.meta.shape
    }

    /// `[start, end)` byte range within the data region.
    #[must_use]
    pub fn data_offsets(&self) -> (usize, usize) {
        self.meta.data_offsets
    }

    /// The tensor's raw bytes, little-endian element order, no padding.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Element count: product of the shape dims (scalars have 1).
    #[must_use]
    pub fn elem_count(&self) -> u64 {
        self.meta
            .shape
            .iter()
            .map(|&d| u64::try_from(d).unwrap_or(u64::MAX))
            .product()
    }
}

// ── Parsed file ───────────────────────────────────────────────────

/// A parsed and fully validated safetensors manifest, borrowing the input
/// bytes.
///
/// Construct with [`SafeTensors::parse`]. All validation happens at parse
/// time: every tensor reachable afterwards satisfies the validation
/// contract in the module docs.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    tensors: BTreeMap<String, TensorMeta>,
    metadata: BTreeMap<String, String>,
    data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    /// Parse and validate a safetensors file held in `bytes`.
    ///
    /// Zero-copy: the returned manifest borrows `bytes`; tensor byte views
    /// point into it.
    ///
    /// # Errors
    ///
    /// Every violation of the validation contract (module docs) returns a
    /// matching [`KatakuriError`] variant; see the variant docs for the
    /// exact conditions.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, KatakuriError> {
        let prefix: [u8; 8] = bytes
            .get(..8)
            .and_then(|s| s.try_into().ok())
            .ok_or(KatakuriError::TruncatedFile {
                file_len: bytes.len(),
            })?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(KatakuriError::HeaderTooLarge { header_len });
        }
        let available = bytes.len() - 8;
        if header_len > available as u64 {
            return Err(KatakuriError::HeaderLengthExceedsFile {
                header_len,
                available,
            });
        }
        let header_end = 8 + usize::try_from(header_len).expect("bounded by len");
        let header: Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| {
            KatakuriError::HeaderParse {
                detail: e.to_string(),
            }
        })?;
        let map = header.as_object().ok_or(KatakuriError::HeaderNotAnObject)?;
        let data = &bytes[header_end..];

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, entry) in map {
            if name == "__metadata__" {
                let obj = entry.as_object().ok_or_else(|| {
                    KatakuriError::InvalidMetadata {
                        reason: format!("__metadata__ must be an object, got {entry}"),
                    }
                })?;
                for (k, v) in obj {
                    let s = v.as_str().ok_or_else(|| KatakuriError::InvalidMetadata {
                        reason: format!("__metadata__[{k}] must be a string"),
                    })?;
                    metadata.insert(k.clone(), s.to_string());
                }
                continue;
            }
            let meta = parse_tensor_entry(name, entry, data.len())?;
            tensors.insert(name.clone(), meta);
        }
        validate_no_overlap(&tensors)?;

        Ok(Self {
            tensors,
            metadata,
            data,
        })
    }

    /// Tensor names, lexicographic order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Number of tensors in the manifest.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The `__metadata__` string map (empty when absent).
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Length of the data region in bytes.
    #[must_use]
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Borrow a validated tensor view by name.
    ///
    /// # Errors
    ///
    /// [`KatakuriError::TensorNotFound`] when `name` is not in the manifest.
    pub fn tensor(&self, name: &str) -> Result<Tensor<'_, 'a>, KatakuriError> {
        let meta = self.tensors.get(name).ok_or_else(|| {
            KatakuriError::TensorNotFound {
                name: name.to_string(),
            }
        })?;
        let (start, end) = meta.data_offsets;
        Ok(Tensor {
            meta,
            bytes: &self.data[start..end],
        })
    }

    /// Upload one tensor's bytes into a device buffer via
    /// [`zunesha::Device::create_device_buffer`] (GPU-resident weights
    /// contract). The host view remains valid — the same weights can feed
    /// a precision CPU path or a Borsalino GPU path.
    ///
    /// # Errors
    ///
    /// [`KatakuriError::TensorNotFound`] on a lookup miss; any device
    /// failure (including `NoDevice` — the typed signal that no GPU
    /// backend exists and weights stay host-side) maps to
    /// [`KatakuriError::Device`].
    pub fn upload_to_device<D: zunesha::Device>(
        &self,
        name: &str,
        device: &D,
    ) -> Result<DeviceTensor, KatakuriError> {
        let t = self.tensor(name)?;
        let buffer = device.create_device_buffer(t.bytes())?;
        Ok(DeviceTensor {
            buffer,
            meta: t.meta.clone(),
        })
    }
}

/// A tensor's manifest plus its bytes living in a `zunesha::Buffer` on a
/// device.
#[derive(Debug)]
pub struct DeviceTensor {
    /// Device-owned byte buffer holding the tensor's data.
    pub buffer: zunesha::Buffer,
    /// The manifest entry the bytes were validated against.
    pub meta: TensorMeta,
}

// ── Entry parsing ─────────────────────────────────────────────────

fn parse_tensor_entry(
    name: &str,
    entry: &Value,
    data_len: usize,
) -> Result<TensorMeta, KatakuriError> {
    let obj = entry.as_object().ok_or_else(|| KatakuriError::InvalidTensorInfo {
        tensor: name.to_string(),
        reason: format!("entry must be an object, got {entry}"),
    })?;
    for key in obj.keys() {
        if key != "dtype" && key != "shape" && key != "data_offsets" {
            return Err(KatakuriError::InvalidTensorInfo {
                tensor: name.to_string(),
                reason: format!("unknown field '{key}'"),
            });
        }
    }

    let dtype_str = obj.get("dtype").and_then(Value::as_str).ok_or_else(|| {
        KatakuriError::InvalidTensorInfo {
            tensor: name.to_string(),
            reason: "missing or non-string 'dtype'".to_string(),
        }
    })?;
    let dtype = Dtype::parse(dtype_str).ok_or_else(|| KatakuriError::InvalidDtype {
        tensor: name.to_string(),
        dtype: dtype_str.to_string(),
    })?;

    let shape_v = obj.get("shape").and_then(Value::as_array).ok_or_else(|| {
        KatakuriError::InvalidTensorInfo {
            tensor: name.to_string(),
            reason: "missing or non-array 'shape'".to_string(),
        }
    })?;
    let mut shape = Vec::with_capacity(shape_v.len());
    for dim in shape_v {
        let d = dim
            .as_u64()
            .and_then(|d| usize::try_from(d).ok())
            .ok_or_else(|| KatakuriError::InvalidTensorInfo {
                tensor: name.to_string(),
                reason: "shape dims must be non-negative integers".to_string(),
            })?;
        shape.push(d);
    }

    let offs = obj
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| KatakuriError::InvalidTensorInfo {
            tensor: name.to_string(),
            reason: "missing or non-array 'data_offsets'".to_string(),
        })?;
    if offs.len() != 2 {
        return Err(KatakuriError::InvalidTensorInfo {
            tensor: name.to_string(),
            reason: "data_offsets must have exactly two entries".to_string(),
        });
    }
    let to_usize = |v: &Value| -> Option<usize> {
        v.as_u64().and_then(|x| usize::try_from(x).ok())
    };
    let (Some(start), Some(end)) = (to_usize(&offs[0]), to_usize(&offs[1])) else {
        return Err(KatakuriError::InvalidTensorInfo {
            tensor: name.to_string(),
            reason: "data_offsets entries must be non-negative integers".to_string(),
        });
    };

    if start > end || end > data_len {
        return Err(KatakuriError::OffsetOutOfBounds {
            tensor: name.to_string(),
            start,
            end,
            data_len,
        });
    }

    let expected = span_for(&shape, dtype).ok_or_else(|| KatakuriError::SizeOverflow {
        tensor: name.to_string(),
    })?;
    let found = u64::try_from(end - start).unwrap_or(u64::MAX);
    if expected != found {
        return Err(KatakuriError::SizeMismatch {
            tensor: name.to_string(),
            expected,
            found,
        });
    }

    Ok(TensorMeta {
        dtype,
        shape,
        data_offsets: (start, end),
    })
}

fn validate_no_overlap(tensors: &BTreeMap<String, TensorMeta>) -> Result<(), KatakuriError> {
    let mut spans: Vec<(&str, usize, usize)> = tensors
        .iter()
        .map(|(n, m)| (n.as_str(), m.data_offsets.0, m.data_offsets.1))
        .collect();
    spans.sort_unstable_by_key(|&(_, start, _)| start);
    for pair in spans.windows(2) {
        let (prev_name, _, prev_end) = pair[0];
        let (next_name, next_start, _) = pair[1];
        if next_start < prev_end {
            return Err(KatakuriError::OverlappingTensors {
                first: prev_name.to_string(),
                second: next_name.to_string(),
                offset: next_start,
            });
        }
    }
    Ok(())
}
```

#### 3.4 (continued) — append this block verbatim to the END of `src/safetensors.rs`

It is the in-file test module (shown separately only to keep one fenced block
per document section; it belongs to the same single file, after
`validate_no_overlap`). Do not modify the implementation above to satisfy a
test — the implementation is the contract.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn build_file(header: &Value, data: &[u8]) -> Vec<u8> {
        let header_bytes = serde_json::to_vec(header).expect("serialize header");
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(data);
        out
    }

    fn simple_header() -> Value {
        json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}})
    }

    const SIMPLE_DATA: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    // ── Dtype ─────────────────────────────────────────────────────

    #[test]
    fn dtype_byte_sizes_and_parse_roundtrip() {
        let all = [
            (Dtype::Bool, "BOOL", 1usize),
            (Dtype::U8, "U8", 1),
            (Dtype::I8, "I8", 1),
            (Dtype::F8E5M2, "F8_E5M2", 1),
            (Dtype::F8E4M3, "F8_E4M3", 1),
            (Dtype::I16, "I16", 2),
            (Dtype::U16, "U16", 2),
            (Dtype::F16, "F16", 2),
            (Dtype::Bf16, "BF16", 2),
            (Dtype::I32, "I32", 4),
            (Dtype::U32, "U32", 4),
            (Dtype::F32, "F32", 4),
            (Dtype::F64, "F64", 8),
            (Dtype::I64, "I64", 8),
            (Dtype::U64, "U64", 8),
        ];
        for (dt, s, size) in all {
            assert_eq!(dt.byte_size(), size, "{s}");
            assert_eq!(Dtype::parse(s), Some(dt), "{s}");
        }
        // Unknown and case-mismatched strings reject.
        assert_eq!(Dtype::parse("F128"), None);
        assert_eq!(Dtype::parse("f32"), None);
        assert_eq!(Dtype::parse(""), None);
    }

    // ── Header-level rejection ───────────────────────────────────

    #[test]
    fn parse_rejects_short_file() {
        assert!(matches!(
            SafeTensors::parse(&[]),
            Err(KatakuriError::TruncatedFile { file_len: 0 })
        ));
        assert!(matches!(
            SafeTensors::parse(&[0u8; 7]),
            Err(KatakuriError::TruncatedFile { file_len: 7 })
        ));
    }

    #[test]
    fn parse_rejects_header_too_large() {
        let mut bytes = 100_000_001u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::HeaderTooLarge { header_len: 100_000_001 })
        ));
    }

    #[test]
    fn parse_rejects_header_len_exceeding_file() {
        let mut bytes = 100u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::HeaderLengthExceedsFile {
                header_len: 100,
                available: 2
            })
        ));
    }

    #[test]
    fn parse_rejects_invalid_json() {
        let mut bytes = 4u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"notj");
        match SafeTensors::parse(&bytes) {
            Err(KatakuriError::HeaderParse { detail }) => {
                assert!(!detail.is_empty());
            }
            other => panic!("expected HeaderParse, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_non_object_header() {
        let mut bytes = 5u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"[1,2]");
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::HeaderNotAnObject)
        ));
    }

    // ── Metadata ─────────────────────────────────────────────────

    #[test]
    fn parse_rejects_invalid_metadata() {
        for bad in [json!(3), json!([1]), json!({"k": 1})] {
            let header = json!({"__metadata__": bad});
            let bytes = build_file(&header, &[]);
            assert!(
                matches!(
                    SafeTensors::parse(&bytes),
                    Err(KatakuriError::InvalidMetadata { .. })
                ),
                "metadata {bad} should reject"
            );
        }
    }

    #[test]
    fn metadata_roundtrip() {
        let header = json!({
            "__metadata__": {"format": "pt", "source": "katakuri-test"},
            "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        });
        let bytes = build_file(&header, &SIMPLE_DATA);
        let st = SafeTensors::parse(&bytes).expect("parse");
        assert_eq!(st.tensor_count(), 1);
        let md = st.metadata();
        assert_eq!(md.get("format"), Some(&"pt".to_string()));
        assert_eq!(md.get("source"), Some(&"katakuri-test".to_string()));
    }

    // ── Tensor-entry rejection ───────────────────────────────────

    #[test]
    fn parse_rejects_unknown_dtype() {
        let header = json!({"w": {"dtype": "F128", "shape": [2], "data_offsets": [0, 8]}});
        let bytes = build_file(&header, &[0u8; 8]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::InvalidDtype {
                tensor,
                dtype
            }) if tensor == "w" && dtype == "F128"
        ));
    }

    #[test]
    fn parse_rejects_missing_fields() {
        for (label, entry) in [
            ("dtype", json!({"shape": [2], "data_offsets": [0, 8]})),
            ("shape", json!({"dtype": "F32", "data_offsets": [0, 8]})),
            (
                "data_offsets",
                json!({"dtype": "F32", "shape": [2]}),
            ),
        ] {
            let header = json!({"w": entry});
            let bytes = build_file(&header, &[0u8; 8]);
            match SafeTensors::parse(&bytes) {
                Err(KatakuriError::InvalidTensorInfo { reason, .. }) => {
                    assert!(reason.contains(label), "{label}: {reason}");
                }
                other => panic!("{label}: expected InvalidTensorInfo, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_rejects_unknown_tensor_field() {
        let header = json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8], "foo": 1}});
        let bytes = build_file(&header, &[0u8; 8]);
        match SafeTensors::parse(&bytes) {
            Err(KatakuriError::InvalidTensorInfo { reason, .. }) => {
                assert!(reason.contains('\''), "reason names the field: {reason}");
            }
            other => panic!("expected InvalidTensorInfo, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_bad_shape_entries() {
        for shape in [json!([-1]), json!([1.5]), json!("x")] {
            let header = json!({"w": {"dtype": "F32", "shape": shape, "data_offsets": [0, 8]}});
            let bytes = build_file(&header, &[0u8; 8]);
            assert!(
                matches!(
                    SafeTensors::parse(&bytes),
                    Err(KatakuriError::InvalidTensorInfo { .. })
                ),
                "shape {shape} should reject"
            );
        }
        // shape present but not an array at all
        let header = json!({"w": {"dtype": "F32", "shape": 2, "data_offsets": [0, 8]}});
        let bytes = build_file(&header, &[0u8; 8]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::InvalidTensorInfo { .. })
        ));
    }

    #[test]
    fn parse_rejects_bad_offsets_shape() {
        for offs in [json!([0]), json!([0, 8, 16]), json!("x")] {
            let header = json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": offs}});
            let bytes = build_file(&header, &[0u8; 8]);
            assert!(
                matches!(
                    SafeTensors::parse(&bytes),
                    Err(KatakuriError::InvalidTensorInfo { .. })
                ),
                "offsets {offs} should reject"
            );
        }
    }

    // ── Offset / span validation ─────────────────────────────────

    #[test]
    fn parse_rejects_offset_out_of_bounds() {
        let header =
            json!({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}});
        let bytes = build_file(&header, &[0u8; 8]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::OffsetOutOfBounds {
                tensor,
                start: 0,
                end: 16,
                data_len: 8
            }) if tensor == "w"
        ));
        // start > end is also out of bounds
        let header =
            json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [6, 2]}});
        let bytes = build_file(&header, &[0u8; 8]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::OffsetOutOfBounds { start: 6, end: 2, .. })
        ));
    }

    #[test]
    fn parse_rejects_size_mismatch() {
        let header =
            json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 12]}});
        let bytes = build_file(&header, &[0u8; 12]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::SizeMismatch {
                tensor,
                expected: 8,
                found: 12
            }) if tensor == "w"
        ));
    }

    #[test]
    fn parse_rejects_size_overflow() {
        let header = json!({"w": {"dtype": "F64", "shape": [u64::MAX, u64::MAX], "data_offsets": [0, 0]}});
        let bytes = build_file(&header, &[]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::SizeOverflow { tensor }) if tensor == "w"
        ));
    }

    #[test]
    fn parse_rejects_overlapping_tensors() {
        let header = json!({
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}
        });
        let bytes = build_file(&header, &[0u8; 12]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::OverlappingTensors {
                first,
                second,
                offset: 4
            }) if first == "a" && second == "b"
        ));
    }

    // ── Acceptance ───────────────────────────────────────────────

    #[test]
    fn parse_allows_gap_between_tensors() {
        let header = json!({
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [12, 20]}
        });
        let mut data = vec![0xAA; 20];
        data[12..20].copy_from_slice(&[9, 8, 7, 6, 5, 4, 3, 2]);
        let bytes = build_file(&header, &data);
        let st = SafeTensors::parse(&bytes).expect("gap is legal");
        assert_eq!(st.tensor_count(), 2);
        assert_eq!(st.data_len(), 20);
        assert_eq!(st.tensor("a").unwrap().bytes(), &data[0..8]);
        let b = st.tensor("b").unwrap();
        assert_eq!(b.data_offsets(), (12, 20));
        assert_eq!(b.bytes(), &[9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn parse_scalar_tensor() {
        let header = json!({"s": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}});
        let bytes = build_file(&header, &[1, 2, 3, 4]);
        let st = SafeTensors::parse(&bytes).expect("scalar parses");
        let t = st.tensor("s").unwrap();
        assert_eq!(t.elem_count(), 1, "empty product is 1");
        assert_eq!(t.bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn parse_zero_dim_tensor() {
        let header = json!({"z": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}});
        let bytes = build_file(&header, &[]);
        let st = SafeTensors::parse(&bytes).expect("zero-dim parses");
        let t = st.tensor("z").unwrap();
        assert_eq!(t.elem_count(), 0);
        assert!(t.bytes().is_empty());
    }

    #[test]
    fn parse_empty_manifest() {
        let bytes = build_file(&json!({}), &[]);
        let st = SafeTensors::parse(&bytes).expect("empty manifest parses");
        assert_eq!(st.tensor_count(), 0);
        assert_eq!(st.names().count(), 0);
        assert!(st.metadata().is_empty());
        assert_eq!(st.data_len(), 0);
    }

    #[test]
    fn names_are_lexicographic() {
        let header = simple_header();
        let bytes = build_file(&header, &SIMPLE_DATA);
        let st = SafeTensors::parse(&bytes).expect("parse");
        let names: Vec<&str> = st.names().collect();
        assert_eq!(names, ["w"]);
        let header = json!({
            "b.w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "a.w": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
            "c.w": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}
        });
        let bytes = build_file(&header, &[0u8; 24]);
        let st = SafeTensors::parse(&bytes).expect("parse");
        let names: Vec<&str> = st.names().collect();
        assert_eq!(names, ["a.w", "b.w", "c.w"]);
    }

    #[test]
    fn tensor_lookup_hit_and_miss() {
        let bytes = build_file(&simple_header(), &SIMPLE_DATA);
        let st = SafeTensors::parse(&bytes).expect("parse");
        let t = st.tensor("w").expect("hit");
        assert_eq!(t.bytes(), &SIMPLE_DATA);
        assert!(matches!(
            st.tensor("nope"),
            Err(KatakuriError::TensorNotFound { name }) if name == "nope"
        ));
    }

    // ── Device seam ──────────────────────────────────────────────

    #[test]
    fn upload_to_nodevice_reports_no_device_and_host_stays_available() {
        let bytes = build_file(&simple_header(), &SIMPLE_DATA);
        let st = SafeTensors::parse(&bytes).expect("parse");
        let stub = zunesha::NoDeviceStub;
        let err = st.upload_to_device("w", &stub).expect_err("stub has no device");
        assert!(matches!(
            err,
            KatakuriError::Device(zunesha::DeviceError::NoDevice)
        ));
        // NoDevice path is first-class: the host view is unaffected.
        let t = st.tensor("w").expect("host view still available");
        assert_eq!(t.bytes(), &SIMPLE_DATA);
    }

    // ── MiniLM-shaped round-trip ─────────────────────────────────

    fn minilm_entries(vocab: usize, hidden: usize, layers: usize) -> Vec<(String, Vec<usize>)> {
        let mut entries = vec![
            ("embeddings.word_embeddings.weight".to_string(), vec![vocab, hidden]),
            ("embeddings.position_embeddings.weight".to_string(), vec![512, hidden]),
            ("embeddings.token_type_embeddings.weight".to_string(), vec![2, hidden]),
            ("embeddings.LayerNorm.weight".to_string(), vec![hidden]),
            ("embeddings.LayerNorm.bias".to_string(), vec![hidden]),
        ];
        for layer in 0..layers {
            for suffix in [
                "attention.self.query.weight",
                "attention.self.query.bias",
                "attention.self.key.weight",
                "attention.self.key.bias",
                "attention.self.value.weight",
                "attention.self.value.bias",
                "attention.output.dense.weight",
                "attention.output.dense.bias",
                "attention.output.LayerNorm.weight",
                "attention.output.LayerNorm.bias",
                "intermediate.dense.weight",
                "intermediate.dense.bias",
                "output.dense.weight",
                "output.dense.bias",
                "output.LayerNorm.weight",
                "output.LayerNorm.bias",
            ] {
                let name = format!("encoder.layer.{layer}.{suffix}");
                let shape = if suffix.ends_with(".bias") || suffix.contains("LayerNorm") {
                    vec![hidden]
                } else {
                    vec![hidden, hidden]
                };
                entries.push((name, shape));
            }
        }
        entries.push(("pooler.dense.weight".to_string(), vec![hidden, hidden]));
        entries.push(("pooler.dense.bias".to_string(), vec![hidden]));
        entries
    }

    fn build_safetensors(entries: &[(String, Vec<usize>)]) -> (Vec<u8>, Vec<Vec<u8>>) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut payloads = Vec::new();
        for (idx, (name, shape)) in entries.iter().enumerate() {
            let len: usize = shape.iter().product::<usize>() * 4; // F32
            let payload: Vec<u8> = (0..len).map(|i| ((idx * 31 + i) % 251) as u8).collect();
            let start = data.len();
            data.extend_from_slice(&payload);
            header.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, start + len]}),
            );
            payloads.push(payload);
        }
        (build_file(&Value::Object(header), &data), payloads)
    }

    #[test]
    fn roundtrip_minilm_shaped_manifest() {
        let entries = minilm_entries(32, 8, 6);
        assert_eq!(entries.len(), 103, "5 embedding + 6×16 encoder + 2 pooler");
        let (file, payloads) = build_safetensors(&entries);
        let st = SafeTensors::parse(&file).expect("MiniLM-shaped file parses");
        assert_eq!(st.tensor_count(), 103);
        for name in [
            "embeddings.word_embeddings.weight",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.5.output.LayerNorm.bias",
            "pooler.dense.weight",
        ] {
            assert!(st.tensor(name).is_ok(), "missing {name}");
        }
        for (idx, (name, shape)) in entries.iter().enumerate() {
            let t = st.tensor(name).expect(name);
            assert_eq!(t.dtype(), Dtype::F32, "{name}");
            assert_eq!(t.shape(), &shape[..], "{name}");
            let expected_elems: u64 = shape.iter().map(|&d| d as u64).product();
            assert_eq!(t.elem_count(), expected_elems, "{name}");
            assert_eq!(t.bytes(), payloads[idx].as_slice(), "{name} bytes");
        }
        let covered: usize = payloads.iter().map(Vec::len).sum();
        assert_eq!(st.data_len(), covered, "gap-free build covers exactly");
    }
}
```

### 3.5 `src/verify.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Verification obligations for Katakuri (the Borsalino pattern, adapted to
//! a parser crate).
//!
//! Borsalino bundles GPU-kernel safety obligations; Katakuri bundles
//! **parse invariants** — the properties `SafeTensors::parse` enforces
//! before any tensor becomes reachable. Every property is tier
//! [`VerificationTier::Impossible`]: the validator makes the illegal state
//! unrepresentable (a manifest that parses is a manifest that satisfies
//! them).
//!
//! Bundles export to the karpal SMT/Lean/Kani backends.
//!
//! # Properties
//!
//! | Property | What it prevents | Enforced by |
//! |---|---|---|
//! | [`IsOffsetInBounds`] | offsets escaping the data region | `OffsetOutOfBounds` rejection |
//! | [`IsSpanConsistent`] | span ≠ shape×dtype bytes | `SizeMismatch` / `SizeOverflow` rejection |
//! | [`IsManifestNonOverlapping`] | two tensors claiming the same bytes | `OverlappingTensors` rejection |

use karpal_proof::Property;
use karpal_verify::{Obligation, ObligationBundle, Origin, Sort, Term, VerificationTier};

/// Every tensor's byte range lies within `[0, data_len]` with `start <= end`.
pub struct IsOffsetInBounds;

impl Property for IsOffsetInBounds {
    const NAME: &'static str = "katakuri.safetensors.offset_in_bounds";
}

/// Every tensor's byte span equals its element count times its dtype size.
pub struct IsSpanConsistent;

impl Property for IsSpanConsistent {
    const NAME: &'static str = "katakuri.safetensors.span_consistent";
}

/// No two tensors claim overlapping byte ranges.
pub struct IsManifestNonOverlapping;

impl Property for IsManifestNonOverlapping {
    const NAME: &'static str = "katakuri.safetensors.manifest_non_overlapping";
}

/// Obligation bundle for the safetensors parse surface (all three
/// properties, tier `Impossible`).
#[must_use]
pub fn safetensors_parse_obligations() -> ObligationBundle {
    ObligationBundle::new(
        "katakuri_safetensors_parse",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
    )
    .with(offset_in_bounds_obligation())
    .with(span_consistent_obligation())
    .with(manifest_non_overlapping_obligation())
}

fn offset_in_bounds_obligation() -> Obligation {
    Obligation::for_property::<IsOffsetInBounds>(
        "offset_in_bounds",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
        VerificationTier::Impossible,
        Term::and([
            Term::app("le", [Term::int(0), Term::var("start")]),
            Term::app("le", [Term::var("start"), Term::var("end")]),
            Term::app("le", [Term::var("end"), Term::var("data_len")]),
        ]),
    )
    .with_decl("start", Sort::Int)
    .with_decl("end", Sort::Int)
    .with_decl("data_len", Sort::Int)
}

fn span_consistent_obligation() -> Obligation {
    Obligation::for_property::<IsSpanConsistent>(
        "span_consistent",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
        VerificationTier::Impossible,
        Term::eq(
            Term::app("sub", [Term::var("end"), Term::var("start")]),
            Term::app("mul", [Term::var("elem_count"), Term::var("dtype_size")]),
        ),
    )
    .with_decl("start", Sort::Int)
    .with_decl("end", Sort::Int)
    .with_decl("elem_count", Sort::Int)
    .with_decl("dtype_size", Sort::Int)
}

fn manifest_non_overlapping_obligation() -> Obligation {
    Obligation::for_property::<IsManifestNonOverlapping>(
        "manifest_non_overlapping",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
        VerificationTier::Impossible,
        Term::app("le", [Term::var("prev_end"), Term::var("next_start")]),
    )
    .with_decl("prev_end", Sort::Int)
    .with_decl("next_start", Sort::Int)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_obligations_bundle_carries_three_impossible_tier_obligations() {
        let bundle = safetensors_parse_obligations();
        let obligations = bundle.obligations();
        assert_eq!(obligations.len(), 3);
        let names: Vec<&str> = obligations.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(
            names,
            ["offset_in_bounds", "span_consistent", "manifest_non_overlapping"]
        );
        for o in obligations {
            assert!(matches!(o.tier, VerificationTier::Impossible));
        }
        assert_eq!(
            IsOffsetInBounds::NAME,
            "katakuri.safetensors.offset_in_bounds"
        );
        assert_eq!(IsSpanConsistent::NAME, "katakuri.safetensors.span_consistent");
        assert_eq!(
            IsManifestNonOverlapping::NAME,
            "katakuri.safetensors.manifest_non_overlapping"
        );
    }
}
```

### 3.6 `src/kani_harnesses.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Katakuri validation arithmetic.
//!
//! Run with:
//! ```sh
//! cargo kani --harness span_consistency_arithmetic
//! cargo kani --harness span_overflow_rejected
//! cargo kani --harness offset_containment_implies_valid_slice
//! ```
//!
//! Requires Kani installed: <https://model-checking.github.io/kani/>
//! Inert in normal builds (`#[cfg(kani)]`).

use crate::safetensors::{span_for, Dtype};

/// Bounded dims: span is exactly product × dtype size, a dtype multiple.
#[cfg(kani)]
#[kani::proof]
fn span_consistency_arithmetic() {
    let d0: u64 = kani::any();
    let d1: u64 = kani::any();
    kani::assume(d0 <= 65_536 && d1 <= 65_536);
    let shape: Vec<usize> = vec![d0 as usize, d1 as usize];
    let span = span_for(&shape, Dtype::F32).expect("no overflow under bounds");
    assert_eq!(span, d0 * d1 * 4);
    assert_eq!(span % 4, 0);
}

/// u64::MAX dims: overflow is detected, never wrapped.
#[cfg(kani)]
#[kani::proof]
fn span_overflow_rejected() {
    let shape = vec![u64::MAX as usize];
    assert!(span_for(&shape, Dtype::U64).is_none());
}

/// Containment (start ≤ end ≤ data_len) implies safe slicing.
#[cfg(kani)]
#[kani::proof]
fn offset_containment_implies_valid_slice() {
    let start: usize = kani::any();
    let end: usize = kani::any();
    let data_len: usize = kani::any();
    kani::assume(data_len <= 1 << 20);
    kani::assume(start <= end && end <= data_len);
    let data = vec![0u8; data_len];
    let slice = &data[start..end]; // must not panic
    assert_eq!(slice.len(), end - start);
}
```

### 3.7 `tests/real_file.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Env-gated round-trip against a real safetensors file (no fixture is
//! committed; real model files are far too large for the repo).
//!
//! Run with any real file, e.g. one of the local ComfyUI safetensors, or a
//! downloaded MiniLM `model.safetensors`:
//!
//! ```sh
//! KATAKURI_TEST_SAFETENSORS=/path/to/model.safetensors \
//!   cargo test --test real_file -- --ignored --nocapture
//! ```

use katakuri::safetensors::SafeTensors;

/// FNV-1a; a tiny in-test checksum so the test adds no dependencies.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
#[ignore = "set KATAKURI_TEST_SAFETENSORS=<path to a real .safetensors> to run"]
fn real_file_roundtrip() {
    let path = std::env::var("KATAKURI_TEST_SAFETENSORS")
        .expect("KATAKURI_TEST_SAFETENSORS not set");
    let file = std::fs::read(&path).expect("read file");
    let st = SafeTensors::parse(&file).expect("parse real file");

    let header_len =
        u64::from_le_bytes(file[0..8].try_into().expect("8-byte prefix")) as usize;
    let data_start = 8 + header_len;

    let mut count = 0usize;
    let mut covered: u64 = 0;
    for name in st.names() {
        let t = st.tensor(name).expect("validated tensor");
        let (s, e) = t.data_offsets();
        // Independent read of the same span, straight from the raw file
        // bytes — bypassing the parser's slicing entirely.
        let direct = &file[data_start + s..data_start + e];
        assert_eq!(
            fnv1a(t.bytes()),
            fnv1a(direct),
            "byte mismatch on tensor {name}"
        );
        assert_eq!(t.bytes().len(), e - s, "{name} span");
        covered += u64::try_from(e - s).unwrap_or(u64::MAX);
        count += 1;
    }
    assert!(count > 0, "real file has tensors");
    assert!(
        covered <= st.data_len() as u64,
        "coverage {covered} exceeds data region {}",
        st.data_len()
    );
}
```

### 3.8 `CHANGELOG.md`

```markdown
# Changelog

All notable changes to Katakuri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — S1: safetensors parse/validate (2026-10-16)

- `katakuri::safetensors`: `SafeTensors::parse` — zero-copy, fully
  validated manifest over borrowed bytes. Validation contract: 100 MB
  header cap, strict tensor-entry fields, dtype/shape/offset checks,
  span == shape×dtype (checked u64 arithmetic), no-overlap with gaps
  allowed. `Tensor` views (`dtype`/`shape`/`data_offsets`/`bytes`/
  `elem_count`), `__metadata__` extraction.
- Device seam: `SafeTensors::upload_to_device` over `zunesha::Device`
  (`create_device_buffer`, GPU-resident weights). NoDevice path is
  first-class: parsing needs no device; upload failure maps to
  `KatakuriError::Device(DeviceError::NoDevice)` with host views intact.
- Borsalino verification pattern: `verify` feature with karpal-verify
  obligation bundles (offset-in-bounds, span-consistent,
  manifest-non-overlapping; tier `Impossible`), plus `#[cfg(kani)]`
  proof harnesses for the validation arithmetic.
- MiniLM-shaped round-trip test (103-tensor BERT manifest, byte-exact),
  rejection tests for every validation rule, env-gated real-file
  round-trip (`KATAKURI_TEST_SAFETENSORS`).

### Added — repo scaffold (2026-10-16)

- Library crate skeleton (`katakuri`, lib target; replaces the scaffold's
  binary stub) per IA conventions: Apache-2.0 + SPDX headers on every source
  file, pinned stable toolchain (`rust-toolchain.toml`), central
  `KatakuriError` (thiserror), and module stubs for the four planned
  surfaces — `safetensors` (S1), `tokenizer` (S2), `model` (S3), `hub` (S4).
- Licensing and process stack: Apache-2.0 `LICENSE`, `CONTRIBUTING.md` with
  CLA reference, Forgejo mirror under `industrial-algebra` with dual-push
  wiring.

### Dependencies (S1)

- `serde_json 1` — the safetensors header is JSON; the ecosystem-standard
  parser (serde/itoa/ryu/memchr transitives only).
- `zunesha 0.1.2` — the device substrate (Buffer upload, NoDevice path).
- `karpal-verify`/`karpal-proof 0.6.1` — verification obligations, behind
  the `verify` feature (pinned to Borsalino's pin).
```

## 4. Tests (the enumeration — all in the §3.4 contract block unless noted)

| # | Test (name) | Rule pinned |
|---|---|---|
| 1 | `dtype_byte_sizes_and_parse_roundtrip` | all 15 dtypes: byte_size + case-sensitive parse; unknowns reject |
| 2 | `parse_rejects_short_file` | `TruncatedFile` (0 and 7 bytes) |
| 3 | `parse_rejects_header_too_large` | `HeaderTooLarge` at 100,000,001 — checked BEFORE availability |
| 4 | `parse_rejects_header_len_exceeding_file` | `HeaderLengthExceedsFile { header_len, available }` |
| 5 | `parse_rejects_invalid_json` | `HeaderParse` with non-empty detail |
| 6 | `parse_rejects_non_object_header` | `HeaderNotAnObject` |
| 7 | `parse_rejects_invalid_metadata` | `InvalidMetadata`: non-object, array, non-string value |
| 8 | `metadata_roundtrip` | string map extracted; tensor count unaffected |
| 9 | `parse_rejects_unknown_dtype` | `InvalidDtype { tensor, dtype }` |
| 10 | `parse_rejects_missing_fields` | `InvalidTensorInfo`; reason names dtype/shape/data_offsets |
| 11 | `parse_rejects_unknown_tensor_field` | `InvalidTensorInfo`; reason names the field |
| 12 | `parse_rejects_bad_shape_entries` | negative, fractional, string, non-array shape |
| 13 | `parse_rejects_bad_offsets_shape` | 1-entry, 3-entry, string offsets |
| 14 | `parse_rejects_offset_out_of_bounds` | end > data_len AND start > end |
| 15 | `parse_rejects_size_mismatch` | expected 8, found 12 |
| 16 | `parse_rejects_size_overflow` | u64::MAX dims → `SizeOverflow` |
| 17 | `parse_rejects_overlapping_tensors` | `OverlappingTensors { first, second, offset }` |
| 18 | `parse_allows_gap_between_tensors` | gap legal; bytes exact both sides |
| 19 | `parse_scalar_tensor` | empty shape → elem_count 1, span = dtype size |
| 20 | `parse_zero_dim_tensor` | zero dim → elem_count 0, empty bytes |
| 21 | `parse_empty_manifest` | `{}` parses; counts zero |
| 22 | `names_are_lexicographic` | BTreeMap ordering |
| 23 | `tensor_lookup_hit_and_miss` | `TensorNotFound { name }` |
| 24 | `upload_to_nodevice_reports_no_device_and_host_stays_available` | NoDevice typed error; host unaffected |
| 25 | `roundtrip_minilm_shaped_manifest` | 103 tensors, byte-exact, dtype/shape/elem_count |
| 26 | `parse_obligations_bundle_carries_three_impossible_tier_obligations` (§3.5, runs under `--features verify`) | bundle shape + tier |
| 27 | `real_file_roundtrip` (§3.7, `#[ignore]`, env-gated) | independent byte-span checksum vs raw file |

Acceptance mapping (handover §5 S1): round-trip = #25 + #27; corrupted-tensor
rejection = #2–#17; Ijima weight-loading half expressible = #22/#23/#24 (the
name→dtype/shape/bytes lookup + NoDevice path Ijima's loader needs).

## 5. Constraints

- Dependencies are PINNED to §3.1 — no new crates beyond
  thiserror/serde_json/zunesha/karpal-verify/karpal-proof, no version edits.
- Every `.rs` file starts with the two-line Apache-2.0 SPDX header (in the
  contract blocks verbatim).
- No `unwrap()`/`expect()`/`panic!()` in non-test library code paths. The two
  `expect` calls inside `SafeTensors::parse` are structurally unreachable
  (bounds checked immediately before) — keep them exactly as written.
- Feature gates are additive only. `verify` code must not leak into default
  builds; the no-verify configuration must compile clean.
- Files outside §2's list are out of bounds.
- No git commands (branches, commits, pushes) — leave the working tree dirty.

## 6. Completion

Run, filter, and fix until ALL are green (pipe long outputs; never paste full
logs; cargo timeouts in seconds, ≤ 600):

```bash
cargo test -p katakuri 2>&1 | grep -E 'test result|FAILED|error' | tail -10
cargo test -p katakuri --features verify 2>&1 | grep -E 'test result|FAILED|error' | tail -10
cargo clippy --all-targets -- -D warnings 2>&1 | tail -5
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -5
cargo fmt --check && echo FMT-OK
cargo doc --no-deps 2>&1 | tail -3
```

Expected: 25 unit tests green in default config; +1 verify test with the
feature; both clippy configs zero warnings; fmt clean; doc builds with zero
warnings.

## 7. Out of scope (fences)

- Serialization/writing of safetensors (consumers read; hub is S4).
- Any file I/O or network (`std::fs::read` belongs to consumers/hub).
- Tokenization (S2), model specs (S3), forward passes (never — hard boundary).
- Running Kani harnesses (needs the Kani toolchain; compile-inert here).
- Ijima integration PR (separate, in Ijima's repo).
- Touching `zunesha` (it is a pinned upstream dep, not a workspace member).
- Updating README.md (post-S1 PR will link the module docs).

## 8. Author-side checklist (already applied — for the verify pass)

- No RPITIT traits in the contract (plain `impl Iterator` return on a method
  is fine in edition 2024 with lifetime auto-capture).
- `verify`-gated items: `#[cfg(feature = "verify")] pub mod verify;` is the
  ONLY referencing site; no ungated re-export. No-verify mental compile: ✓.
- Every demanded assertion is observable through the public API
  (`names`, `tensor_count`, `tensor().*`, `metadata`, `data_len`,
  `upload_to_device`'s error, and `pub` error-variant fields).
- Cumulative files (`Cargo.toml`, `src/lib.rs`, `CHANGELOG.md`) are full
  final contents, not deltas. `src/safetensors.rs` is two verbatim blocks
  (impl + trailing test module) — concatenate, do not interleave.
- API surfaces cross-checked against the cached karpal 0.6.1 and zunesha
  0.1.2 sources, not local checkouts that may be ahead.
