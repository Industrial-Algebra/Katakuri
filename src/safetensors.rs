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
        let prefix: [u8; 8] =
            bytes
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
                let obj = entry
                    .as_object()
                    .ok_or_else(|| KatakuriError::InvalidMetadata {
                        reason: format!("__metadata__ must be an object, got {entry}"),
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
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| KatakuriError::TensorNotFound {
                name: name.to_string(),
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
    let obj = entry
        .as_object()
        .ok_or_else(|| KatakuriError::InvalidTensorInfo {
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
    let to_usize =
        |v: &Value| -> Option<usize> { v.as_u64().and_then(|x| usize::try_from(x).ok()) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

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
            Err(KatakuriError::HeaderTooLarge {
                header_len: 100_000_001
            })
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
            ("data_offsets", json!({"dtype": "F32", "shape": [2]})),
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
        let header = json!({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}});
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
        let header = json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [6, 2]}});
        let bytes = build_file(&header, &[0u8; 8]);
        assert!(matches!(
            SafeTensors::parse(&bytes),
            Err(KatakuriError::OffsetOutOfBounds {
                start: 6,
                end: 2,
                ..
            })
        ));
    }

    #[test]
    fn parse_rejects_size_mismatch() {
        let header = json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 12]}});
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
        let header =
            json!({"w": {"dtype": "F64", "shape": [u64::MAX, u64::MAX], "data_offsets": [0, 0]}});
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
        let err = st
            .upload_to_device("w", &stub)
            .expect_err("stub has no device");
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
            (
                "embeddings.word_embeddings.weight".to_string(),
                vec![vocab, hidden],
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                vec![512, hidden],
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                vec![2, hidden],
            ),
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
