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
    #[error(
        "tensor '{tensor}': data offsets [{start}, {end}) exceed the {data_len}-byte data region"
    )]
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
        #[from]
        zunesha::DeviceError,
    ),
}
