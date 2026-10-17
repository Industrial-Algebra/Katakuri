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

    /// A `tokenizer.json` carries a normalizer kind Katakuri does not
    /// implement (only `BertNormalizer` is supported).
    #[error("unsupported normalizer kind '{kind}' (only BertNormalizer)")]
    UnsupportedNormalizer {
        /// The rejected kind string.
        kind: String,
    },

    /// A `tokenizer.json` carries a pre-tokenizer kind Katakuri does not
    /// implement (only `BertPreTokenizer` is supported).
    #[error("unsupported pre-tokenizer kind '{kind}' (only BertPreTokenizer)")]
    UnsupportedPreTokenizer {
        /// The rejected kind string.
        kind: String,
    },

    /// A `tokenizer.json` carries a post-processor Katakuri does not
    /// implement, or a template shape other than `[CLS] A [SEP]`.
    #[error("unsupported post-processor: {kind}")]
    UnsupportedPostProcessor {
        /// The rejected kind or shape description.
        kind: String,
    },

    /// A `tokenizer.json` carries a model kind other than WordPiece.
    #[error("unsupported model kind '{kind}' (only WordPiece)")]
    UnsupportedModel {
        /// The rejected kind string.
        kind: String,
    },

    /// A `tokenizer.json` enables truncation with a strategy other than
    /// `LongestFirst` (the single-sequence BERT default).
    #[error("unsupported truncation strategy '{strategy}' (only LongestFirst)")]
    UnsupportedTruncationStrategy {
        /// The rejected strategy string.
        strategy: String,
    },

    /// A `tokenizer.json` enables truncation with a direction other than
    /// `Right` (left truncation is not implemented).
    #[error("unsupported truncation direction '{direction}' (only Right)")]
    UnsupportedTruncationDirection {
        /// The rejected direction string.
        direction: String,
    },

    /// A `config.json` value is malformed, missing, or outside the
    /// supported BERT subset. `detail` names the field and the problem.
    #[error("invalid model config: {detail}")]
    InvalidConfigJson {
        /// Field-level description of the divergence.
        detail: String,
    },

    /// A parsed weight set diverges from the spec's layout manifest.
    /// `detail` lists every divergence found (missing, unexpected,
    /// wrong-shape, wrong-dtype), one per line.
    #[error("weight layout mismatch: {detail}")]
    WeightLayoutMismatch {
        /// All divergences, one per line.
        detail: String,
    },

    /// A `tokenizer.json` is malformed: a required field is missing or has
    /// the wrong shape (model vocab, added-token entries, truncation
    /// parameters, added-token id/content mismatch against the vocab).
    #[error("tokenizer.json invalid: {detail}")]
    InvalidTokenizerJson {
        /// What was wrong.
        detail: String,
    },

    /// A vocab source contained no tokens at all.
    #[error("vocab is empty")]
    EmptyVocab,

    /// A BERT-class special token (`[CLS]`, `[SEP]`, `[UNK]`, `[PAD]`) is
    /// absent from the vocab.
    #[error("vocab is missing the special token '{token}'")]
    MissingSpecialToken {
        /// The missing special token text.
        token: String,
    },

    /// A Zunesha device operation failed (includes `NoDevice` — the typed
    /// signal that no GPU backend is available and weights stay host-side).
    #[error("device error: {0}")]
    Device(
        /// The underlying Zunesha device error.
        #[from]
        zunesha::DeviceError,
    ),

    /// A repo id, revision, or filename fails the strict path-safety
    /// grammar (empty segment, `/`, `..`, NUL, or a character outside
    /// ASCII alnum + `-._`).
    #[error("unsafe repo path: {detail}")]
    UnsafeRepoPath {
        /// What was rejected and why.
        detail: String,
    },

    /// Transport failure while fetching a model file (DNS, TLS, HTTP
    /// status, body read, size cap).
    #[error("hub fetch failed: {detail}")]
    HubFetch {
        /// Transport-level description.
        detail: String,
    },

    /// Local cache failure (unusable root, IO error, corrupt entry that
    /// cannot be repaired by refetch).
    #[error("hub cache error: {detail}")]
    HubCache {
        /// Cache-level description.
        detail: String,
    },
}
