// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The shared Katakuri ⇄ `borsalino-tensor` layout contract (S5, T0).
//!
//! Katakuri **emits** a [`TensorManifest`]; the tensor layer **consumes** it.
//! The type is data, not buffers: no `zunesha` dependency lives here, because
//! binding entries to `zunesha::Buffer` handles is the consumer's concern
//! (Borsalino PR #61 §4.4).
//!
//! # Unstable — the T0 co-sign gate
//!
//! **Every public type in this module is unstable and NOT frozen.** The
//! contract says "co-design it with the Katakuri session before either side
//! freezes an API" (Borsalino PR #61 §6, slice T0). This module is the
//! Katakuri proposal; it lands in `0.2.0` as a proposal, and the shapes below
//! freeze only after the Borsalino tensor-extension session reviews them
//! against T0's zero-copy `zunesha::Buffer` needs. Treat a `0.2.x` patch
//! release as free to change these fields.
//!
//! # What the manifest says
//!
//! - **Order:** [`TensorManifest::entries`] is the canonical parameter order —
//!   exactly [`MlpSpec::expected_tensors`]'s depth-first layer order.
//! - **dtype:** F32 only in v1 ([`ManifestDtype::F32`]). Casts happen at the
//!   consumer's load boundary, explicitly, never silently.
//! - **layout:** row-major, contiguous, no padding
//!   ([`ManifestLayout::RowMajorContiguous`]) — zero-copy into a buffer.
//! - **determinism:** the metadata the compute layer needs so it does not
//!   select fast paths that change bit patterns (W3, Katakuri side).
//!
//! ```
//! use katakuri::manifest::{
//!     DeterminismPolicy, ManifestDtype, ManifestEntry, ManifestLayout, TensorManifest,
//! };
//!
//! let manifest = TensorManifest {
//!     entries: vec![ManifestEntry {
//!         name: "head.bias".to_string(),
//!         shape: vec![1],
//!         byte_offset: 0,
//!         byte_len: 4,
//!     }],
//!     dtype: ManifestDtype::F32,
//!     layout: ManifestLayout::RowMajorContiguous,
//!     determinism: DeterminismPolicy::pinned(),
//! };
//! assert_eq!(manifest.entries[0].shape, vec![1]);
//! ```
//!
//! [`MlpSpec::expected_tensors`]: crate::model::MlpSpec::expected_tensors

/// A typed shape/stride/dtype layout manifest for one checkpoint's
/// parameters.
///
/// **Unstable — T0 co-sign gate.** The Borsalino tensor-extension session
/// reviews this shape before it freezes (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorManifest {
    /// Canonical parameter order (`== expected_tensors()` order).
    pub entries: Vec<ManifestEntry>,
    /// The element type of every entry.
    pub dtype: ManifestDtype,
    /// The memory layout every entry follows.
    pub layout: ManifestLayout,
    /// Determinism metadata for the compute layer.
    pub determinism: DeterminismPolicy,
}

/// One parameter's name, shape, and byte span inside a safetensors data
/// section.
///
/// **Unstable — T0 co-sign gate.**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Canonical tensor name — `"hidden.0.weight"`, `"head.bias"`, …
    pub name: String,
    /// Shape in row-major dimension order: `[out, in]` for weights, `[out]`
    /// for biases. The row-major stride is implied by this shape.
    pub shape: Vec<u32>,
    /// Byte offset into the safetensors data section.
    pub byte_offset: u64,
    /// Byte length of the tensor: element count × dtype size.
    pub byte_len: u64,
}

/// The element type of a manifest's parameters.
///
/// **Unstable — T0 co-sign gate.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestDtype {
    /// 32-bit IEEE float — the v1 contract (Borsalino §4.3). Casts to and
    /// from any other precision happen at the consumer's load boundary,
    /// explicitly.
    F32,
}

/// The memory layout of a manifest's parameters.
///
/// **Unstable — T0 co-sign gate.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestLayout {
    /// Row-major, contiguous, no padding between elements or rows —
    /// zero-copy into a `zunesha::Buffer`.
    RowMajorContiguous,
}

/// Determinism metadata the compute layer must honour (wishlist W3, Katakuri
/// side).
///
/// The compute itself is Borsalino's; this type carries only what the loader
/// can promise: the dtype is pinned, and bit-pattern-changing fast paths are
/// forbidden.
///
/// **Unstable — T0 co-sign gate.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeterminismPolicy {
    /// The dtype the compute layer must not substitute (no silent TF32/f16
    /// promotion).
    pub dtype_pinned: ManifestDtype,
    /// Fast-path selection policy for the compute layer.
    pub fast_paths: FastPathPolicy,
}

impl DeterminismPolicy {
    /// The only policy v1 permits: F32 pinned, fast paths forbidden.
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            dtype_pinned: ManifestDtype::F32,
            fast_paths: FastPathPolicy::Forbidden,
        }
    }
}

/// Fast-path selection policy for the compute layer.
///
/// **Unstable — T0 co-sign gate.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastPathPolicy {
    /// No backend-selected fast path may change a bit pattern: no fast-math,
    /// no TF32 substitution, no reassociation. Amari's ordering decisions
    /// must be replayable.
    Forbidden,
}
