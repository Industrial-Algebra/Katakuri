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
//!   tokenizers 0.21.x (S2). Replaces the `tokenizers` crate dependency.
//! - [`model`] — typed BERT config + weight-layout manifests, validated
//!   against parsed safetensors; the Ijima swap boundary (S3). Replaces
//!   candle-transformers `models::bert`.
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
