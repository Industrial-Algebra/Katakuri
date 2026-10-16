// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! WordPiece (BERT-class) tokenization matching Ijima's exact tokenization
//! contract: special tokens, padding, and truncation semantics as used by
//! `embeddings_candle.rs`.
//!
//! Bespoke by design (the zip-yank incident that created this repo argues for
//! owning the surface; WordPiece for the used model family is small and fully
//! auditable) — replaces the `tokenizers` crate dependency. Lands with slice
//! S2; acceptance is byte-identical token id sequences versus HF tokenizers
//! on a fixed test corpus (golden vectors committed).
