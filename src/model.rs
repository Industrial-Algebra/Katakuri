// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Architecture specs as typed data: BERT config (layers / heads / dims /
//! hidden act) and named-tensor layout manifests.
//!
//! Specs are data, never code — no forward implementations live here.
//! Replaces candle-transformers `models::bert`. Lands with slice S3 together
//! with the swap-boundary port plan for `embeddings_candle.rs`; candle stays
//! behind Ijima's non-default `embeddings-candle` feature throughout.
