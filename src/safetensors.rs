// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Weight parse + validate: dtype, shape, layout, and manifest checks, with
//! payloads landing in a `zunesha::Buffer` when a device exists or in host
//! memory on Zunesha's `NoDevice` path. No math.
//!
//! Replaces candle's `VarBuilder` and safetensors glue on Ijima's
//! `embeddings-candle` surface. Lands with slice S1; acceptance is
//! round-tripping a real MiniLM safetensors file, rejecting corrupted
//! tensors at dtype/shape validation, and expressing Ijima's weight-loading
//! half.
