// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Model fetch + content-addressed cache over plain HTTPS, offline-first.
//!
//! Replaces `hf-hub`. Optional feature; lands with slice S4. If exotic
//! tokenizers appear later, this layer carries converter tooling rather than
//! runtime dependencies.
