// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Central error type for Katakuri.
//!
//! One error enum for the whole crate, per IA conventions. The enum is empty
//! at scaffold time — the crate has no fallible surface yet. Variants arrive
//! slice-by-slice with TDD: each slice writes the failing test that names its
//! failure mode *before* the variant that expresses it, so no variant exists
//! without a test proving its diagnostic shape.

use thiserror::Error;

/// The error type for every fallible Katakuri operation.
///
/// Empty until slice S1 (`safetensors`) introduces the first fallible
/// surface — parse, dtype, shape, and layout validation failures.
#[derive(Debug, Error)]
pub enum KatakuriError {}
