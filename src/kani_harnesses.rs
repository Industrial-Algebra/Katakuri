// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Katakuri validation arithmetic.
//!
//! Run with:
//! ```sh
//! cargo kani --harness span_consistency_arithmetic
//! cargo kani --harness span_overflow_rejected
//! cargo kani --harness offset_containment_implies_valid_slice
//! ```
//!
//! Requires Kani installed: <https://model-checking.github.io/kani/>
//! Inert in normal builds (`#[cfg(kani)]`).

use crate::safetensors::{Dtype, span_for};

/// Bounded dims: span is exactly product × dtype size, a dtype multiple.
#[cfg(kani)]
#[kani::proof]
fn span_consistency_arithmetic() {
    let d0: u64 = kani::any();
    let d1: u64 = kani::any();
    kani::assume(d0 <= 65_536 && d1 <= 65_536);
    let shape: Vec<usize> = vec![d0 as usize, d1 as usize];
    let span = span_for(&shape, Dtype::F32).expect("no overflow under bounds");
    assert_eq!(span, d0 * d1 * 4);
    assert_eq!(span % 4, 0);
}

/// u64::MAX dims: overflow is detected, never wrapped.
#[cfg(kani)]
#[kani::proof]
fn span_overflow_rejected() {
    let shape = vec![u64::MAX as usize];
    assert!(span_for(&shape, Dtype::U64).is_none());
}

/// Containment (start ≤ end ≤ data_len) implies safe slicing.
#[cfg(kani)]
#[kani::proof]
fn offset_containment_implies_valid_slice() {
    let start: usize = kani::any();
    let end: usize = kani::any();
    let data_len: usize = kani::any();
    kani::assume(data_len <= 1 << 20);
    kani::assume(start <= end && end <= data_len);
    let data = vec![0u8; data_len];
    let slice = &data[start..end]; // must not panic
    assert_eq!(slice.len(), end - start);
}
