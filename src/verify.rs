// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Verification obligations for Katakuri (the Borsalino pattern, adapted to
//! a parser crate).
//!
//! Borsalino bundles GPU-kernel safety obligations; Katakuri bundles
//! **parse invariants** — the properties `SafeTensors::parse` enforces
//! before any tensor becomes reachable. Every property is tier
//! [`Impossible`](karpal_verify::VerificationTier::Impossible): the
//! validator makes the illegal state unrepresentable (a manifest that
//! parses is a manifest that satisfies them).
//!
//! Bundles export to the karpal SMT/Lean/Kani backends.
//!
//! # Properties
//!
//! | Property | What it prevents | Enforced by |
//! |---|---|---|
//! | [`IsOffsetInBounds`](crate::verify::IsOffsetInBounds) | offsets escaping the data region | `OffsetOutOfBounds` rejection |
//! | [`IsSpanConsistent`](crate::verify::IsSpanConsistent) | span ≠ shape×dtype bytes | `SizeMismatch` / `SizeOverflow` rejection |
//! | [`IsManifestNonOverlapping`](crate::verify::IsManifestNonOverlapping) | two tensors claiming the same bytes | `OverlappingTensors` rejection |

use karpal_proof::Property;
use karpal_verify::{Obligation, ObligationBundle, Origin, Sort, Term, VerificationTier};

/// Every tensor's byte range lies within `[0, data_len]` with `start <= end`.
pub struct IsOffsetInBounds;

impl Property for IsOffsetInBounds {
    const NAME: &'static str = "katakuri.safetensors.offset_in_bounds";
}

/// Every tensor's byte span equals its element count times its dtype size.
pub struct IsSpanConsistent;

impl Property for IsSpanConsistent {
    const NAME: &'static str = "katakuri.safetensors.span_consistent";
}

/// No two tensors claim overlapping byte ranges.
pub struct IsManifestNonOverlapping;

impl Property for IsManifestNonOverlapping {
    const NAME: &'static str = "katakuri.safetensors.manifest_non_overlapping";
}

/// Obligation bundle for the safetensors parse surface (all three
/// properties, tier `Impossible`).
#[must_use]
pub fn safetensors_parse_obligations() -> ObligationBundle {
    ObligationBundle::new(
        "katakuri_safetensors_parse",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
    )
    .with(offset_in_bounds_obligation())
    .with(span_consistent_obligation())
    .with(manifest_non_overlapping_obligation())
}

fn offset_in_bounds_obligation() -> Obligation {
    Obligation::for_property::<IsOffsetInBounds>(
        "offset_in_bounds",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
        VerificationTier::Impossible,
        Term::and([
            Term::app("le", [Term::int(0), Term::var("start")]),
            Term::app("le", [Term::var("start"), Term::var("end")]),
            Term::app("le", [Term::var("end"), Term::var("data_len")]),
        ]),
    )
    .with_decl("start", Sort::Int)
    .with_decl("end", Sort::Int)
    .with_decl("data_len", Sort::Int)
}

fn span_consistent_obligation() -> Obligation {
    Obligation::for_property::<IsSpanConsistent>(
        "span_consistent",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
        VerificationTier::Impossible,
        Term::eq(
            Term::app("sub", [Term::var("end"), Term::var("start")]),
            Term::app("mul", [Term::var("elem_count"), Term::var("dtype_size")]),
        ),
    )
    .with_decl("start", Sort::Int)
    .with_decl("end", Sort::Int)
    .with_decl("elem_count", Sort::Int)
    .with_decl("dtype_size", Sort::Int)
}

fn manifest_non_overlapping_obligation() -> Obligation {
    Obligation::for_property::<IsManifestNonOverlapping>(
        "manifest_non_overlapping",
        Origin::new("katakuri", "safetensors::SafeTensors::parse"),
        VerificationTier::Impossible,
        Term::app("le", [Term::var("prev_end"), Term::var("next_start")]),
    )
    .with_decl("prev_end", Sort::Int)
    .with_decl("next_start", Sort::Int)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_obligations_bundle_carries_three_impossible_tier_obligations() {
        let bundle = safetensors_parse_obligations();
        let obligations = bundle.obligations();
        assert_eq!(obligations.len(), 3);
        let names: Vec<&str> = obligations.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "offset_in_bounds",
                "span_consistent",
                "manifest_non_overlapping"
            ]
        );
        for o in obligations {
            assert!(matches!(o.tier, VerificationTier::Impossible));
        }
        assert_eq!(
            IsOffsetInBounds::NAME,
            "katakuri.safetensors.offset_in_bounds"
        );
        assert_eq!(
            IsSpanConsistent::NAME,
            "katakuri.safetensors.span_consistent"
        );
        assert_eq!(
            IsManifestNonOverlapping::NAME,
            "katakuri.safetensors.manifest_non_overlapping"
        );
    }
}
