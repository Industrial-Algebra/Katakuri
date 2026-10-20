// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Canonical content hashing for local artifacts (S5, wishlist W5).
//!
//! One sha256 implementation serves both surfaces: local artifact identity
//! ([`artifact_sha256`], used by validated checkpoint loading for the
//! `ia.schema-hash` / `ia.model-hash` metadata convention) and the
//! content-addressed hub cache ([`crate::hub`], whose digest sidecars use the
//! same lowercase-hex format).
//!
//! This module is behind the `digest` feature, and the `hub` feature depends
//! on it: local digests must not require the hub, but the hub must not carry a
//! second hash implementation.
//!
//! ```
//! # #[cfg(feature = "digest")] {
//! use katakuri::digest::artifact_sha256;
//!
//! assert_eq!(
//!     artifact_sha256(b"abc"),
//!     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
//! );
//! # }
//! ```

/// Lowercase hex sha256 of `bytes` — the canonical artifact identity.
///
/// Naming is deliberate: this is the local-file digest used by the
/// checkpoint metadata convention (S5) and by the hub sidecars (S4). The
/// bytes are hashed verbatim; callers that hash a *region* (a safetensors
/// data section, a canonical spec serialization) pass exactly those bytes.
#[must_use]
pub fn artifact_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest.iter() {
        // Writing to a String is infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Lowercase hex sha256 of `bytes` (S4 hub name, kept for API compatibility).
///
/// Identical to [`artifact_sha256`]; [`crate::hub`] re-exports both so the
/// S4 surface is unchanged.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    artifact_sha256(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(
            artifact_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            artifact_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hub_name_is_the_same_function() {
        for input in [b"" as &[u8], b"abc", b"hello hub", &[0u8; 33]] {
            assert_eq!(sha256_hex(input), artifact_sha256(input));
        }
    }

    #[test]
    fn output_is_lowercase_hex_of_fixed_width() {
        let digest = artifact_sha256(b"any bytes at all");
        assert_eq!(digest.len(), 64);
        assert!(
            digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "unexpected digest characters: {digest}"
        );
    }
}
