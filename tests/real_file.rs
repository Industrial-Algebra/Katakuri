// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Env-gated round-trip against a real safetensors file (no fixture is
//! committed; real model files are far too large for the repo).
//!
//! Run with any real file, e.g. one of the local ComfyUI safetensors, or a
//! downloaded MiniLM `model.safetensors`:
//!
//! ```sh
//! KATAKURI_TEST_SAFETENSORS=/path/to/model.safetensors \
//!   cargo test --test real_file -- --ignored --nocapture
//! ```

use katakuri::safetensors::SafeTensors;

/// FNV-1a; a tiny in-test checksum so the test adds no dependencies.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
#[ignore = "set KATAKURI_TEST_SAFETENSORS=<path to a real .safetensors> to run"]
fn real_file_roundtrip() {
    let path =
        std::env::var("KATAKURI_TEST_SAFETENSORS").expect("KATAKURI_TEST_SAFETENSORS not set");
    let file = std::fs::read(&path).expect("read file");
    let st = SafeTensors::parse(&file).expect("parse real file");

    let header_len = u64::from_le_bytes(file[0..8].try_into().expect("8-byte prefix")) as usize;
    let data_start = 8 + header_len;

    let mut count = 0usize;
    let mut covered: u64 = 0;
    for name in st.names() {
        let t = st.tensor(name).expect("validated tensor");
        let (s, e) = t.data_offsets();
        // Independent read of the same span, straight from the raw file
        // bytes — bypassing the parser's slicing entirely.
        let direct = &file[data_start + s..data_start + e];
        assert_eq!(
            fnv1a(t.bytes()),
            fnv1a(direct),
            "byte mismatch on tensor {name}"
        );
        assert_eq!(t.bytes().len(), e - s, "{name} span");
        covered += u64::try_from(e - s).unwrap_or(u64::MAX);
        count += 1;
    }
    assert!(count > 0, "real file has tensors");
    assert!(
        covered <= st.data_len() as u64,
        "coverage {covered} exceeds data region {}",
        st.data_len()
    );
}
