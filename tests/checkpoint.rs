// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Validated checkpoint loading (S5): the trust-policy pipeline end to end.
//!
//! Everything here uses the public API: synthetic checkpoints are built in
//! memory (header + data section), written to a unique temp path, and loaded
//! through [`load_validated`]. The one committed fixture
//! (`tests/fixtures/tiny-mlp.safetensors`, 769 bytes) exercises the full
//! policy-on path, including a subnormal weight.
//!
//! Compiled only with `std` + `digest`: the loader's file-I/O and hash
//! surfaces.

#![cfg(all(feature = "std", feature = "digest"))]

use katakuri::checkpoint::{
    META_MODEL_HASH, META_SCHEMA_HASH, META_SPEC, META_TRAINING_DATA, TrustPolicy, load_validated,
};
use katakuri::digest::artifact_sha256;
use katakuri::error::{KatakuriError, NonfiniteKind};
use katakuri::manifest::{DeterminismPolicy, ManifestDtype, ManifestLayout, TensorManifest};
use katakuri::model::{Activation, MlpLayerSpec, MlpSpec};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Helpers ───────────────────────────────────────────────────────

/// The all-on policy used by the full-path tests.
fn strict_policy() -> TrustPolicy {
    TrustPolicy {
        regular_file_only: true,
        byte_cap: 1 << 20,
        verify_schema_hash: true,
        verify_model_hash: true,
        reject_nonfinite: true,
    }
}

/// The same policy with every check off — every flag is an explicit choice.
fn permissive_policy() -> TrustPolicy {
    TrustPolicy {
        regular_file_only: false,
        byte_cap: u64::MAX,
        verify_schema_hash: false,
        verify_model_hash: false,
        reject_nonfinite: false,
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "katakuri-checkpoint-test-{}-{tag}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_temp(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = temp_dir(tag).join("model.safetensors");
    std::fs::write(&path, bytes).expect("write checkpoint fixture");
    path
}

/// A legal spec JSON value with the given input width, hidden widths, and
/// activation string.
fn spec_json(input_width: u64, hidden: &[u64], activation: &str) -> Value {
    json!({
        "model_type": "ia-mlp",
        "schema_version": 1,
        "input_width": input_width,
        "hidden": hidden.iter().map(|w| json!({ "width": w })).collect::<Vec<_>>(),
        "head": { "width": 1 },
        "hidden_activation": activation,
    })
}

fn spec(input_width: u64, hidden: &[u64], activation: &str) -> MlpSpec {
    MlpSpec::from_json(&spec_json(input_width, hidden, activation).to_string())
        .expect("generated spec is legal")
}

/// Builds a synthetic safetensors checkpoint from an `MlpSpec`: zero-filled
/// tensors in canonical order, with the S5 metadata convention.
struct CheckpointBuilder {
    spec: MlpSpec,
    tensors: Vec<Vec<f32>>,
    /// Metadata edits applied after the correct defaults are computed.
    edits: Vec<(String, Option<String>)>,
}

impl CheckpointBuilder {
    fn new(spec: MlpSpec) -> Self {
        let tensors = spec
            .expected_tensors()
            .iter()
            .map(|(_, shape)| vec![0.0_f32; shape.iter().product()])
            .collect();
        Self {
            spec,
            tensors,
            edits: Vec::new(),
        }
    }

    /// Replace one tensor's values (index in canonical order).
    fn tensor(mut self, index: usize, values: Vec<f32>) -> Self {
        self.tensors[index] = values;
        self
    }

    /// Override (or add) one metadata entry.
    fn set_meta(mut self, key: &str, value: &str) -> Self {
        self.edits.push((key.to_string(), Some(value.to_string())));
        self
    }

    /// Remove one metadata entry.
    fn remove_meta(mut self, key: &str) -> Self {
        self.edits.push((key.to_string(), None));
        self
    }

    fn build(self) -> Vec<u8> {
        let expected = self.spec.expected_tensors();
        assert_eq!(self.tensors.len(), expected.len(), "one payload per tensor");

        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for ((name, shape), values) in expected.iter().zip(self.tensors.iter()) {
            let count: usize = shape.iter().product();
            assert_eq!(values.len(), count, "payload size for {name}");
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }

        let canonical = self.spec.canonical_json();
        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        metadata.insert(META_SPEC.to_string(), canonical.clone());
        metadata.insert(
            META_SCHEMA_HASH.to_string(),
            artifact_sha256(canonical.as_bytes()),
        );
        metadata.insert(META_MODEL_HASH.to_string(), artifact_sha256(&data));
        metadata.insert(
            META_TRAINING_DATA.to_string(),
            artifact_sha256(b"synthetic training traces"),
        );
        for (key, value) in self.edits {
            match value {
                Some(value) => {
                    metadata.insert(key, value);
                }
                None => {
                    metadata.remove(&key);
                }
            }
        }
        header.insert("__metadata__".to_string(), json!(metadata));

        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }
}

/// The data section of an already-valid safetensors file.
fn data_section(bytes: &[u8]) -> &[u8] {
    let len = u64::from_le_bytes(bytes[..8].try_into().expect("prefix")) as usize;
    &bytes[8 + len..]
}

fn load(
    bytes: &[u8],
    tag: &str,
    policy: &TrustPolicy,
) -> Result<katakuri::checkpoint::ValidatedCheckpoint, KatakuriError> {
    let path = write_temp(tag, bytes);
    load_validated(&path, policy)
}

// ── The committed fixture: full policy-on path ────────────────────

#[test]
fn fixture_passes_the_full_policy_path() {
    let bytes: &[u8] = include_bytes!("fixtures/tiny-mlp.safetensors");
    let path = write_temp("fixture", bytes);
    let loaded = load_validated(&path, &strict_policy()).expect("committed fixture validates");

    assert_eq!(loaded.spec.input_width, 2);
    assert_eq!(
        loaded.spec.hidden,
        vec![MlpLayerSpec { width: 3 }],
        "hidden layer list"
    );
    assert_eq!(loaded.spec.head, MlpLayerSpec { width: 1 });
    assert_eq!(loaded.spec.hidden_activation, Activation::Relu);

    // Manifest: canonical order, shapes, spans.
    let entries = &loaded.manifest.entries;
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "hidden.0.weight",
            "hidden.0.bias",
            "head.weight",
            "head.bias"
        ]
    );
    let shapes: Vec<Vec<u32>> = entries.iter().map(|e| e.shape.clone()).collect();
    assert_eq!(shapes, vec![vec![3, 2], vec![3], vec![1, 3], vec![1]]);
    let spans: Vec<(u64, u64)> = entries
        .iter()
        .map(|e| (e.byte_offset, e.byte_len))
        .collect();
    assert_eq!(spans, vec![(0, 24), (24, 12), (36, 12), (48, 4)]);
    assert_eq!(loaded.manifest.dtype, ManifestDtype::F32);
    assert_eq!(loaded.manifest.layout, ManifestLayout::RowMajorContiguous);
    assert_eq!(loaded.manifest.determinism, DeterminismPolicy::pinned());

    // Provenance is carried, not interpreted.
    assert_eq!(
        loaded
            .metadata
            .get(META_TRAINING_DATA)
            .map(String::len)
            .expect("provenance present"),
        64
    );

    // Tensors: canonical order, bit-exact values.
    let tensor_names: Vec<&str> = loaded.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tensor_names, names);
    let weight: Vec<f32> = loaded.tensors[0]
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    assert_eq!(weight, vec![0.5, -0.25, 1.0, 0.75, -1.5, 2.0]);
    assert_eq!(loaded.tensors[3].data, (-0.375_f32).to_le_bytes());

    // The fixture's subnormal bias survived the scan: it is finite.
    let subnormal = f32::from_le_bytes(
        loaded.tensors[1].data[4..8]
            .try_into()
            .expect("second bias element"),
    );
    assert_ne!(subnormal, 0.0, "subnormal must not be flushed to zero");
    assert!(
        subnormal.abs() < f32::MIN_POSITIVE,
        "expected a subnormal, got {subnormal:e}"
    );
    assert!(subnormal.is_finite());

    // Canonical round-trip: re-serializing tensors in manifest order
    // reproduces the data section byte for byte.
    let reserialized: Vec<u8> = loaded
        .tensors
        .iter()
        .flat_map(|t| t.data.iter().copied())
        .collect();
    assert_eq!(reserialized, data_section(bytes));
}

// ── Manifest agreement property ───────────────────────────────────

#[test]
fn manifest_agrees_with_expected_tensors_over_generated_specs() {
    let activations = ["relu", "tanh", "gelu_erf", "gelu_tanh"];
    let layer_sets: [&[u64]; 6] = [&[], &[1], &[3], &[2, 5], &[1, 1, 1], &[16, 8, 4, 2]];
    let mut cases = 0;

    for (i, input_width) in [1_u64, 2, 7, 64, 4096].iter().enumerate() {
        for (j, hidden) in layer_sets.iter().enumerate() {
            let activation = activations[(i + j) % activations.len()];
            let spec = spec(*input_width, hidden, activation);
            let expected = spec.expected_tensors();
            let bytes = CheckpointBuilder::new(spec.clone()).build();
            let loaded = load(&bytes, "property", &strict_policy())
                .unwrap_or_else(|e| panic!("generated spec {input_width}/{hidden:?} loads: {e}"));

            assert_eq!(loaded.manifest.entries.len(), expected.len());
            assert_eq!(loaded.tensors.len(), expected.len());

            for (entry, (name, shape)) in loaded.manifest.entries.iter().zip(expected.iter()) {
                assert_eq!(&entry.name, name);
                let as_u32: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
                assert_eq!(entry.shape, as_u32);
                let elements: u64 = shape.iter().map(|&d| d as u64).product();
                assert_eq!(entry.byte_len, elements * 4, "F32 byte length for {name}");
            }

            // No gaps, no padding: entries tile the data section.
            let mut cursor = 0_u64;
            for entry in &loaded.manifest.entries {
                assert_eq!(entry.byte_offset, cursor, "entry order is contiguous");
                cursor += entry.byte_len;
            }
            assert_eq!(cursor as usize, data_section(&bytes).len());

            cases += 1;
        }
    }

    assert!(cases >= 20, "property coverage: {cases} generated specs");
}

// ── Shape / schema mismatch ───────────────────────────────────────

#[test]
fn shape_mismatch_is_rejected() {
    // Built for width 3, declared as width 4: every hidden weight/bias
    // diverges while the names still match.
    let built = spec(2, &[3], "relu");
    let declared = spec(2, &[4], "relu");
    let bytes = CheckpointBuilder::new(built)
        .set_meta(META_SPEC, &declared.canonical_json())
        .build();
    match load(&bytes, "shape", &strict_policy()) {
        Err(KatakuriError::WeightLayoutMismatch { detail }) => {
            assert!(detail.contains("hidden.0.weight"), "{detail}");
            assert!(detail.contains("wrong shape"), "{detail}");
        }
        other => panic!("expected WeightLayoutMismatch, got {other:?}"),
    }
}

#[test]
fn missing_and_unexpected_tensors_are_rejected() {
    // Built with one hidden layer, declared with none: the extra tensors are
    // unexpected and the head shape diverges.
    let built = spec(2, &[3], "relu");
    let declared = spec(2, &[], "relu");
    let bytes = CheckpointBuilder::new(built)
        .set_meta(META_SPEC, &declared.canonical_json())
        .build();
    match load(&bytes, "unexpected", &strict_policy()) {
        Err(KatakuriError::WeightLayoutMismatch { detail }) => {
            assert!(
                detail.contains("unexpected tensor 'hidden.0.weight'"),
                "{detail}"
            );
            assert!(
                detail.contains("unexpected tensor 'hidden.0.bias'"),
                "{detail}"
            );
        }
        other => panic!("expected WeightLayoutMismatch, got {other:?}"),
    }
}

// ── Hash verification ─────────────────────────────────────────────

#[test]
fn tampered_payload_byte_is_a_model_hash_mismatch() {
    let spec = spec(2, &[3], "relu");
    let mut bytes = CheckpointBuilder::new(spec).build();
    // The data section is last: flipping its final byte changes the digest.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;

    match load(&bytes, "model-hash", &strict_policy()) {
        Err(KatakuriError::ModelHashMismatch { expected, actual }) => {
            assert_eq!(expected.len(), 64);
            assert_eq!(actual.len(), 64);
            assert_ne!(expected, actual);
        }
        other => panic!("expected ModelHashMismatch, got {other:?}"),
    }
}

#[test]
fn tampered_spec_is_a_schema_hash_mismatch() {
    // Same layout, different activation: the layout check passes, so the
    // schema hash is what catches the edit.
    let built = spec(2, &[3], "relu");
    let altered = spec(2, &[3], "tanh");
    let declared = built.canonical_json();
    assert_ne!(declared, altered.canonical_json());
    let bytes = CheckpointBuilder::new(built)
        .set_meta(META_SPEC, &altered.canonical_json())
        .build();

    match load(&bytes, "schema-hash", &strict_policy()) {
        Err(KatakuriError::SchemaHashMismatch { expected, actual }) => {
            // `expected` is the declared digest; `actual` is what the parsed
            // (altered) spec hashes to.
            assert_eq!(expected, artifact_sha256(declared.as_bytes()));
            assert_eq!(actual, artifact_sha256(altered.canonical_json().as_bytes()));
            assert_ne!(expected, actual);
        }
        other => panic!("expected SchemaHashMismatch, got {other:?}"),
    }
}

#[test]
fn absent_declared_hashes_reject_only_when_verified() {
    let spec = spec(2, &[3], "relu");
    let without_schema = CheckpointBuilder::new(spec.clone())
        .remove_meta(META_SCHEMA_HASH)
        .build();
    match load(&without_schema, "absent-schema", &strict_policy()) {
        Err(KatakuriError::SchemaHashMismatch { expected, .. }) => {
            assert_eq!(expected, "<absent>");
        }
        other => panic!("expected SchemaHashMismatch, got {other:?}"),
    }

    let without_model = CheckpointBuilder::new(spec.clone())
        .remove_meta(META_MODEL_HASH)
        .build();
    match load(&without_model, "absent-model", &strict_policy()) {
        Err(KatakuriError::ModelHashMismatch { expected, .. }) => {
            assert_eq!(expected, "<absent>");
        }
        other => panic!("expected ModelHashMismatch, got {other:?}"),
    }

    // Policy off: the same files load (nothing is silently on).
    load(&without_schema, "absent-schema-off", &permissive_policy())
        .expect("hash verification is a policy choice");
    load(&without_model, "absent-model-off", &permissive_policy())
        .expect("hash verification is a policy choice");
}

#[test]
fn hash_mismatch_is_reported_before_the_nonfinite_scan() {
    let spec = spec(2, &[3], "relu");
    // head.bias is the last tensor and holds one element.
    let bytes = CheckpointBuilder::new(spec.clone())
        .tensor(3, vec![f32::NAN])
        .set_meta(META_MODEL_HASH, "0".repeat(64).as_str())
        .build();
    match load(&bytes, "hash-before-scan", &strict_policy()) {
        Err(KatakuriError::ModelHashMismatch { .. }) => {}
        other => panic!("hashes must be verified before the scan, got {other:?}"),
    }
}

// ── Nonfinite scan ────────────────────────────────────────────────

#[test]
fn nonfinite_values_are_rejected_with_the_tensor_name() {
    for (value, kind) in [
        (f32::NAN, NonfiniteKind::Nan),
        (f32::INFINITY, NonfiniteKind::PositiveInfinity),
        (f32::NEG_INFINITY, NonfiniteKind::NegativeInfinity),
    ] {
        let spec = spec(2, &[3], "relu");
        // head.weight is tensor index 2; poison its middle element.
        let mut values = vec![0.0_f32; 3];
        values[1] = value;
        let bytes = CheckpointBuilder::new(spec).tensor(2, values).build();
        match load(&bytes, "nonfinite", &strict_policy()) {
            Err(KatakuriError::NonfiniteTensorValue { tensor, which }) => {
                assert_eq!(tensor, "head.weight");
                assert_eq!(which, kind);
            }
            other => panic!("expected NonfiniteTensorValue for {kind}, got {other:?}"),
        }
    }
}

#[test]
fn subnormals_are_finite_and_kept() {
    let spec = spec(2, &[3], "relu");
    let subnormal = f32::from_bits(1); // smallest positive subnormal
    assert!(subnormal.is_finite() && subnormal > 0.0);
    let bytes = CheckpointBuilder::new(spec.clone())
        .tensor(3, vec![subnormal])
        .build();
    let loaded = load(&bytes, "subnormal", &strict_policy()).expect("subnormal is legal");
    assert_eq!(loaded.tensors[3].data, subnormal.to_le_bytes());
    // And the largest subnormal, below MIN_POSITIVE, also passes.
    let largest_subnormal = f32::from_bits(0x007f_ffff);
    assert!(largest_subnormal < f32::MIN_POSITIVE && largest_subnormal.is_finite());
    let bytes = CheckpointBuilder::new(spec)
        .tensor(3, vec![largest_subnormal])
        .build();
    load(&bytes, "subnormal-max", &strict_policy()).expect("normal-boundary subnormal is legal");
}

#[test]
fn scan_off_lets_nan_through() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec)
        .tensor(3, vec![f32::NAN])
        .build();
    let loaded = load(&bytes, "scan-off", &permissive_policy())
        .expect("scan is a policy choice, not a parse invariant");
    let value = f32::from_le_bytes(loaded.tensors[3].data[..4].try_into().expect("4 bytes"));
    assert!(value.is_nan());

    // The same bytes with the scan on reject: the policy is the only
    // difference.
    assert!(matches!(
        load(&bytes, "scan-on", &strict_policy()),
        Err(KatakuriError::NonfiniteTensorValue {
            which: NonfiniteKind::Nan,
            ..
        })
    ));
}

// ── Byte cap ──────────────────────────────────────────────────────

#[test]
fn byte_cap_boundary_is_exact() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec).build();
    let len = bytes.len() as u64;
    let path = write_temp("cap", &bytes);

    let exact = TrustPolicy {
        byte_cap: len,
        ..strict_policy()
    };
    load_validated(&path, &exact).expect("a file exactly at the cap is accepted");

    let one_over = TrustPolicy {
        byte_cap: len - 1,
        ..strict_policy()
    };
    match load_validated(&path, &one_over) {
        Err(KatakuriError::CheckpointTooLarge { size, cap }) => {
            assert_eq!(size, len);
            assert_eq!(cap, len - 1);
        }
        other => panic!("expected CheckpointTooLarge, got {other:?}"),
    }

    let zero = TrustPolicy {
        byte_cap: 0,
        ..strict_policy()
    };
    assert!(matches!(
        load_validated(&path, &zero),
        Err(KatakuriError::CheckpointTooLarge { cap: 0, .. })
    ));
}

// ── Metadata convention (§5.5) ───────────────────────────────────

#[test]
fn unknown_ia_metadata_rejects() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec)
        .set_meta("ia.unknown-key", "value")
        .build();
    match load(&bytes, "unknown-ia", &strict_policy()) {
        Err(KatakuriError::UnknownIaMetadata { key }) => assert_eq!(key, "ia.unknown-key"),
        other => panic!("expected UnknownIaMetadata, got {other:?}"),
    }
}

#[test]
fn non_ia_metadata_is_tolerated() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec)
        .set_meta("format", "pt")
        .set_meta("writer", "tests")
        .build();
    let loaded = load(&bytes, "non-ia", &strict_policy()).expect("non-ia metadata is free-form");
    assert_eq!(
        loaded.metadata.get("writer").map(String::as_str),
        Some("tests")
    );
}

#[test]
fn provenance_length_is_bounded() {
    let spec = spec(2, &[3], "relu");
    for (value, accepted) in [
        ("".to_string(), false),
        ("a".repeat(4096), true),
        ("a".repeat(4097), false),
    ] {
        let bytes = CheckpointBuilder::new(spec.clone())
            .set_meta(META_TRAINING_DATA, &value)
            .build();
        let result = load(&bytes, "provenance", &strict_policy());
        if accepted {
            result.expect("4096 bytes is the inclusive bound");
        } else if let Err(KatakuriError::ProvenanceMalformed { key }) = result {
            assert_eq!(key, META_TRAINING_DATA);
        } else {
            panic!(
                "expected ProvenanceMalformed for a length-{} value",
                value.len()
            );
        }
    }
}

#[test]
fn missing_spec_metadata_rejects() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec).remove_meta(META_SPEC).build();
    match load(&bytes, "no-spec", &strict_policy()) {
        Err(KatakuriError::ProvenanceMalformed { key }) => assert_eq!(key, META_SPEC),
        other => panic!("expected ProvenanceMalformed, got {other:?}"),
    }
}

#[test]
fn malformed_spec_json_is_an_mlp_spec_error() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec)
        .set_meta(META_SPEC, "{\"model_type\":\"ia-mlp\"}")
        .build();
    assert!(matches!(
        load(&bytes, "bad-spec", &strict_policy()),
        Err(KatakuriError::MlpSpecParse { .. })
    ));
}

// ── Regular-file policy ──────────────────────────────────────────

#[test]
fn missing_path_is_an_io_error() {
    let dir = temp_dir("missing");
    let path = dir.join("does-not-exist.safetensors");
    match load_validated(&path, &strict_policy()) {
        Err(KatakuriError::CheckpointIo { path: reported, .. }) => {
            assert!(reported.contains("does-not-exist"), "{reported}");
        }
        other => panic!("expected CheckpointIo, got {other:?}"),
    }
}

#[test]
fn a_directory_is_not_a_regular_file() {
    let dir = temp_dir("directory");
    match load_validated(&dir, &strict_policy()) {
        Err(KatakuriError::CheckpointNotRegularFile { .. }) => {}
        other => panic!("expected CheckpointNotRegularFile, got {other:?}"),
    }
    // Policy off: opening a directory for reading still fails, as I/O.
    assert!(matches!(
        load_validated(&dir, &permissive_policy()),
        Err(KatakuriError::CheckpointIo { .. })
    ));
}

#[cfg(unix)]
#[test]
fn symlinks_reject_under_the_regular_file_policy() {
    let spec = spec(2, &[3], "relu");
    let bytes = CheckpointBuilder::new(spec).build();
    let dir = temp_dir("symlink");
    let target = dir.join("real.safetensors");
    std::fs::write(&target, &bytes).expect("write target");
    let link = dir.join("link.safetensors");
    std::os::unix::fs::symlink(&target, &link).expect("create symlink");

    match load_validated(&link, &strict_policy()) {
        Err(KatakuriError::CheckpointNotRegularFile { path }) => {
            assert!(path.contains("link.safetensors"), "{path}");
        }
        other => panic!("expected CheckpointNotRegularFile, got {other:?}"),
    }

    // Nothing is silently on: with the policy off, the symlink is read.
    let loaded = load_validated(&link, &permissive_policy()).expect("symlink allowed by policy");
    assert_eq!(loaded.tensors.len(), 4);

    // A hardlink IS a regular file and is accepted even under the policy.
    let hard = dir.join("hard.safetensors");
    std::fs::hard_link(&target, &hard).expect("create hardlink");
    load_validated(&hard, &strict_policy()).expect("hardlinks are regular files");
}

#[cfg(unix)]
#[test]
fn fifos_reject_under_the_regular_file_policy() {
    let dir = temp_dir("fifo");
    let fifo = dir.join("pipe.safetensors");
    let status = std::process::Command::new("mkfifo").arg(&fifo).status();
    match status {
        Ok(status) if status.success() => {}
        _ => {
            eprintln!("skipping fifo test: mkfifo unavailable");
            return;
        }
    }

    match load_validated(&fifo, &strict_policy()) {
        Err(KatakuriError::CheckpointNotRegularFile { path }) => {
            assert!(path.contains("pipe.safetensors"), "{path}");
        }
        // The pre-check must reject before any open, so this arm is a
        // failure, not a hang.
        other => panic!("expected CheckpointNotRegularFile, got {other:?}"),
    }
}

// ── Feature / type agreements ────────────────────────────────────

/// `TensorManifest` and the spec types are pure data: this test only needs
/// `digest` because it loads a checkpoint, but the assertion is that the
/// manifest's shape is fully described by the model layer.
#[test]
fn manifest_is_derivable_from_the_model_layer_alone() {
    let spec = spec(4, &[5, 6], "gelu_tanh");
    let expected = spec.expected_tensors();
    let manifest = TensorManifest {
        entries: expected
            .iter()
            .map(|(name, shape)| katakuri::manifest::ManifestEntry {
                name: name.clone(),
                shape: shape.iter().map(|&d| d as u32).collect(),
                byte_offset: 0,
                byte_len: 0,
            })
            .collect(),
        dtype: ManifestDtype::F32,
        layout: ManifestLayout::RowMajorContiguous,
        determinism: DeterminismPolicy::pinned(),
    };
    assert_eq!(manifest.entries.len(), expected.len());
    assert_eq!(manifest.entries[0].name, "hidden.0.weight");
    assert_eq!(
        manifest.entries.last().expect("non-empty").name,
        "head.bias"
    );
}
