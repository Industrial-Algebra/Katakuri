// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Optimizer-state checkpoint convention (S6, wishlist W4) end to end.
//!
//! Everything here uses the public API. Synthetic optimizer-state artifacts
//! are built in memory (header + data section), written to a unique temp
//! path, and loaded through [`load_optimizer_state`], which composes the S5
//! [`TrustPolicy`] seam rather than bypassing it.
//!
//! Compiled only with `std` + `digest`: the loader's file-I/O and hash
//! surfaces (same gate as the S5 `checkpoint` module).

#![cfg(all(feature = "std", feature = "digest"))]

use katakuri::checkpoint::TrustPolicy;
use katakuri::digest::artifact_sha256;
use katakuri::error::{KatakuriError, NonfiniteKind};
use katakuri::model::{Activation, MlpLayerSpec, MlpSpec};
use katakuri::optimizer::{
    META_OPTIMIZER, META_OPTIMIZER_STATE_VERSION, META_STEP, MomentKind, OPTIMIZER_ADAMW,
    OPTIMIZER_STATE_VERSION, OptimizerState, OptimizerStateError, SUFFIX_FIRST_MOMENT,
    SUFFIX_SECOND_MOMENT, expected_optimizer_tensors, load_optimizer_state, moment_name,
    split_moment_name,
};
use katakuri::safetensors::Dtype;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Helpers ───────────────────────────────────────────────────────

/// Every policy check on (the S5 `strict_policy` counterpart).
fn strict_policy() -> TrustPolicy {
    TrustPolicy {
        regular_file_only: true,
        byte_cap: 1 << 20,
        verify_schema_hash: true,
        verify_model_hash: true,
        reject_nonfinite: true,
    }
}

/// Every policy check off — each flag is an explicit choice.
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
        "katakuri-optimizer-test-{}-{tag}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_temp(tag: &str, bytes: &[u8]) -> PathBuf {
    let path = temp_dir(tag).join("optimizer.safetensors");
    std::fs::write(&path, bytes).expect("write optimizer fixture");
    path
}

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

fn f32_bytes(count: usize, value: f32) -> Vec<u8> {
    std::iter::repeat_n(value.to_le_bytes(), count)
        .flatten()
        .collect()
}

fn f64_bytes(count: usize, value: f64) -> Vec<u8> {
    std::iter::repeat_n(value.to_le_bytes(), count)
        .flatten()
        .collect()
}

/// Builds a synthetic optimizer-state artifact for a spec: by default the
/// complete moment set (both moments per parameter) in canonical form, with
/// the convention metadata block.
struct StateBuilder {
    spec: MlpSpec,
    /// Also carry the parameter tensors (the "weights alongside moments"
    /// artifact shape).
    with_weights: bool,
    drops: Vec<String>,
    /// Replacements / additions, keyed by name.
    overrides: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    /// Metadata edits applied after the correct defaults are computed.
    edits: Vec<(String, Option<String>)>,
    /// Emit tensors in reversed physical order (canonicalization tests).
    reverse: bool,
}

impl StateBuilder {
    fn new(spec: MlpSpec) -> Self {
        Self {
            spec,
            with_weights: false,
            drops: Vec::new(),
            overrides: Vec::new(),
            edits: Vec::new(),
            reverse: false,
        }
    }

    /// Include the parameter tensors too (full weights + moments artifact).
    #[must_use]
    fn with_weights(mut self) -> Self {
        self.with_weights = true;
        self
    }

    /// Drop one tensor from the emitted set.
    #[must_use]
    fn drop_tensor(mut self, name: &str) -> Self {
        self.drops.push(name.to_string());
        self
    }

    /// Replace (or add) one tensor with explicit dtype/shape/payload.
    #[must_use]
    fn set_tensor(mut self, name: &str, dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> Self {
        self.overrides.push((name.to_string(), dtype, shape, data));
        self
    }

    /// Override (or add) one metadata entry.
    #[must_use]
    fn set_meta(mut self, key: &str, value: &str) -> Self {
        self.edits.push((key.to_string(), Some(value.to_string())));
        self
    }

    /// Remove one metadata entry.
    #[must_use]
    fn remove_meta(mut self, key: &str) -> Self {
        self.edits.push((key.to_string(), None));
        self
    }

    /// Emit the tensors in REVERSED physical order (offsets recompute): a
    /// load-legal non-canonical arrangement.
    #[must_use]
    fn reversed(mut self) -> Self {
        self.reverse = true;
        self
    }

    fn build(self) -> Vec<u8> {
        let mut map: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> = BTreeMap::new();
        for (name, shape) in self.spec.expected_tensors() {
            let count: usize = shape.iter().product();
            if self.with_weights {
                map.insert(
                    name.clone(),
                    (Dtype::F32, shape.clone(), f32_bytes(count, 0.0)),
                );
            }
            for kind in [MomentKind::First, MomentKind::Second] {
                map.insert(
                    moment_name(&name, kind),
                    (Dtype::F32, shape.clone(), f32_bytes(count, 0.0)),
                );
            }
        }
        for name in &self.drops {
            map.remove(name);
        }
        for (name, dtype, shape, data) in self.overrides {
            map.insert(name, (dtype, shape, data));
        }

        let mut names: Vec<String> = map.keys().cloned().collect();
        if self.reverse {
            names.reverse();
        }

        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for name in &names {
            let (dtype, shape, payload) = &map[name];
            let start = data.len();
            data.extend_from_slice(payload);
            header.insert(
                name.clone(),
                json!({
                    "dtype": dtype_name(*dtype),
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }

        let canonical = self.spec.canonical_json();
        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        metadata.insert("ia.spec".to_string(), canonical.clone());
        metadata.insert(
            "ia.schema-hash".to_string(),
            artifact_sha256(canonical.as_bytes()),
        );
        metadata.insert("ia.model-hash".to_string(), artifact_sha256(&data));
        metadata.insert(
            "ia.provenance.training-data".to_string(),
            artifact_sha256(b"synthetic pairwise traces"),
        );
        metadata.insert(
            META_OPTIMIZER_STATE_VERSION.to_string(),
            OPTIMIZER_STATE_VERSION.to_string(),
        );
        metadata.insert(META_OPTIMIZER.to_string(), OPTIMIZER_ADAMW.to_string());
        metadata.insert(META_STEP.to_string(), "42".to_string());
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

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        other => panic!("test builder does not emit {other:?}"),
    }
}

fn load(
    bytes: &[u8],
    tag: &str,
    policy: &TrustPolicy,
    spec: &MlpSpec,
) -> Result<katakuri::optimizer::ValidatedOptimizerState, OptimizerStateError> {
    let path = write_temp(tag, bytes);
    load_optimizer_state(&path, policy, spec)
}

fn expect_error(
    bytes: &[u8],
    tag: &str,
    policy: &TrustPolicy,
    spec: &MlpSpec,
) -> OptimizerStateError {
    match load(bytes, tag, policy, spec) {
        Ok(state) => panic!("expected rejection, got {state:?}"),
        Err(error) => error,
    }
}

// ── Naming: the frozen golden name set ────────────────────────────

#[test]
fn golden_moment_name_set_is_exact() {
    let spec = spec(2, &[3], "relu");
    assert_eq!(
        expected_optimizer_tensors(&spec),
        vec![
            ("head.bias.adamw.m".to_string(), vec![1]),
            ("head.bias.adamw.v".to_string(), vec![1]),
            ("head.weight.adamw.m".to_string(), vec![1, 3]),
            ("head.weight.adamw.v".to_string(), vec![1, 3]),
            ("hidden.0.bias.adamw.m".to_string(), vec![3]),
            ("hidden.0.bias.adamw.v".to_string(), vec![3]),
            ("hidden.0.weight.adamw.m".to_string(), vec![3, 2]),
            ("hidden.0.weight.adamw.v".to_string(), vec![3, 2]),
        ],
        "golden moment name set (lexicographic canonical order)"
    );
}

#[test]
fn golden_suffixes_are_frozen() {
    assert_eq!(SUFFIX_FIRST_MOMENT, ".adamw.m");
    assert_eq!(SUFFIX_SECOND_MOMENT, ".adamw.v");
    assert_eq!(
        moment_name("hidden.0.weight", MomentKind::First),
        "hidden.0.weight.adamw.m"
    );
    assert_eq!(
        moment_name("hidden.0.weight", MomentKind::Second),
        "hidden.0.weight.adamw.v"
    );
    // The naming rule in machine form: a name ending in a suffix is optimizer
    // state, never a parameter.
    assert_eq!(
        split_moment_name("hidden.0.weight.adamw.m"),
        Some(("hidden.0.weight", MomentKind::First))
    );
    assert_eq!(
        split_moment_name("head.bias.adamw.v"),
        Some(("head.bias", MomentKind::Second))
    );
    assert_eq!(split_moment_name("hidden.0.weight"), None);
    assert_eq!(
        split_moment_name(".adamw.m"),
        None,
        "an empty parameter part is not a moment"
    );
    let _ = (
        META_OPTIMIZER_STATE_VERSION,
        OPTIMIZER_STATE_VERSION,
        META_OPTIMIZER,
        META_STEP,
    );
}

#[test]
fn golden_name_set_is_lexicographic_and_matches_the_parameter_set() {
    let spec = spec(4, &[2, 5], "tanh");
    let moments = expected_optimizer_tensors(&spec);
    let names: Vec<&str> = moments.iter().map(|(name, _)| name.as_str()).collect();
    assert!(
        names.windows(2).all(|w| w[0] < w[1]),
        "moment names are lexicographically sorted: {names:?}"
    );
    // Every parameter contributes exactly two moments, same shape as the
    // parameter — the naming rule is the parameter name plus a suffix.
    let params = spec.expected_tensors();
    assert_eq!(moments.len(), 2 * params.len());
    for (param, shape) in &params {
        for kind in [MomentKind::First, MomentKind::Second] {
            let name = moment_name(param, kind);
            assert!(
                moments.contains(&(name.clone(), shape.clone())),
                "moment '{name}' carries the parameter shape {shape:?}"
            );
        }
    }
}

// ── Valid artifacts ───────────────────────────────────────────────

#[test]
fn complete_moment_set_validates_under_the_strict_policy() {
    let spec = spec(2, &[3], "relu");
    let state = load(
        &StateBuilder::new(spec.clone()).build(),
        "complete",
        &strict_policy(),
        &spec,
    )
    .expect("complete optimizer state validates");

    assert_eq!(state.optimizer_state_version, OPTIMIZER_STATE_VERSION);
    assert_eq!(state.step, 42, "step comes from metadata, not per-tensor");
    let names: Vec<&str> = state.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "head.bias.adamw.m",
            "head.bias.adamw.v",
            "head.weight.adamw.m",
            "head.weight.adamw.v",
            "hidden.0.bias.adamw.m",
            "hidden.0.bias.adamw.v",
            "hidden.0.weight.adamw.m",
            "hidden.0.weight.adamw.v",
        ],
        "validated tensors are in lexicographic canonical order"
    );
    assert!(state.tensors.iter().all(|t| t.dtype == Dtype::F32));
    let shapes: Vec<(&str, &[usize])> = state
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    assert_eq!(
        shapes,
        vec![
            ("head.bias.adamw.m", &[1][..]),
            ("head.bias.adamw.v", &[1][..]),
            ("head.weight.adamw.m", &[1, 3][..]),
            ("head.weight.adamw.v", &[1, 3][..]),
            ("hidden.0.bias.adamw.m", &[3][..]),
            ("hidden.0.bias.adamw.v", &[3][..]),
            ("hidden.0.weight.adamw.m", &[3, 2][..]),
            ("hidden.0.weight.adamw.v", &[3, 2][..]),
        ],
        "every moment carries its parameter's shape"
    );

    let total: usize = state.tensors.iter().map(|t| t.data.len()).sum();
    assert_eq!(state.canonical_data_bytes().len(), total);
    assert_eq!(total, 4 * (1 + 1 + 3 + 3 + 3 + 3 + 6 + 6));

    let (first, second) = state
        .moments_for("hidden.0.weight")
        .expect("both moments for hidden.0.weight");
    assert_eq!(first.name, "hidden.0.weight.adamw.m");
    assert_eq!(second.name, "hidden.0.weight.adamw.v");
    assert_eq!(first.shape, vec![3, 2]);
    assert!(state.moments_for("hidden.9.weight").is_none());
}

#[test]
fn weights_alongside_moments_validate() {
    // The wishlist's "alongside model weights" artifact shape: parameters
    // keep their plain names, moments add the suffix.
    let spec = spec(2, &[3], "relu");
    let state = load(
        &StateBuilder::new(spec.clone()).with_weights().build(),
        "alongside",
        &strict_policy(),
        &spec,
    )
    .expect("weights + moments validate");
    assert_eq!(
        state.tensors.len(),
        4 + 8,
        "parameters plus two moments each"
    );
    assert_eq!(state.step, 42);
}

#[test]
fn physical_order_does_not_change_the_canonical_result() {
    let spec = spec(3, &[4], "gelu_tanh");
    let canonical = StateBuilder::new(spec.clone()).build();
    let reversed = StateBuilder::new(spec.clone()).reversed().build();
    assert_ne!(
        canonical, reversed,
        "the fixtures really differ in physical order"
    );

    let a = load(&canonical, "canon-a", &permissive_policy(), &spec).expect("canonical load");
    let b = load(&reversed, "canon-b", &permissive_policy(), &spec).expect("reversed load");
    assert_eq!(
        a.canonical_data_bytes(),
        b.canonical_data_bytes(),
        "canonical order is a property of the validated view, not of the file"
    );
    let a_names: Vec<&str> = a.tensors.iter().map(|t| t.name.as_str()).collect();
    let b_names: Vec<&str> = b.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(a_names, b_names);
}

// ── All-or-nothing completeness ───────────────────────────────────

#[test]
fn missing_single_moment_rejects() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .drop_tensor("hidden.0.bias.adamw.v")
            .build(),
        "missing-v",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::MissingMoment { param, moment } => {
            assert_eq!(param, "hidden.0.bias");
            assert_eq!(moment, MomentKind::Second);
        }
        other => panic!("expected MissingMoment, got {other:?}"),
    }
}

#[test]
fn partial_state_for_a_multi_layer_spec_rejects() {
    let spec = spec(4, &[3, 2, 5], "gelu_erf");
    // A file holding only the first layer's moments (a truncated resume).
    let mut builder = StateBuilder::new(spec.clone());
    for (param, _) in spec.expected_tensors() {
        if !param.starts_with("hidden.0.") {
            builder = builder
                .drop_tensor(&moment_name(&param, MomentKind::First))
                .drop_tensor(&moment_name(&param, MomentKind::Second));
        }
    }
    let error = expect_error(&builder.build(), "partial", &permissive_policy(), &spec);
    match error {
        OptimizerStateError::MissingMoment { param, moment } => {
            assert_eq!(param, "hidden.1.weight");
            assert_eq!(
                moment,
                MomentKind::First,
                "the first divergence in parameter order"
            );
        }
        other => panic!("expected MissingMoment, got {other:?}"),
    }
}

#[test]
fn parameter_subset_rejects() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .with_weights()
            .drop_tensor("head.bias")
            .build(),
        "param-subset",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::PartialParameters { detail } => {
            assert!(
                detail.contains("head.bias"),
                "names the missing parameter: {detail}"
            );
        }
        other => panic!("expected PartialParameters, got {other:?}"),
    }
}

// ── Dtype / shape agreement with the parameter ────────────────────

#[test]
fn moment_shape_mismatch_rejects() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor(
                "hidden.0.bias.adamw.m",
                Dtype::F32,
                vec![2],
                f32_bytes(2, 0.0),
            )
            .build(),
        "shape",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::MomentShapeMismatch {
            tensor,
            expected,
            actual,
        } => {
            assert_eq!(tensor, "hidden.0.bias.adamw.m");
            assert_eq!(expected, vec![3], "the parameter's shape");
            assert_eq!(actual, vec![2]);
        }
        other => panic!("expected MomentShapeMismatch, got {other:?}"),
    }
}

#[test]
fn moment_dtype_mismatch_rejects() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor(
                "head.weight.adamw.v",
                Dtype::F64,
                vec![1, 3],
                f64_bytes(3, 0.0),
            )
            .build(),
        "dtype",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::MomentDtypeMismatch {
            tensor,
            expected,
            actual,
        } => {
            assert_eq!(tensor, "head.weight.adamw.v");
            assert_eq!(expected, Dtype::F32, "the parameter's dtype");
            assert_eq!(actual, Dtype::F64);
        }
        other => panic!("expected MomentDtypeMismatch, got {other:?}"),
    }
}

// ── Metadata checks ───────────────────────────────────────────────

#[test]
fn missing_metadata_keys_reject() {
    let spec = spec(2, &[3], "relu");
    for key in [META_OPTIMIZER_STATE_VERSION, META_OPTIMIZER, META_STEP] {
        let error = expect_error(
            &StateBuilder::new(spec.clone()).remove_meta(key).build(),
            "meta-missing",
            &permissive_policy(),
            &spec,
        );
        match error {
            OptimizerStateError::MissingMetadata { key: missing } => assert_eq!(missing, key),
            other => panic!("expected MissingMetadata for {key}, got {other:?}"),
        }
    }
}

#[test]
fn unsupported_variants_reject() {
    let spec = spec(2, &[3], "relu");

    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta(META_OPTIMIZER_STATE_VERSION, "2")
            .build(),
        "version",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::UnsupportedStateVersion { found } => assert_eq!(found, 2),
        other => panic!("expected UnsupportedStateVersion, got {other:?}"),
    }

    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta(META_OPTIMIZER, "sgd")
            .build(),
        "optimizer",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::UnsupportedOptimizer { found } => assert_eq!(found, "sgd"),
        other => panic!("expected UnsupportedOptimizer, got {other:?}"),
    }
}

#[test]
fn malformed_metadata_values_reject() {
    let spec = spec(2, &[3], "relu");
    for (key, value) in [
        (META_OPTIMIZER_STATE_VERSION, "one"),
        (META_STEP, "later"),
        (META_STEP, "-1"),
    ] {
        let error = expect_error(
            &StateBuilder::new(spec.clone()).set_meta(key, value).build(),
            "meta-malformed",
            &permissive_policy(),
            &spec,
        );
        match error {
            OptimizerStateError::MalformedMetadata { key: k, .. } => assert_eq!(k, key),
            other => panic!("expected MalformedMetadata for {key}={value}, got {other:?}"),
        }
    }
}

#[test]
fn zero_step_is_legal() {
    let spec = spec(2, &[3], "relu");
    let state = load(
        &StateBuilder::new(spec.clone())
            .set_meta(META_STEP, "0")
            .build(),
        "step-zero",
        &permissive_policy(),
        &spec,
    )
    .expect("a resume before the first update is legal");
    assert_eq!(state.step, 0);
}

#[test]
fn foreign_spec_metadata_rejects() {
    let subject = spec(2, &[3], "relu");
    let foreign = spec(5, &[2], "tanh");
    let error = expect_error(
        &StateBuilder::new(subject.clone())
            .set_meta("ia.spec", &foreign.canonical_json())
            .build(),
        "spec-mismatch",
        &permissive_policy(),
        &subject,
    );
    match error {
        OptimizerStateError::SpecMismatch { detail } => {
            assert!(!detail.is_empty(), "detail explains the divergence");
        }
        other => panic!("expected SpecMismatch, got {other:?}"),
    }
}

// ── Collision / ambiguity rejection ───────────────────────────────

#[test]
fn nested_suffix_names_reject() {
    // "a name ending in .adamw.m is never a parameter": a nested suffix is
    // neither a parameter nor a moment of this spec.
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor(
                "hidden.0.weight.adamw.m.adamw.m",
                Dtype::F32,
                vec![3, 2],
                f32_bytes(6, 0.0),
            )
            .build(),
        "nested",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::UnexpectedTensor { name } => {
            assert_eq!(name, "hidden.0.weight.adamw.m.adamw.m");
        }
        other => panic!("expected UnexpectedTensor, got {other:?}"),
    }
}

#[test]
fn suffix_tensor_without_a_parameter_rejects() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor(
                "hidden.7.bias.adamw.m",
                Dtype::F32,
                vec![3],
                f32_bytes(3, 0.0),
            )
            .build(),
        "stray",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::UnexpectedTensor { name } => {
            assert_eq!(name, "hidden.7.bias.adamw.m");
        }
        other => panic!("expected UnexpectedTensor, got {other:?}"),
    }
}

#[test]
fn moment_shaped_tensor_at_a_parameter_name_rejects() {
    // A bare parameter name inside a moments-only artifact is ambiguous, not
    // silently accepted.
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor("head.bias", Dtype::F32, vec![1], f32_bytes(1, 0.0))
            .build(),
        "param-in-moments",
        &permissive_policy(),
        &spec,
    );
    match error {
        OptimizerStateError::PartialParameters { detail } => {
            assert!(
                detail.contains("found 1 of 4"),
                "reports the incomplete parameter set: {detail}"
            );
            assert!(
                !detail.contains("missing head.bias"),
                "head.bias is present; only its siblings are missing: {detail}"
            );
        }
        other => panic!("expected PartialParameters, got {other:?}"),
    }
}

// ── Trust-policy composition (never bypassed) ─────────────────────

#[test]
fn policy_file_checks_apply_to_optimizer_state() {
    let spec = spec(2, &[3], "relu");
    let bytes = StateBuilder::new(spec.clone()).build();

    let tiny_cap = TrustPolicy {
        byte_cap: 64,
        ..permissive_policy()
    };
    let error = expect_error(&bytes, "cap", &tiny_cap, &spec);
    assert!(
        matches!(
            error,
            OptimizerStateError::Checkpoint(KatakuriError::CheckpointTooLarge { .. })
        ),
        "byte cap applies: {error:?}"
    );

    #[cfg(unix)]
    {
        let dir = temp_dir("symlink");
        let target = dir.join("real.safetensors");
        std::fs::write(&target, &bytes).expect("write target");
        let link = dir.join("link.safetensors");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let error = match load_optimizer_state(&link, &strict_policy(), &spec) {
            Ok(state) => panic!("expected rejection, got {state:?}"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                OptimizerStateError::Checkpoint(KatakuriError::CheckpointNotRegularFile { .. })
            ),
            "regular-file check applies: {error:?}"
        );
    }
}

#[test]
fn policy_hash_checks_apply_to_optimizer_state() {
    let spec = spec(2, &[3], "relu");

    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta("ia.model-hash", &"0".repeat(64))
            .build(),
        "model-hash",
        &strict_policy(),
        &spec,
    );
    assert!(
        matches!(
            error,
            OptimizerStateError::Checkpoint(KatakuriError::ModelHashMismatch { .. })
        ),
        "model hash is verified: {error:?}"
    );

    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta("ia.schema-hash", &"0".repeat(64))
            .build(),
        "schema-hash",
        &strict_policy(),
        &spec,
    );
    assert!(
        matches!(
            error,
            OptimizerStateError::Checkpoint(KatakuriError::SchemaHashMismatch { .. })
        ),
        "schema hash is verified against the caller's spec: {error:?}"
    );

    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta("ia.bogus", "anything")
            .build(),
        "unknown-ia",
        &permissive_policy(),
        &spec,
    );
    assert!(
        matches!(
            error,
            OptimizerStateError::Checkpoint(KatakuriError::UnknownIaMetadata { .. })
        ),
        "the S5 ia.* metadata convention applies: {error:?}"
    );
}

#[test]
fn nonfinite_moments_reject_only_when_the_policy_asks() {
    let spec = spec(2, &[3], "relu");
    let bytes = StateBuilder::new(spec.clone())
        .set_tensor(
            "hidden.0.weight.adamw.v",
            Dtype::F32,
            vec![3, 2],
            f32_bytes(6, f32::NAN),
        )
        .build();

    let error = expect_error(&bytes, "nan-strict", &strict_policy(), &spec);
    match error {
        OptimizerStateError::Checkpoint(KatakuriError::NonfiniteTensorValue { tensor, which }) => {
            assert_eq!(tensor, "hidden.0.weight.adamw.v");
            assert_eq!(which, NonfiniteKind::Nan);
        }
        other => panic!("expected NonfiniteTensorValue, got {other:?}"),
    }

    // Policy off: values are the compute layer's question, and S1's parse
    // stays value-blind.
    let state = load(&bytes, "nan-permissive", &permissive_policy(), &spec)
        .expect("policy off accepts nonfinite moments");
    assert_eq!(state.tensors.len(), 8);
}

// ── The parse seam stays value-blind ──────────────────────────────

#[test]
fn parse_is_value_blind_and_unvalidated() {
    let spec = spec(2, &[3], "relu");
    let bytes = StateBuilder::new(spec.clone())
        .set_tensor(
            "head.bias.adamw.m",
            Dtype::F32,
            vec![1],
            f32_bytes(1, f32::INFINITY),
        )
        .remove_meta(META_STEP)
        .build();

    let parsed = OptimizerState::parse(&bytes).expect("parse accepts any S1-valid file");
    assert_eq!(
        parsed.tensors().len(),
        8,
        "tensors are exposed in canonical order"
    );
    assert!(
        parsed.metadata().get(META_STEP).is_none(),
        "metadata is exposed but not interpreted"
    );
    // Validation is a separate, explicit step.
    let error = parsed
        .validate_optimizer_state(&spec)
        .expect_err("missing step metadata rejects at validation");
    assert!(matches!(error, OptimizerStateError::MissingMetadata { .. }));
}

// ── The S5 fixture spec is a legal optimizer-state subject ────────

#[test]
fn tiny_fixture_spec_moment_set() {
    let spec = spec(2, &[3], "relu");
    assert_eq!(spec.hidden, vec![MlpLayerSpec { width: 3 }]);
    assert_eq!(spec.head, MlpLayerSpec { width: 1 });
    assert_eq!(spec.hidden_activation, Activation::Relu);
    let state = load(
        &StateBuilder::new(spec.clone()).build(),
        "fixture-spec",
        &strict_policy(),
        &spec,
    )
    .expect("the S5 fixture spec validates as an optimizer-state subject");
    assert_eq!(state.optimizer_state_version, 1);
}

// ── Round 1 review: bounded diagnostics (P2a) ─────────────────────
//
// The doctrine: a rejection never copies unbounded untrusted input into its
// message. Every file-derived string or shape echoed by a diagnostic is
// capped at a documented small constant, truncated on a UTF-8 boundary,
// escaped, and length-stamped — so a 100,000-byte metadata value (or tensor
// name, or shape rank) still produces a small message.

/// No optimizer diagnostic may exceed this many bytes, whatever the input.
const MAX_DIAGNOSTIC_BYTES: usize = 1024;

/// Assert the diagnostic stays small and render a bounded excerpt for the
/// failure message (a panicking assertion must not itself dump 100 KB).
fn assert_bounded(error: &OptimizerStateError) {
    let message = error.to_string();
    assert!(
        message.len() <= MAX_DIAGNOSTIC_BYTES,
        "diagnostic is {} bytes (cap {MAX_DIAGNOSTIC_BYTES}); first 120 chars: {:?}",
        message.len(),
        message.chars().take(120).collect::<String>()
    );
}

#[test]
fn malformed_metadata_diagnostic_is_bounded() {
    let spec = spec(2, &[3], "relu");
    let huge = "x".repeat(100_000);
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta(META_STEP, &huge)
            .build(),
        "huge-step",
        &permissive_policy(),
        &spec,
    );
    assert!(matches!(
        error,
        OptimizerStateError::MalformedMetadata { .. }
    ));
    assert_bounded(&error);
    assert!(
        error.to_string().contains("100000"),
        "the true length is stamped: {}",
        error
    );
}

#[test]
fn metadata_diagnostic_truncates_on_a_utf8_boundary() {
    let spec = spec(2, &[3], "relu");
    // 2 bytes per char: a naive 64-byte slice would split a `char`.
    let widthful = "é".repeat(5_000);
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta(META_OPTIMIZER_STATE_VERSION, &widthful)
            .build(),
        "utf8-step",
        &permissive_policy(),
        &spec,
    );
    assert!(matches!(
        error,
        OptimizerStateError::MalformedMetadata { .. }
    ));
    assert_bounded(&error);
}

#[test]
fn diagnostics_escape_control_characters() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta(META_STEP, "12\n34")
            .build(),
        "control-step",
        &permissive_policy(),
        &spec,
    );
    let message = error.to_string();
    assert_bounded(&error);
    assert!(
        !message.contains('\n'),
        "a diagnostic is one line — control characters are escaped: {message:?}"
    );
}

#[test]
fn unexpected_tensor_name_diagnostic_is_bounded() {
    let spec = spec(2, &[3], "relu");
    let huge_name = format!("{}{SUFFIX_FIRST_MOMENT}", "n".repeat(100_000));
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor(&huge_name, Dtype::F32, vec![1], f32_bytes(1, 0.0))
            .build(),
        "huge-name",
        &permissive_policy(),
        &spec,
    );
    assert!(matches!(
        error,
        OptimizerStateError::UnexpectedTensor { .. }
    ));
    assert_bounded(&error);
    assert!(
        error.to_string().contains("100000"),
        "the true name length is stamped: {}",
        error
    );
}

#[test]
fn moment_shape_diagnostic_is_bounded_for_huge_ranks() {
    let spec = spec(2, &[3], "relu");
    // Rank 100_000 of `1`s: a legal S1 shape (product 1, so a 4-byte span)
    // and a perfectly legal way to make `{:?}` of a shape enormous.
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_tensor(
                "head.bias.adamw.m",
                Dtype::F32,
                vec![1; 100_000],
                f32_bytes(1, 0.0),
            )
            .build(),
        "huge-rank",
        &permissive_policy(),
        &spec,
    );
    assert!(matches!(
        error,
        OptimizerStateError::MomentShapeMismatch { .. }
    ));
    assert_bounded(&error);
    assert!(
        error.to_string().contains("100000"),
        "the true rank is stamped: {}",
        error
    );
}

#[test]
fn parameter_mismatch_diagnostic_is_bounded_for_huge_ranks() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .with_weights()
            .set_tensor("head.bias", Dtype::F32, vec![1; 100_000], f32_bytes(1, 0.0))
            .build(),
        "huge-rank-param",
        &permissive_policy(),
        &spec,
    );
    assert!(matches!(
        error,
        OptimizerStateError::ParameterMismatch { .. }
    ));
    assert_bounded(&error);
}

#[test]
fn unsupported_optimizer_diagnostic_is_bounded() {
    let spec = spec(2, &[3], "relu");
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta(META_OPTIMIZER, &"o".repeat(100_000))
            .build(),
        "huge-optimizer",
        &permissive_policy(),
        &spec,
    );
    assert!(matches!(
        error,
        OptimizerStateError::UnsupportedOptimizer { .. }
    ));
    assert_bounded(&error);
}

// ── Round 1 review: semantic spec comparison (P2b) ────────────────

#[test]
fn whitespace_equivalent_declared_spec_is_accepted() {
    let spec = spec(2, &[3], "relu");
    let canonical = spec.canonical_json();
    let pretty = serde_json::to_string_pretty(
        &serde_json::from_str::<Value>(&canonical).expect("canonical JSON parses"),
    )
    .expect("re-serialize");
    assert_ne!(pretty, canonical, "the fixtures really differ textually");

    let state = load(
        &StateBuilder::new(spec.clone())
            .set_meta("ia.spec", &pretty)
            .build(),
        "spec-pretty",
        &strict_policy(),
        &spec,
    )
    .expect("whitespace-equivalent declared spec is the same spec");
    assert_eq!(state.step, 42);
}

#[test]
fn reordered_key_declared_spec_is_accepted() {
    let spec = spec(2, &[3], "relu");
    let reordered = r#"{
        "hidden_activation" : "relu" ,
        "head" : { "width" : 1 } ,
        "hidden" : [ { "width" : 3 } ] ,
        "input_width" : 2 ,
        "schema_version" : 1 ,
        "model_type" : "ia-mlp"
    }"#;
    let state = load(
        &StateBuilder::new(spec.clone())
            .set_meta("ia.spec", reordered)
            .build(),
        "spec-reordered",
        &permissive_policy(),
        &spec,
    )
    .expect("key order is not semantic");
    assert_eq!(state.step, 42);
}

#[test]
fn malformed_declared_spec_is_not_a_mismatch() {
    let spec = spec(2, &[3], "relu");
    for (tag, declared) in [
        ("spec-junk", "not json at all"),
        ("spec-bert", r#"{"model_type":"bert","schema_version":1}"#),
        ("spec-empty", ""),
    ] {
        let error = expect_error(
            &StateBuilder::new(spec.clone())
                .set_meta("ia.spec", declared)
                .build(),
            tag,
            &permissive_policy(),
            &spec,
        );
        assert!(
            !matches!(error, OptimizerStateError::SpecMismatch { .. }),
            "a malformed declared spec is its own typed error, not a mismatch ({tag}): {error:?}"
        );
        assert_bounded(&error);
    }
}

#[test]
fn malformed_declared_spec_diagnostic_is_bounded() {
    let spec = spec(2, &[3], "relu");
    // Legal JSON, illegal spec: the activation string is the bad value and it
    // is 100 KB long — the MlpSpec parser quotes it in its own detail.
    let bad = json!({
        "model_type": "ia-mlp",
        "schema_version": 1,
        "input_width": 2,
        "hidden": [{"width": 3}],
        "head": {"width": 1},
        "hidden_activation": "a".repeat(100_000),
    })
    .to_string();
    let error = expect_error(
        &StateBuilder::new(spec.clone())
            .set_meta("ia.spec", &bad)
            .build(),
        "spec-huge-value",
        &permissive_policy(),
        &spec,
    );
    assert!(
        !matches!(error, OptimizerStateError::SpecMismatch { .. }),
        "typed as a malformed spec: {error:?}"
    );
    assert_bounded(&error);
}
