// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Optimizer-state checkpoint convention (S6, wishlist W4 — Amari Task 42).
//!
//! Katakuri never computes — no autodiff, no optimizer math (S5 non-goals
//! stand). What it *does* own is the **shape of a resumable training
//! artifact**: where AdamW's first/second moments live, what they are called,
//! what they must agree with, and when a checkpoint is complete. The frozen
//! convention itself is `docs/optimizer-state-convention.md`; this module is
//! its executable half.
//!
//! # The convention in one paragraph
//!
//! Optimizer state is namespaced **per parameter tensor**: the parameter keeps
//! its plain name (`"hidden.0.weight"`), and AdamW's moments are that name
//! plus a reserved suffix — `".adamw.m"` (first moment) and `".adamw.v"`
//! (second moment). Each moment carries the **same dtype and the same shape**
//! as its parameter (no transposes at rest, no widened f64 accumulators at
//! rest). `__metadata__` carries `optimizer_state_version` (`1` for this
//! convention), `optimizer` (`"adamw"`), and `step` (a `u64` — the step
//! counter is metadata, never a per-tensor). Tensors are read into
//! **lexicographic canonical order**, so the validated view is
//! byte-reproducible regardless of the file's physical layout. Validity is
//! **all-or-nothing**: every parameter must have *both* moments, or the
//! artifact is rejected with a typed error — never partially trusted.
//!
//! # Two accepted artifact shapes
//!
//! [`OptimizerState::validate_optimizer_state`] accepts, and only accepts:
//!
//! 1. **Moments only** — the tensor set is exactly the derived moment set.
//!    The parameter tensors live in the model checkpoint, and the spec is the
//!    shape/dtype oracle.
//! 2. **Weights alongside moments** — the tensor set is exactly the full
//!    parameter set *plus* the full moment set (the wishlist's "alongside
//!    model weights" artifact). A parameter tensor present without its
//!    siblings is [`OptimizerStateError::PartialParameters`]: partial weights
//!    plus complete moments is ambiguous, so it is never inferred.
//!
//! Anything else is [`OptimizerStateError::UnexpectedTensor`] — including a
//! name ending in a moment suffix that is not a moment of this spec (nested
//! suffixes such as `"hidden.0.weight.adamw.m.adamw.m"` land here). That is
//! the collision rule: **a name ending in `.adamw.m` or `.adamw.v` is never a
//! parameter**.
//!
//! # Composition with S5, never bypass
//!
//! [`load_optimizer_state`] reuses the S5 [`TrustPolicy`] pipeline
//! ([`crate::checkpoint::load_validated`]'s seam) — regular-file check, byte
//! cap, the `ia.*` metadata convention, and the policy-gated hash and
//! nonfinite checks all apply unchanged. It does **not** re-implement safe
//! opening or value scanning, and it does not call
//! [`SafeTensors::parse`](crate::safetensors::SafeTensors::parse) with values
//! inspected: [`OptimizerState::parse`] is value-blind, exactly like S1.
//!
//! # Pipeline (ordered, failure is total)
//!
//! 1. **Trust policy** — read through [`crate::checkpoint::TrustPolicy`]
//!    (`regular_file_only`, `byte_cap`).
//! 2. **S1 parse** — value-blind, full safetensors validation.
//! 3. **`ia.*` metadata convention** (S5 §5.5): unknown `ia.*` keys reject.
//! 4. **Optimizer metadata** — `optimizer_state_version` = 1, `optimizer` =
//!    `"adamw"`, `step` parses as a `u64`; `ia.spec`, when present, must equal
//!    the caller's spec.
//! 5. **All-or-nothing moments** — both moments for every parameter.
//! 6. **No unexpected tensors** — parameters only as a complete set.
//! 7. **Dtype/shape agreement** — moments equal their parameter's dtype
//!    (F32) and shape; parameter tensors, when present, equal the spec.
//! 8. **Hashes** (`verify_model_hash`, `verify_schema_hash`) — before the
//!    value scan, so a tampered artifact reports a hash mismatch rather than
//!    a value diagnostic.
//! 9. **Nonfinite scan** (`reject_nonfinite`): NaN / ±Infinity in a moment
//!    rejects. Subnormals are finite and are kept (S5 doctrine).
//!
//! Feature gates: `std` **and** `digest` — the same gate as `checkpoint`,
//! because this module is file I/O plus digest verification on top of it.

use crate::checkpoint::{
    META_MODEL_HASH, META_SCHEMA_HASH, META_SPEC, ParsedTensor, TrustPolicy,
    check_metadata_convention, data_section, read_checkpoint,
};
use crate::digest::artifact_sha256;
use crate::error::{KatakuriError, NonfiniteKind};
use crate::model::MlpSpec;
use crate::safetensors::{Dtype, SafeTensors};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use thiserror::Error;

/// Suffix marking AdamW's **first moment** (the exponential moving average of
/// the gradient) on a parameter tensor's name.
pub const SUFFIX_FIRST_MOMENT: &str = ".adamw.m";

/// Suffix marking AdamW's **second moment** (the exponential moving average
/// of the squared gradient) on a parameter tensor's name.
pub const SUFFIX_SECOND_MOMENT: &str = ".adamw.v";

/// `__metadata__` key carrying the convention version (`u32`). This
/// convention is version [`OPTIMIZER_STATE_VERSION`].
pub const META_OPTIMIZER_STATE_VERSION: &str = "optimizer_state_version";

/// `__metadata__` key naming the optimizer whose state this is. This
/// convention covers exactly one value: [`OPTIMIZER_ADAMW`].
pub const META_OPTIMIZER: &str = "optimizer";

/// `__metadata__` key carrying the training step counter (`u64`) the state
/// belongs to. Step is metadata, never a per-tensor.
pub const META_STEP: &str = "step";

/// The one optimizer name this convention covers.
pub const OPTIMIZER_ADAMW: &str = "adamw";

/// The one convention version this module validates.
pub const OPTIMIZER_STATE_VERSION: u32 = 1;

/// Byte cap on untrusted text quoted into a diagnostic (the
/// `MAX_DIAGNOSTIC_NAME_BYTES` doctrine: a rejection never copies an
/// unbounded input just to say "rejected").
///
/// Metadata values and tensor names come straight out of the file, and a
/// safetensors header is allowed to be 100 MB. Quoting one verbatim turns a
/// one-line rejection into a 100 KB allocation (PR #22 round 1, P2a).
pub const MAX_DIAGNOSTIC_EXCERPT_BYTES: usize = 64;

/// Largest number of names rendered into a diagnostic list.
///
/// A missing-name list is *not* spec-grammar-bounded: [`MlpSpec`]'s fields
/// are public, so a hand-built 10,000-layer spec is legal input and derives
/// 20,002 parameters (PR #22 round 2). Lists are therefore capped at this
/// many entries and stamped with the true remainder.
pub const MAX_DIAGNOSTIC_NAME_LIST: usize = 4;

/// Largest shape rank rendered into a diagnostic.
///
/// S1 does not bound a shape's rank — `[1, 1, … 1]` (100,000 dims) is a legal
/// shape with a 4-byte span — so `{:?}` of an untrusted shape is as unbounded
/// as an untrusted string. Rendered shapes keep at most this many dims and
/// stamp the true rank.
pub const MAX_DIAGNOSTIC_SHAPE_DIMS: usize = 8;

/// Render untrusted text for a diagnostic.
///
/// At most [`MAX_DIAGNOSTIC_EXCERPT_BYTES`] input bytes are kept (truncated on
/// a UTF-8 boundary, so multi-byte characters are never split), control
/// characters are escaped so a diagnostic stays one line, and a truncated
/// excerpt is stamped with the input's true byte length. Short printable
/// input is returned unchanged.
///
/// Bounded output: escaping can expand one control character to as much as
/// `\u{10ffff}`, so the result is at most
/// `MAX_DIAGNOSTIC_EXCERPT_BYTES × 10 + ~32` bytes.
#[must_use]
pub fn bounded_excerpt(text: &str) -> String {
    let (kept, truncated) = if text.len() <= MAX_DIAGNOSTIC_EXCERPT_BYTES {
        (text, false)
    } else {
        let mut end = MAX_DIAGNOSTIC_EXCERPT_BYTES;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        (&text[..end], true)
    };
    let mut out = String::with_capacity(kept.len() + 32);
    out.extend(kept.chars().flat_map(char::escape_debug));
    if truncated {
        let true_len = format!("... ({} bytes)", text.len());
        out.push_str(&true_len);
    }
    out
}

/// Render a shape for a diagnostic: at most [`MAX_DIAGNOSTIC_SHAPE_DIMS`]
/// dims, with the true rank stamped when truncated (`[1, 1, ...] (100000
/// dims)`). Short shapes render exactly as `{:?}` renders them.
#[must_use]
pub fn shape_excerpt(shape: &[usize]) -> String {
    let shown = shape.len().min(MAX_DIAGNOSTIC_SHAPE_DIMS);
    let mut out = String::with_capacity(8 * 8 + 24);
    out.push('[');
    for (index, dim) in shape[..shown].iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&dim.to_string());
    }
    if shape.len() > shown {
        let tail = format!(", ...] ({} dims)", shape.len());
        out.push_str(&tail);
    } else {
        out.push(']');
    }
    out
}

/// Render a list of names for a diagnostic: at most
/// [`MAX_DIAGNOSTIC_NAME_LIST`] entries, each bounded by [`bounded_excerpt`],
/// then `... and N more` with the true remainder.
///
/// Provenance-free by construction: it holds no matter how the list was
/// produced, which is the point — a spec is a public struct, not a
/// parse-validated document.
#[must_use]
pub fn name_list_excerpt(names: &[&str]) -> String {
    let shown = names.len().min(MAX_DIAGNOSTIC_NAME_LIST);
    let mut out = String::with_capacity(shown * (MAX_DIAGNOSTIC_EXCERPT_BYTES + 2) + 32);
    for (index, name) in names[..shown].iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&bounded_excerpt(name));
    }
    if names.len() > shown {
        let tail = format!("... and {} more", names.len() - shown);
        if shown > 0 {
            out.push_str(", ");
        }
        out.push_str(&tail);
    }
    out
}

/// Which of AdamW's two moments a tensor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MomentKind {
    /// First moment `m` — the EMA of the gradient (`".adamw.m"`).
    First,
    /// Second moment `v` — the EMA of the squared gradient (`".adamw.v"`).
    Second,
}

impl MomentKind {
    /// The reserved name suffix for this moment.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::First => SUFFIX_FIRST_MOMENT,
            Self::Second => SUFFIX_SECOND_MOMENT,
        }
    }
}

impl std::fmt::Display for MomentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::First => "first moment (.adamw.m)",
            Self::Second => "second moment (.adamw.v)",
        };
        f.write_str(text)
    }
}

/// The canonical tensor name of one moment of `param`: the parameter name
/// plus the moment's reserved suffix (`"hidden.0.weight.adamw.m"`).
///
/// The parameter keeps its plain name; the suffix namespace is reserved, so a
/// name ending in a suffix is never a parameter.
#[must_use]
pub fn moment_name(param: &str, moment: MomentKind) -> String {
    format!("{param}{}", moment.suffix())
}

/// Split a tensor name into `(parameter name, moment)` when it ends in a
/// reserved moment suffix, or `None` for a plain parameter name.
///
/// This is the naming rule's machine form: `".adamw.m"` / `".adamw.v"` names
/// are optimizer state, everything else is not. A suffix with an empty
/// parameter part (`".adamw.m"`) is not a moment and yields `None`.
#[must_use]
pub fn split_moment_name(name: &str) -> Option<(&str, MomentKind)> {
    let split = name
        .strip_suffix(SUFFIX_FIRST_MOMENT)
        .map(|param| (param, MomentKind::First))
        .or_else(|| {
            name.strip_suffix(SUFFIX_SECOND_MOMENT)
                .map(|param| (param, MomentKind::Second))
        })?;
    if split.0.is_empty() {
        None
    } else {
        Some(split)
    }
}

/// The derived optimizer-state tensor set for `spec` as `(name, shape)`
/// pairs, in **lexicographic canonical order**.
///
/// Derived from [`MlpSpec::expected_tensors`]: every parameter contributes a
/// first- and a second-moment tensor of the parameter's shape. This is the
/// golden oracle a writer and a loader must agree on.
#[must_use]
pub fn expected_optimizer_tensors(spec: &MlpSpec) -> Vec<(String, Vec<usize>)> {
    let params = spec.expected_tensors();
    let mut out = Vec::with_capacity(2 * params.len());
    for (param, shape) in params {
        for moment in [MomentKind::First, MomentKind::Second] {
            out.push((moment_name(&param, moment), shape.clone()));
        }
    }
    out.sort();
    out
}

/// A parsed optimizer-state checkpoint: the `__metadata__` map and every
/// tensor in lexicographic canonical order.
///
/// [`OptimizerState::parse`] is S1's value-blind parse plus canonical
/// ordering — it asserts nothing about the convention. Call
/// [`OptimizerState::validate_optimizer_state`] for that, or
/// [`load_optimizer_state`] to do both under a [`TrustPolicy`].
#[derive(Clone, Debug)]
pub struct OptimizerState {
    metadata: BTreeMap<String, String>,
    tensors: Vec<ParsedTensor>,
}

impl OptimizerState {
    /// Parse an optimizer-state artifact held in `bytes`.
    ///
    /// Value-blind (S1 doctrine unchanged): layout, spans, and dtype strings
    /// are validated, values never are, and no convention check runs. Tensors
    /// come back in lexicographic canonical order.
    ///
    /// # Errors
    ///
    /// Any [`KatakuriError`] from S1's parse (truncated file, bad header,
    /// offsets out of bounds, …).
    pub fn parse(bytes: &[u8]) -> Result<Self, OptimizerStateError> {
        let parsed = SafeTensors::parse(bytes)?;
        Self::from_parsed(parsed.metadata().clone(), &parsed)
    }

    /// Collect every parsed tensor into canonical (lexicographic) order.
    fn from_parsed(
        metadata: BTreeMap<String, String>,
        parsed: &SafeTensors<'_>,
    ) -> Result<Self, OptimizerStateError> {
        let mut tensors = Vec::with_capacity(parsed.tensor_count());
        for name in parsed.names() {
            let tensor = parsed.tensor(name)?;
            tensors.push(ParsedTensor {
                name: name.to_string(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                data: tensor.bytes().to_vec(),
            });
        }
        Ok(Self { metadata, tensors })
    }

    /// The full `__metadata__` string map, uninterpreted.
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Every tensor in lexicographic canonical order.
    #[must_use]
    pub fn tensors(&self) -> &[ParsedTensor] {
        &self.tensors
    }

    /// Borrow one tensor by name, if present.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&ParsedTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Validate this view against the convention, using `spec` as the
    /// parameter oracle.
    ///
    /// Checks, in order: optimizer metadata (`optimizer_state_version` = 1,
    /// `optimizer` = `"adamw"`, `step` a `u64`, `ia.spec` agreeing with
    /// `spec` when present), the all-or-nothing moment set, unexpected
    /// tensors (including suffix-named collisions), and each moment's
    /// dtype/shape against its parameter.
    ///
    /// # Errors
    ///
    /// [`OptimizerStateError`] — one variant per violated rule, naming the
    /// offending tensor or metadata key.
    pub fn validate_optimizer_state(
        &self,
        spec: &MlpSpec,
    ) -> Result<ValidatedOptimizerState, OptimizerStateError> {
        let version = self.required_meta(META_OPTIMIZER_STATE_VERSION)?;
        let version: u32 = version
            .parse()
            .map_err(|_| OptimizerStateError::MalformedMetadata {
                key: META_OPTIMIZER_STATE_VERSION.to_string(),
                detail: format!("expected a u32, found '{}'", bounded_excerpt(version)),
            })?;
        if version != OPTIMIZER_STATE_VERSION {
            return Err(OptimizerStateError::UnsupportedStateVersion { found: version });
        }

        let optimizer = self.required_meta(META_OPTIMIZER)?;
        if optimizer != OPTIMIZER_ADAMW {
            return Err(OptimizerStateError::UnsupportedOptimizer {
                found: bounded_excerpt(optimizer),
            });
        }

        let step = self.required_meta(META_STEP)?;
        let step: u64 = step
            .parse()
            .map_err(|_| OptimizerStateError::MalformedMetadata {
                key: META_STEP.to_string(),
                detail: format!("expected a u64, found '{}'", bounded_excerpt(step)),
            })?;

        // The artifact is bound to a parameter checkpoint. When it declares
        // `ia.spec` itself, that declaration must agree — an optimizer state
        // for a different architecture is a silent-failure generator.
        //
        // Agreement is SEMANTIC, not textual (PR #22 round 1, P2b): the
        // declared document is parsed and compared in canonical form — the
        // same canonical JSON `ia.schema-hash` digests — so whitespace, key
        // order, and any tolerated extra field cannot produce a mismatch.
        // A declaration that is not a valid spec is its own typed failure,
        // never a mismatch.
        if let Some(declared) = self.metadata.get(META_SPEC) {
            let declared = match MlpSpec::from_json(declared) {
                Ok(parsed) => parsed,
                Err(KatakuriError::MlpSpecParse { detail }) => {
                    return Err(OptimizerStateError::MalformedSpecMetadata {
                        detail: bounded_excerpt(&detail),
                    });
                }
                Err(other) => {
                    // Defensive: `MlpSpec::from_json` only returns
                    // `MlpSpecParse` today; a future variant must not leak an
                    // unbounded message either.
                    return Err(OptimizerStateError::MalformedSpecMetadata {
                        detail: bounded_excerpt(&other.to_string()),
                    });
                }
            };
            if declared.canonical_json() != spec.canonical_json() {
                return Err(OptimizerStateError::SpecMismatch {
                    detail: "checkpoint metadata 'ia.spec' names a different architecture \
                             than the caller's parameter spec"
                        .to_string(),
                });
            }
        }

        let params = spec.expected_tensors();
        let present: BTreeSet<&str> = self.tensors.iter().map(|t| t.name.as_str()).collect();
        let param_names: BTreeSet<&str> = params.iter().map(|(name, _)| name.as_str()).collect();
        let moment_names: BTreeSet<String> = params
            .iter()
            .flat_map(|(param, _)| {
                [
                    moment_name(param, MomentKind::First),
                    moment_name(param, MomentKind::Second),
                ]
            })
            .collect();

        // All-or-nothing: every parameter carries both moments, in parameter
        // order so the first divergence is the one reported.
        for (param, _) in &params {
            for moment in [MomentKind::First, MomentKind::Second] {
                let name = moment_name(param, moment);
                if !present.contains(name.as_str()) {
                    return Err(OptimizerStateError::MissingMoment {
                        param: bounded_excerpt(param),
                        moment,
                    });
                }
            }
        }

        // Nothing outside the derived moment set (collisions included), and
        // parameters only as a COMPLETE set.
        let mut params_found = 0usize;
        for tensor in &self.tensors {
            let name = tensor.name.as_str();
            if moment_names.contains(name) {
                continue;
            }
            if param_names.contains(name) {
                params_found += 1;
                continue;
            }
            return Err(OptimizerStateError::UnexpectedTensor {
                name: bounded_excerpt(&tensor.name),
            });
        }
        if params_found != 0 && params_found != param_names.len() {
            let missing: Vec<&str> = param_names
                .iter()
                .copied()
                .filter(|name| !present.contains(name))
                .collect();
            return Err(OptimizerStateError::PartialParameters {
                detail: format!(
                    "parameter tensors must be absent or complete; found {} of {}; missing {}",
                    params_found,
                    param_names.len(),
                    name_list_excerpt(&missing)
                ),
            });
        }

        // Dtype/shape agreement. The parameter tensor is the oracle when it
        // is in the artifact; otherwise the spec is (they are equal by the
        // check below).
        for (param, shape) in &params {
            if params_found != 0 {
                let tensor = self
                    .tensor(param)
                    .expect("presence and completeness already established");
                if tensor.dtype != Dtype::F32 || tensor.shape != *shape {
                    return Err(OptimizerStateError::ParameterMismatch {
                        tensor: bounded_excerpt(param),
                        detail: format!(
                            "expected dtype F32 and shape {}, found dtype {:?} and shape {}",
                            shape_excerpt(shape),
                            tensor.dtype,
                            shape_excerpt(&tensor.shape)
                        ),
                    });
                }
            }
            for moment in [MomentKind::First, MomentKind::Second] {
                let name = moment_name(param, moment);
                let tensor = self
                    .tensor(&name)
                    .expect("presence already established above");
                if tensor.dtype != Dtype::F32 {
                    return Err(OptimizerStateError::MomentDtypeMismatch {
                        tensor: bounded_excerpt(&name),
                        expected: Dtype::F32,
                        actual: tensor.dtype,
                    });
                }
                if tensor.shape != *shape {
                    return Err(OptimizerStateError::MomentShapeMismatch {
                        tensor: bounded_excerpt(&name),
                        expected: shape_excerpt(shape),
                        actual: shape_excerpt(&tensor.shape),
                    });
                }
            }
        }

        Ok(ValidatedOptimizerState {
            optimizer_state_version: version,
            step,
            tensors: self.tensors.clone(),
        })
    }

    /// Fetch a required metadata key.
    fn required_meta(&self, key: &str) -> Result<&str, OptimizerStateError> {
        self.metadata.get(key).map(String::as_str).ok_or_else(|| {
            OptimizerStateError::MissingMetadata {
                key: key.to_string(),
            }
        })
    }
}

/// A convention-checked optimizer-state artifact.
///
/// Moments and (when the artifact carries them) parameter tensors are in
/// lexicographic canonical order; [`Self::canonical_data_bytes`] is the byte
/// witness of that order, independent of the file's physical layout.
#[derive(Clone, Debug)]
pub struct ValidatedOptimizerState {
    /// Convention version this artifact satisfied (always
    /// [`OPTIMIZER_STATE_VERSION`] today).
    pub optimizer_state_version: u32,
    /// Training step the state belongs to, from `__metadata__["step"]`.
    pub step: u64,
    /// Tensors in lexicographic canonical order: both moments per parameter,
    /// plus the parameter tensors when the artifact is the weights-alongside
    /// shape.
    pub tensors: Vec<ParsedTensor>,
}

impl ValidatedOptimizerState {
    /// The `(first, second)` moment tensors of one parameter, if present.
    #[must_use]
    pub fn moments_for(&self, param: &str) -> Option<(&ParsedTensor, &ParsedTensor)> {
        let first = moment_name(param, MomentKind::First);
        let second = moment_name(param, MomentKind::Second);
        let first = self.tensors.iter().find(|tensor| tensor.name == first)?;
        let second = self.tensors.iter().find(|tensor| tensor.name == second)?;
        Some((first, second))
    }

    /// The data section in canonical (lexicographic) tensor order: every
    /// tensor's bytes, concatenated. Two artifacts holding the same tensors
    /// in different physical layouts yield equal canonical bytes.
    #[must_use]
    pub fn canonical_data_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor.data);
        }
        out
    }
}

/// Load an optimizer-state checkpoint and validate it against `policy` and
/// `spec`.
///
/// The ordered pipeline is documented at the module level; failure is total
/// (no partial loads). Every [`TrustPolicy`] flag that applies to a
/// moments-only artifact applies here, through the same S5 code paths:
/// `regular_file_only`, `byte_cap`, `verify_model_hash` (over
/// `ia.model-hash` / the data section), `verify_schema_hash` (over
/// `ia.schema-hash` / `spec`'s canonical JSON), and `reject_nonfinite` (over
/// every moment).
///
/// # Errors
///
/// [`OptimizerStateError::Checkpoint`] wrapping any S5 failure (I/O, policy,
/// S1 parse, `ia.*` metadata, hashes, nonfinite values), or any
/// [`OptimizerStateError`] variant from
/// [`OptimizerState::validate_optimizer_state`].
pub fn load_optimizer_state(
    path: &Path,
    policy: &TrustPolicy,
    spec: &MlpSpec,
) -> Result<ValidatedOptimizerState, OptimizerStateError> {
    let bytes = read_checkpoint(path, policy)?;
    let parsed = SafeTensors::parse(&bytes)?;
    let metadata = parsed.metadata().clone();

    check_metadata_convention(&metadata)?;
    let state = OptimizerState::from_parsed(metadata, &parsed)?;

    let validated = state.validate_optimizer_state(spec)?;

    if policy.verify_model_hash {
        let actual = artifact_sha256(data_section(&bytes)?);
        let expected = state
            .metadata
            .get(META_MODEL_HASH)
            .map_or("<absent>", String::as_str);
        if expected != actual {
            return Err(KatakuriError::ModelHashMismatch {
                expected: expected.to_string(),
                actual,
            }
            .into());
        }
    }
    if policy.verify_schema_hash {
        let actual = artifact_sha256(spec.canonical_json().as_bytes());
        let expected = state
            .metadata
            .get(META_SCHEMA_HASH)
            .map_or("<absent>", String::as_str);
        if expected != actual {
            return Err(KatakuriError::SchemaHashMismatch {
                expected: expected.to_string(),
                actual,
            }
            .into());
        }
    }

    if policy.reject_nonfinite {
        scan_nonfinite(&validated.tensors)?;
    }

    Ok(validated)
}

/// Scan every F32 tensor for NaN / ±Infinity, in canonical order. Subnormals
/// are finite and pass (S5 doctrine: they are legal f32 values, and whether
/// the compute layer tolerates them is its question).
fn scan_nonfinite(tensors: &[ParsedTensor]) -> Result<(), KatakuriError> {
    for tensor in tensors {
        if tensor.dtype != Dtype::F32 {
            continue;
        }
        for chunk in tensor.data.chunks_exact(4) {
            let Ok(raw) = <[u8; 4]>::try_from(chunk) else {
                // `chunks_exact(4)` yields exactly four bytes; unreachable.
                continue;
            };
            let value = f32::from_le_bytes(raw);
            let which = if value.is_nan() {
                NonfiniteKind::Nan
            } else if value == f32::INFINITY {
                NonfiniteKind::PositiveInfinity
            } else if value == f32::NEG_INFINITY {
                NonfiniteKind::NegativeInfinity
            } else {
                continue;
            };
            return Err(KatakuriError::NonfiniteTensorValue {
                tensor: tensor.name.clone(),
                which,
            });
        }
    }
    Ok(())
}

/// Typed failures of the optimizer-state convention.
///
/// S5 failures (I/O, trust policy, S1 parse, `ia.*` metadata, declared
/// hashes, nonfinite values) arrive as [`Self::Checkpoint`] — the convention
/// adds no second vocabulary for them. The remaining variants are the
/// convention's own rules, each naming the tensor or metadata key at fault.
///
/// # Bounded diagnostics
///
/// **The invariant is provenance-free: no constructor produces an unbounded
/// message, whatever produced its input.**
///
/// - Untrusted text (metadata values, tensor names, a malformed declared
///   spec) is a [`bounded_excerpt`]: at most [`MAX_DIAGNOSTIC_EXCERPT_BYTES`]
///   input bytes, UTF-8-boundary truncated, control-escaped, true length
///   stamped when truncated.
/// - Shapes are rendered by [`shape_excerpt`]: at most
///   [`MAX_DIAGNOSTIC_SHAPE_DIMS`] dims, true rank stamped.
/// - Name lists (a missing-parameter list) are rendered by
///   [`name_list_excerpt`]: at most [`MAX_DIAGNOSTIC_NAME_LIST`] entries,
///   true remainder stamped.
///
/// The list cap deliberately assumes nothing about the spec's provenance:
/// [`MlpSpec`]'s fields are public, so a hand-built 10,000-layer spec (which
/// the strict JSON grammar would reject) is legal input to this API and
/// derives 20,002 parameter names. "Bounded by the spec grammar" holds only
/// for parse-validated specs, so it is not an argument this type may use.
#[derive(Debug, Error)]
pub enum OptimizerStateError {
    /// An S5 checkpoint failure: file/policy checks, S1 parse, `ia.*`
    /// metadata convention, declared hash mismatch, or a nonfinite value
    /// under `reject_nonfinite`.
    #[error(transparent)]
    Checkpoint(
        /// The underlying S5 error.
        #[from]
        KatakuriError,
    ),

    /// All-or-nothing violation: a parameter has one moment but not the
    /// other (or neither). Partial optimizer state is never accepted.
    #[error("parameter '{param}' is missing its {moment} tensor")]
    MissingMoment {
        /// The parameter whose moments are incomplete.
        param: String,
        /// Which moment is absent.
        moment: MomentKind,
    },

    /// A tensor outside the derived moment set (and, when parameters are
    /// carried, outside the expected parameter set). Nested or stray moment
    /// suffixes land here: a name ending in a moment suffix is never a
    /// parameter.
    #[error("unexpected tensor '{name}' in an optimizer-state checkpoint")]
    UnexpectedTensor {
        /// The unexpected tensor's name.
        name: String,
    },

    /// Some parameter tensors are present but not all. An artifact carries
    /// either the full moment set alone or the full parameter set alongside
    /// it — never a partial weight set.
    #[error("ambiguous optimizer-state checkpoint: {detail}")]
    PartialParameters {
        /// Which parameter tensors are missing.
        detail: String,
    },

    /// A parameter tensor carried alongside the moments does not match the
    /// spec's dtype/shape.
    #[error("parameter tensor '{tensor}': {detail}")]
    ParameterMismatch {
        /// The offending parameter tensor.
        tensor: String,
        /// What diverged.
        detail: String,
    },

    /// A moment's shape differs from its parameter's shape (no transposes at
    /// rest, no reshapes).
    ///
    /// `expected`/`actual` are bounded diagnostic renderings
    /// ([`shape_excerpt`]'s form: at most [`MAX_DIAGNOSTIC_SHAPE_DIMS`] dims,
    /// true rank stamped) rather than shape vectors — an untrusted rank is
    /// unbounded. Read the shape from the tensor itself
    /// ([`OptimizerState::tensor`]) when you need it programmatically.
    #[error("moment '{tensor}': wrong shape (expected {expected}, found {actual})")]
    MomentShapeMismatch {
        /// The offending moment tensor (bounded excerpt).
        tensor: String,
        /// The parameter's shape, bounded for diagnosis.
        expected: String,
        /// The shape found in the file, bounded for diagnosis.
        actual: String,
    },

    /// A moment's dtype differs from its parameter's dtype.
    #[error("moment '{tensor}': wrong dtype (expected {expected:?}, found {actual:?})")]
    MomentDtypeMismatch {
        /// The offending moment tensor.
        tensor: String,
        /// The parameter's dtype.
        expected: Dtype,
        /// The dtype found in the file.
        actual: Dtype,
    },

    /// A required convention metadata key is absent.
    #[error("required optimizer metadata '{key}' is absent")]
    MissingMetadata {
        /// The absent metadata key.
        key: String,
    },

    /// A convention metadata value does not parse (`optimizer_state_version`
    /// as a `u32`, `step` as a `u64`).
    #[error("optimizer metadata '{key}' is malformed: {detail}")]
    MalformedMetadata {
        /// The offending metadata key.
        key: String,
        /// What the value should have been.
        detail: String,
    },

    /// `optimizer_state_version` names a convention this crate does not
    /// implement. Forward compatibility is a new version, never a guess.
    #[error("unsupported optimizer_state_version {found} (this crate implements 1)")]
    UnsupportedStateVersion {
        /// The version found in metadata.
        found: u32,
    },

    /// `optimizer` names an optimizer this convention does not cover.
    #[error("unsupported optimizer '{found}' (only \"adamw\" is defined)")]
    UnsupportedOptimizer {
        /// The optimizer name found in metadata.
        found: String,
    },

    /// The artifact declares an `ia.spec` that is not the caller's spec: the
    /// state does not belong to this parameter checkpoint. Comparison is
    /// semantic (canonical form), so whitespace and key order never land
    /// here.
    #[error("optimizer state does not belong to this model: {detail}")]
    SpecMismatch {
        /// What diverged.
        detail: String,
    },

    /// The artifact declares an `ia.spec` that is not a valid spec at all.
    /// Distinct from [`Self::SpecMismatch`]: the declaration cannot be
    /// compared, so it is malformed metadata rather than a disagreement.
    #[error("optimizer metadata 'ia.spec' is not a valid spec: {detail}")]
    MalformedSpecMetadata {
        /// The parser's bounded description of the divergence.
        detail: String,
    },
}
