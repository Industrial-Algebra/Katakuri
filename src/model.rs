// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Architecture specs as typed data: strict BERT `config.json` parsing
//! ([`BertSpec`]), strict `ia-mlp` feedforward parsing ([`MlpSpec`]), and
//! weight-layout manifests validated against parsed safetensors
//! ([`crate::safetensors::SafeTensors`]).
//!
//! Specs are data, never code — no forward implementations live here,
//! and no numerics: weight *values* are never inspected, only layout
//! (names, dtypes, shapes, set completeness). The layer replaces
//! candle-transformers `models::bert` config parsing at Ijima's swap
//! boundary (see `docs/plans/` S3 port plan) and describes Amari's small
//! MLP ranker (S5, wishlist W1); candle stays behind Ijima's non-default
//! `embeddings-candle` feature throughout.
//!
//! The two families are separate specs sharing invariants (strict
//! `model_type` gating, parse-time bounds, `expected_tensors()` as the
//! compatibility oracle) — `BertSpec` is deliberately not generalized.

use crate::error::KatakuriError;
use crate::safetensors::Dtype;
use serde_json::{Map, Value, json};

/// Manifest name of the sole sanctioned non-parameter tensor (an I64
/// buffer candle derives as an arange when it is absent).
const POSITION_IDS_NAME: &str = "embeddings.position_ids";

/// Activation used by the intermediate dense layers. Strict subset of
/// HF's `hidden_act` strings — the three values candle's BERT can
/// compute. Unknown activations reject at parse time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenAct {
    /// `"gelu"` — exact (erf) GELU.
    Gelu,
    /// `"gelu_approximate"` — tanh-approximated GELU.
    GeluApproximate,
    /// `"relu"`.
    Relu,
}

/// The strict subset of a BERT `config.json` that inference consumes.
/// Unknown-but-inference-irrelevant training fields are tolerated and
/// ignored (documented below); everything this crate consumes is
/// validated at parse time. Construct via [`BertSpec::from_config_json`].
#[derive(Clone, Debug)]
pub struct BertSpec {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    type_vocab_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    pad_token_id: u32,
    hidden_act: HiddenAct,
}

impl BertSpec {
    /// Parses and validates a BERT `config.json`.
    ///
    /// Strict construction: `model_type` must be `"bert"` (required —
    /// absent rejects); `architectures` when present must contain
    /// `"BertModel"`; `position_embedding_type` when present must be
    /// `"absolute"`; `hidden_act` must be one of the three
    /// [`HiddenAct`] subset strings (others — e.g. `"gelu_new"` —
    /// reject with [`crate::error::KatakuriError::InvalidConfigJson`]).
    ///
    /// Numeric validation: every size field must be a positive integer
    /// that fits `u32` (id-space compatibility); `hidden_size` must be
    /// divisible by `num_attention_heads`; `layer_norm_eps` must be
    /// finite and `> 0.0`; `pad_token_id` must fit `u32`.
    ///
    /// Tolerated-and-ignored (inference-irrelevant training metadata):
    /// `_name_or_path`, `attention_probs_dropout_prob`,
    /// `gradient_checkpointing`, `hidden_dropout_prob`,
    /// `initializer_range`, `transformers_version`, `use_cache`, and
    /// any other unknown keys — recorded in the docs, never read.
    ///
    /// # Errors
    ///
    /// [`crate::error::KatakuriError::InvalidConfigJson`] on any
    /// malformed, missing, or out-of-subset value, with a `detail`
    /// naming the field and the problem.
    pub fn from_config_json(json: &str) -> Result<Self, KatakuriError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| invalid_config(format!("config.json is not valid JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| invalid_config("config.json root must be a JSON object".to_string()))?;

        match obj.get("model_type").and_then(Value::as_str) {
            Some("bert") => {}
            Some(other) => {
                return Err(invalid_config(format!(
                    "model_type '{other}' is not supported (only \"bert\")"
                )));
            }
            None => {
                return Err(invalid_config(
                    "model_type is required and must be the string \"bert\"".to_string(),
                ));
            }
        }

        if let Some(arch) = obj.get("architectures") {
            let arr = arch.as_array().ok_or_else(|| {
                invalid_config("architectures must be an array of strings".to_string())
            })?;
            if !arr.iter().any(|a| a.as_str() == Some("BertModel")) {
                return Err(invalid_config(
                    "architectures must contain \"BertModel\"".to_string(),
                ));
            }
        }

        if let Some(pos) = obj.get("position_embedding_type") {
            if pos.as_str() != Some("absolute") {
                return Err(invalid_config(format!(
                    "position_embedding_type {pos} is not supported (only \"absolute\")"
                )));
            }
        }

        let hidden_act = match obj.get("hidden_act").and_then(Value::as_str) {
            Some("gelu") => HiddenAct::Gelu,
            Some("gelu_approximate") => HiddenAct::GeluApproximate,
            Some("relu") => HiddenAct::Relu,
            Some(other) => {
                return Err(invalid_config(format!(
                    "hidden_act '{other}' is outside the supported subset \
                     (gelu, gelu_approximate, relu)"
                )));
            }
            None => {
                return Err(invalid_config(
                    "hidden_act is required and must be one of gelu, gelu_approximate, relu"
                        .to_string(),
                ));
            }
        };

        let hidden_size = size_field(obj, "hidden_size")?;
        let num_attention_heads = size_field(obj, "num_attention_heads")?;
        let num_hidden_layers = size_field(obj, "num_hidden_layers")?;
        // Bounded: `expected_tensors()` derives `5 + 16·L + 2` entries,
        // which overflows 32-bit `usize` near `u32::MAX` — a parse-valid
        // spec must never trap in a later method (review round 3). The
        // largest known BERT-family encoders use ≤ 96 layers; 1024 is
        // >10× headroom.
        if num_hidden_layers > 1024 {
            return Err(invalid_config(format!(
                "num_hidden_layers {num_hidden_layers} exceeds the supported \
                 maximum of 1024"
            )));
        }
        let intermediate_size = size_field(obj, "intermediate_size")?;
        let vocab_size = size_field(obj, "vocab_size")?;
        let type_vocab_size = size_field(obj, "type_vocab_size")?;
        let max_position_embeddings = size_field(obj, "max_position_embeddings")?;

        if hidden_size % num_attention_heads != 0 {
            return Err(invalid_config(format!(
                "hidden_size {hidden_size} is not divisible by num_attention_heads \
                 {num_attention_heads}"
            )));
        }

        let layer_norm_eps = obj
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                invalid_config("layer_norm_eps is required and must be a number".to_string())
            })?;
        if !layer_norm_eps.is_finite() || layer_norm_eps <= 0.0 {
            return Err(invalid_config(format!(
                "layer_norm_eps must be finite and > 0.0, got {layer_norm_eps}"
            )));
        }

        let pad_raw = obj
            .get("pad_token_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                invalid_config(
                    "pad_token_id is required and must be a non-negative integer".to_string(),
                )
            })?;
        let pad_token_id = u32::try_from(pad_raw)
            .map_err(|_| invalid_config(format!("pad_token_id {pad_raw} does not fit u32")))?;

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_hidden_layers,
            intermediate_size,
            vocab_size,
            type_vocab_size,
            max_position_embeddings,
            layer_norm_eps,
            pad_token_id,
            hidden_act,
        })
    }

    /// Validates a parsed weight set against this spec's layout
    /// manifest: every expected tensor present with the exact derived
    /// shape and dtype F32; no unexpected tensors (`position_ids` the
    /// sole sanctioned extra, then only as I64
    /// `[1, max_position_embeddings]`).
    ///
    /// Diagnostics-first: ALL problems are collected into one error
    /// `detail` (missing / unexpected / wrong-shape / wrong-dtype
    /// lines), not first-error-only.
    ///
    /// # Errors
    ///
    /// [`crate::error::KatakuriError::WeightLayoutMismatch`] when the
    /// set diverges, listing every divergence found.
    pub fn validate_weights(
        &self,
        weights: &crate::safetensors::SafeTensors<'_>,
    ) -> Result<(), KatakuriError> {
        let expected = self.expected_tensors();
        let expected_names: std::collections::BTreeSet<&str> =
            expected.iter().map(|(name, _)| name.as_str()).collect();
        let mut problems: Vec<String> = Vec::new();

        for (name, shape) in &expected {
            match weights.tensor(name) {
                Ok(t) => {
                    if t.dtype() != Dtype::F32 {
                        problems.push(format!(
                            "tensor '{name}': wrong dtype (expected F32, found {:?})",
                            t.dtype()
                        ));
                    }
                    if t.shape() != shape.as_slice() {
                        problems.push(format!(
                            "tensor '{name}': wrong shape (expected {}, found {})",
                            fmt_shape(shape),
                            fmt_shape(t.shape())
                        ));
                    }
                }
                Err(_) => problems.push(format!(
                    "missing tensor '{name}' (expected shape {})",
                    fmt_shape(shape)
                )),
            }
        }

        for name in weights.names() {
            if expected_names.contains(name) {
                continue;
            }
            if name == POSITION_IDS_NAME {
                let Ok(t) = weights.tensor(name) else {
                    continue;
                };
                if t.dtype() != Dtype::I64 {
                    problems.push(format!(
                        "tensor '{name}': wrong dtype (expected I64, found {:?})",
                        t.dtype()
                    ));
                }
                let expected_shape = [1usize, self.max_position_embeddings];
                if t.shape() != expected_shape {
                    problems.push(format!(
                        "tensor '{name}': wrong shape (expected {}, found {})",
                        fmt_shape(&expected_shape),
                        fmt_shape(t.shape())
                    ));
                }
            } else {
                problems.push(format!("unexpected tensor '{name}'"));
            }
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(KatakuriError::WeightLayoutMismatch {
                detail: problems.join("\n"),
            })
        }
    }

    /// Cross-checks the spec against a tokenizer: no token id the
    /// tokenizer can emit may index past the embedding table —
    /// `max_token_id() + 1 <= vocab_size`. Cardinality alone is not
    /// the check: sparse vocab ids (entry id 40000 in a 30522-entry
    /// table) preserve cardinality but overflow the table
    /// (review round 1). A tokenizer smaller than the table is legal
    /// (vocab-padded configs leave rows unused).
    ///
    /// # Errors
    ///
    /// [`crate::error::KatakuriError::InvalidConfigJson`] when the
    /// tokenizer can emit an id at or past `vocab_size` (detail
    /// carries the max emittable id and the table size).
    pub fn check_tokenizer_compatibility(
        &self,
        tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Result<(), KatakuriError> {
        let max_id = tokenizer.max_token_id();
        // `>=`, not `+ 1 >`: on 32-bit targets (wasm32) `max_id as usize
        // + 1` wraps at `u32::MAX` and admits the overflow case
        // (review round 2). `max_id >= vocab_size` is target-independent
        // and algebraically identical on 64-bit.
        if max_id as usize >= self.vocab_size {
            return Err(invalid_config(format!(
                "tokenizer can emit id {max_id}, past the embedding table \
                 (vocab_size {})",
                self.vocab_size
            )));
        }
        Ok(())
    }

    /// Hidden size `H`.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Number of encoder layers `L`.
    #[must_use]
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// Intermediate (feed-forward) size `I`.
    #[must_use]
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Token-type vocabulary size.
    #[must_use]
    pub fn type_vocab_size(&self) -> usize {
        self.type_vocab_size
    }

    /// Maximum position embeddings.
    #[must_use]
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// LayerNorm epsilon.
    #[must_use]
    pub fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }

    /// Padding token id.
    #[must_use]
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Activation for the intermediate dense layers.
    #[must_use]
    pub fn hidden_act(&self) -> HiddenAct {
        self.hidden_act
    }

    /// The expected weight tensor set as `(name, shape)` pairs —
    /// `5 + 16·L + 2` entries, layer indices `0..L`, shapes derived
    /// from the accessors. Excludes the optional `position_ids`
    /// buffer. Order: embeddings, then layers in index order (q/k/v →
    /// attn-out → intermediate → output), then pooler.
    #[must_use]
    pub fn expected_tensors(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden_size;
        let i = self.intermediate_size;
        let mut out = Vec::with_capacity(7 + 16 * self.num_hidden_layers);
        out.push((
            "embeddings.word_embeddings.weight".to_string(),
            vec![self.vocab_size, h],
        ));
        out.push((
            "embeddings.position_embeddings.weight".to_string(),
            vec![self.max_position_embeddings, h],
        ));
        out.push((
            "embeddings.token_type_embeddings.weight".to_string(),
            vec![self.type_vocab_size, h],
        ));
        out.push(("embeddings.LayerNorm.weight".to_string(), vec![h]));
        out.push(("embeddings.LayerNorm.bias".to_string(), vec![h]));
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("encoder.layer.{layer}");
            out.push((format!("{prefix}.attention.self.query.weight"), vec![h, h]));
            out.push((format!("{prefix}.attention.self.query.bias"), vec![h]));
            out.push((format!("{prefix}.attention.self.key.weight"), vec![h, h]));
            out.push((format!("{prefix}.attention.self.key.bias"), vec![h]));
            out.push((format!("{prefix}.attention.self.value.weight"), vec![h, h]));
            out.push((format!("{prefix}.attention.self.value.bias"), vec![h]));
            out.push((
                format!("{prefix}.attention.output.dense.weight"),
                vec![h, h],
            ));
            out.push((format!("{prefix}.attention.output.dense.bias"), vec![h]));
            out.push((
                format!("{prefix}.attention.output.LayerNorm.weight"),
                vec![h],
            ));
            out.push((format!("{prefix}.attention.output.LayerNorm.bias"), vec![h]));
            out.push((format!("{prefix}.intermediate.dense.weight"), vec![i, h]));
            out.push((format!("{prefix}.intermediate.dense.bias"), vec![i]));
            out.push((format!("{prefix}.output.dense.weight"), vec![h, i]));
            out.push((format!("{prefix}.output.dense.bias"), vec![h]));
            out.push((format!("{prefix}.output.LayerNorm.weight"), vec![h]));
            out.push((format!("{prefix}.output.LayerNorm.bias"), vec![h]));
        }
        out.push(("pooler.dense.weight".to_string(), vec![h, h]));
        out.push(("pooler.dense.bias".to_string(), vec![h]));
        out
    }
}

fn invalid_config(detail: String) -> KatakuriError {
    KatakuriError::InvalidConfigJson { detail }
}

fn invalid_mlp(detail: String) -> KatakuriError {
    KatakuriError::MlpSpecParse { detail }
}

// ── MLP (ia-mlp) feedforward specs (S5) ───────────────────────────

/// Largest accepted `input_width` for an [`MlpSpec`].
///
/// Derived shapes stay far below any 32-bit wrap point: the widest single
/// layer is `16384 × 16384 = 268_435_456` elements, which fits a 32-bit
/// `usize` (the wasm32 review lesson, S3/S5).
const MLP_MAX_INPUT_WIDTH: u64 = 4096;

/// Largest accepted hidden-layer count. `expected_tensors()` derives
/// `2 + 2·hidden.len()` entries — no multiplication by an unbounded input.
const MLP_MAX_HIDDEN_LAYERS: usize = 8;

/// Largest accepted output width for any [`MlpLayerSpec`].
const MLP_MAX_LAYER_WIDTH: u64 = 16384;

/// Activation with pinned, backend-independent semantics.
///
/// Approximations are NAMED as approximations — never inherited from a
/// backend. The candle lesson (`GeluApproximate`): activation semantics are
/// decided, named, and versioned here, while the arithmetic belongs to the
/// consumer (Katakuri never computes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activation {
    /// Rectified linear unit: `max(0, x)`.
    Relu,
    /// Hyperbolic tangent.
    Tanh,
    /// Exact GELU: `0.5x(1+erf(x/√2))`. Erf is evaluated by the CONSUMER;
    /// this enum names the semantics, it does not compute.
    GeluErf,
    /// Tanh-approximation GELU (candle's `GeluApproximate` semantics — the
    /// exact function Ijima pinned; the pin becomes a named choice).
    GeluTanh,
}

impl Activation {
    /// The canonical spec string for this activation: `"relu"`, `"tanh"`,
    /// `"gelu_erf"`, or `"gelu_tanh"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Tanh => "tanh",
            Self::GeluErf => "gelu_erf",
            Self::GeluTanh => "gelu_tanh",
        }
    }
}

/// One feedforward layer's output width.
///
/// Weight layout is implied by position in the stack: a layer with width `w`
/// following a layer (or input) of width `i` carries `weight: [w, i]`
/// row-major and `bias: [w]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlpLayerSpec {
    /// Output width of this layer.
    pub width: u32,
}

/// Feedforward (MLP) architecture spec — strict family
/// `model_type: "ia-mlp"`.
///
/// A layer list, activations, and shapes: an architecture description, not a
/// hub model name. Construct via [`MlpSpec::from_json`]; `expected_tensors()`
/// is the compatibility oracle (canonical parameter order included).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlpSpec {
    /// Input feature width.
    pub input_width: u32,
    /// Hidden layers, in order. Bounded: `hidden.len() <= 8`.
    pub hidden: Vec<MlpLayerSpec>,
    /// Final linear to a scalar score: output width is always 1.
    pub head: MlpLayerSpec,
    /// Activation applied after every hidden layer.
    pub hidden_activation: Activation,
}

impl MlpSpec {
    /// Parses and validates an `ia-mlp` spec JSON document.
    ///
    /// # Grammar (strict)
    ///
    /// ```json
    /// {
    ///   "model_type": "ia-mlp",
    ///   "schema_version": 1,
    ///   "input_width": 64,
    ///   "hidden": [{"width": 256}],
    ///   "head": {"width": 1},
    ///   "hidden_activation": "gelu_erf"
    /// }
    /// ```
    ///
    /// - `model_type` is **required** and must be exactly `"ia-mlp"`; any
    ///   other value rejects with an error naming both supported families
    ///   (`"bert"` / `"ia-mlp"`).
    /// - `schema_version` is **required** and must be exactly `1` — a
    ///   forward-compatibility gate, not a promise.
    /// - `input_width` is required, `1 ..= 4096`.
    /// - `hidden` is required, length `0 ..= 8`; each entry is an object with
    ///   a required `width`, `1 ..= 16384`.
    /// - `head` is required, an object with a required `width`; the head is a
    ///   scalar score, so `width` must be exactly `1`.
    /// - `hidden_activation` is required and must be one of `"relu"`,
    ///   `"tanh"`, `"gelu_erf"`, `"gelu_tanh"`.
    ///
    /// Unrecognized keys are tolerated and ignored (the S3 construction
    /// discipline: the consumed surface is strict, training metadata is not
    /// this crate's business). The checkpoint metadata convention is where
    /// unknown keys reject — §5.5 of the S5 plan.
    ///
    /// Every bound is checked with plain comparisons on the parsed integers:
    /// no `+1`, no subtraction, no multiplication at parse time, so the
    /// checks cannot wrap on a 32-bit target (wasm32 lesson, re-applied).
    ///
    /// # Errors
    ///
    /// [`KatakuriError::MlpSpecParse`] on malformed JSON, a missing or
    /// out-of-subset field, or a value outside the bounds above; `detail`
    /// names the field and the problem.
    pub fn from_json(json: &str) -> Result<Self, KatakuriError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| invalid_mlp(format!("spec is not valid JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| invalid_mlp("spec root must be a JSON object".to_string()))?;

        match obj.get("model_type").and_then(Value::as_str) {
            Some("ia-mlp") => {}
            Some(other) => {
                return Err(invalid_mlp(format!(
                    "model_type '{other}' is not supported (only \"bert\" or \"ia-mlp\")"
                )));
            }
            None => {
                return Err(invalid_mlp(
                    "model_type is required and must be the string \"ia-mlp\" (or \"bert\" \
                     for the BERT family)"
                        .to_string(),
                ));
            }
        }

        match obj.get("schema_version").and_then(Value::as_u64) {
            Some(1) => {}
            Some(other) => {
                return Err(invalid_mlp(format!(
                    "schema_version {other} is not supported (only 1)"
                )));
            }
            None => {
                return Err(invalid_mlp(
                    "schema_version is required and must be the integer 1".to_string(),
                ));
            }
        }

        let input_width = mlp_width(obj.get("input_width"), "input_width", MLP_MAX_INPUT_WIDTH)?;

        let hidden_raw = obj
            .get("hidden")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_mlp("hidden is required and must be an array".to_string()))?;
        if hidden_raw.len() > MLP_MAX_HIDDEN_LAYERS {
            return Err(invalid_mlp(format!(
                "hidden has {} layers, above the supported maximum of {MLP_MAX_HIDDEN_LAYERS}",
                hidden_raw.len()
            )));
        }
        let mut hidden = Vec::with_capacity(hidden_raw.len());
        for (index, entry) in hidden_raw.iter().enumerate() {
            let layer = entry
                .as_object()
                .ok_or_else(|| invalid_mlp(format!("hidden[{index}] must be an object")))?;
            let field = format!("hidden[{index}].width");
            hidden.push(MlpLayerSpec {
                width: mlp_width(layer.get("width"), &field, MLP_MAX_LAYER_WIDTH)?,
            });
        }

        let head_obj = obj
            .get("head")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid_mlp("head is required and must be an object".to_string()))?;
        let head_width = mlp_width(head_obj.get("width"), "head.width", MLP_MAX_LAYER_WIDTH)?;
        if head_width != 1 {
            return Err(invalid_mlp(format!(
                "head.width must be exactly 1 (the head produces a scalar score), got {head_width}"
            )));
        }

        let hidden_activation = match obj.get("hidden_activation").and_then(Value::as_str) {
            Some("relu") => Activation::Relu,
            Some("tanh") => Activation::Tanh,
            Some("gelu_erf") => Activation::GeluErf,
            Some("gelu_tanh") => Activation::GeluTanh,
            Some(other) => {
                return Err(invalid_mlp(format!(
                    "hidden_activation '{other}' is outside the supported set \
                     (relu, tanh, gelu_erf, gelu_tanh)"
                )));
            }
            None => {
                return Err(invalid_mlp(
                    "hidden_activation is required and must be one of relu, tanh, gelu_erf, \
                     gelu_tanh"
                        .to_string(),
                ));
            }
        };

        Ok(Self {
            input_width,
            hidden,
            head: MlpLayerSpec { width: head_width },
            hidden_activation,
        })
    }

    /// The expected weight tensor set as `(name, shape)` pairs, in the
    /// canonical parameter order: depth-first, layer order —
    /// `hidden.0.weight`, `hidden.0.bias`, …, `head.weight`, `head.bias`.
    ///
    /// Shapes: each `weight` is `[out, in]` (row-major) and each `bias` is
    /// `[out]`; the head's output width is 1. This order IS the canonical
    /// ordering validated checkpoint round-trips rely on (S5 §5.1).
    ///
    /// Bounded arithmetic: with `width <= 16384`, the largest derived
    /// dimension is `16384 × 16384 = 268_435_456`, which fits a 32-bit
    /// `usize` — every parse-valid spec can call this without trapping.
    #[must_use]
    pub fn expected_tensors(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::with_capacity(2 + 2 * self.hidden.len());
        let mut in_width = self.input_width;
        for (index, layer) in self.hidden.iter().enumerate() {
            out.push((
                format!("hidden.{index}.weight"),
                vec![layer.width as usize, in_width as usize],
            ));
            out.push((format!("hidden.{index}.bias"), vec![layer.width as usize]));
            in_width = layer.width;
        }
        out.push((
            "head.weight".to_string(),
            vec![self.head.width as usize, in_width as usize],
        ));
        out.push(("head.bias".to_string(), vec![self.head.width as usize]));
        out
    }

    /// The canonical JSON serialization of this spec: a deterministic,
    /// key-sorted document with exactly the grammar fields.
    ///
    /// The bytes of this string are what `ia.schema-hash` digests (S5 §5.5),
    /// so two semantically identical specs hash identically regardless of the
    /// input document's whitespace or key order.
    #[must_use]
    pub fn canonical_json(&self) -> String {
        let hidden: Vec<Value> = self
            .hidden
            .iter()
            .map(|layer| json!({ "width": layer.width }))
            .collect();
        let value = json!({
            "model_type": "ia-mlp",
            "schema_version": 1,
            "input_width": self.input_width,
            "hidden": hidden,
            "head": json!({ "width": self.head.width }),
            "hidden_activation": self.hidden_activation.as_str(),
        });
        // `serde_json`'s default map is a `BTreeMap`, so key order is sorted
        // and the serialization is deterministic.
        value.to_string()
    }
}

/// Parse one required MLP width field: a non-negative JSON integer, `>= 1`
/// and `<= max`.
///
/// Comparisons only — no `+1`, no subtraction, no multiplication — so the
/// bounds cannot wrap on a 32-bit target.
fn mlp_width(raw: Option<&Value>, field: &str, max: u64) -> Result<u32, KatakuriError> {
    let value = raw.and_then(Value::as_u64).ok_or_else(|| {
        invalid_mlp(format!(
            "{field} is required and must be a non-negative integer"
        ))
    })?;
    if value == 0 || value > max {
        return Err(invalid_mlp(format!(
            "{field} must be in 1..={max}, got {value}"
        )));
    }
    u32::try_from(value).map_err(|_| invalid_mlp(format!("{field} {value} does not fit u32")))
}

/// Parse a positive size field that must fit `u32` (id-space
/// compatibility), then widen to `usize`.
fn size_field(obj: &Map<String, Value>, field: &str) -> Result<usize, KatakuriError> {
    let raw = obj.get(field).and_then(Value::as_u64).ok_or_else(|| {
        invalid_config(format!(
            "{field} is required and must be a non-negative integer"
        ))
    })?;
    if raw == 0 {
        return Err(invalid_config(format!("{field} must be positive, got 0")));
    }
    let bounded = u32::try_from(raw)
        .map_err(|_| invalid_config(format!("{field} {raw} does not fit u32")))?;
    Ok(bounded as usize)
}

fn fmt_shape(shape: &[usize]) -> String {
    let inner: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafeTensors;
    use crate::tokenizer::Tokenizer;
    use serde_json::json;

    fn tiny() -> BertSpec {
        BertSpec::from_config_json(include_str!("../tests/fixtures/tiny-config.json"))
            .expect("tiny config parses")
    }

    fn tiny_json() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/tiny-config.json"))
            .expect("tiny config is valid JSON")
    }

    fn parse_config(v: &Value) -> Result<BertSpec, KatakuriError> {
        BertSpec::from_config_json(&v.to_string())
    }

    fn rejects(v: &Value) -> bool {
        matches!(
            parse_config(v),
            Err(KatakuriError::InvalidConfigJson { .. })
        )
    }

    /// Build a syntactically valid safetensors file from
    /// `(name, dtype, shape)` entries with a zero payload. Layout is
    /// all `validate_weights` reads, so zero bytes suffice.
    fn build_file(entries: &[(String, String, Vec<usize>)]) -> Vec<u8> {
        let mut header = Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, dtype, shape) in entries {
            let dt = Dtype::parse(dtype).expect("known dtype");
            let count: usize = shape.iter().product();
            let len = count.checked_mul(dt.byte_size()).expect("bounded fixture");
            let start = data.len();
            data.resize(start + len, 0);
            header.insert(
                name.clone(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, start + len]}),
            );
        }
        let header_bytes =
            serde_json::to_vec(&Value::Object(header)).expect("serialize fixture header");
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    fn all_f32_entries(spec: &BertSpec) -> Vec<(String, String, Vec<usize>)> {
        spec.expected_tensors()
            .into_iter()
            .map(|(name, shape)| (name, "F32".to_string(), shape))
            .collect()
    }

    #[test]
    fn parses_tiny_and_real_shapes() {
        let spec = tiny();
        assert_eq!(spec.hidden_size(), 8);
        assert_eq!(spec.num_attention_heads(), 4);
        assert_eq!(spec.num_hidden_layers(), 2);
        assert_eq!(spec.intermediate_size(), 16);
        assert_eq!(spec.vocab_size(), 32);
        assert_eq!(spec.type_vocab_size(), 2);
        assert_eq!(spec.max_position_embeddings(), 16);
        assert_eq!(spec.layer_norm_eps(), 1e-12);
        assert_eq!(spec.pad_token_id(), 0);
        assert_eq!(spec.hidden_act(), HiddenAct::Gelu);

        let tensors = spec.expected_tensors();
        assert_eq!(tensors.len(), 39);
        let find = |name: &str| -> Vec<usize> {
            tensors
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, s)| s.clone())
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        assert_eq!(find("embeddings.word_embeddings.weight"), vec![32, 8]);
        assert_eq!(
            find("encoder.layer.1.intermediate.dense.weight"),
            vec![16, 8]
        );
        assert_eq!(find("pooler.dense.weight"), vec![8, 8]);
    }

    #[test]
    fn parses_real_minilm_config() {
        let spec = BertSpec::from_config_json(include_str!("../tests/fixtures/minilm-config.json"))
            .expect("MiniLM config parses");
        assert_eq!(spec.hidden_size(), 384);
        assert_eq!(spec.num_hidden_layers(), 6);
        assert_eq!(spec.num_attention_heads(), 12);
        assert_eq!(spec.intermediate_size(), 1536);
        assert_eq!(spec.vocab_size(), 30522);
        assert_eq!(spec.layer_norm_eps(), 1e-12);
        assert_eq!(spec.hidden_act(), HiddenAct::Gelu);
        assert_eq!(spec.expected_tensors().len(), 103);
    }

    #[test]
    fn rejects_non_bert_model_type() {
        let mut gpt2 = tiny_json();
        gpt2["model_type"] = json!("gpt2");
        assert!(rejects(&gpt2));

        let mut absent = tiny_json();
        absent.as_object_mut().unwrap().remove("model_type");
        assert!(rejects(&absent));
    }

    #[test]
    fn rejects_bad_architectures_and_position_type() {
        let mut bad_arch = tiny_json();
        bad_arch["architectures"] = json!(["AutoModel"]);
        assert!(rejects(&bad_arch));

        let mut bad_pos = tiny_json();
        bad_pos["position_embedding_type"] = json!("relative_key");
        assert!(rejects(&bad_pos));

        let mut absent = tiny_json();
        let obj = absent.as_object_mut().unwrap();
        obj.remove("architectures");
        obj.remove("position_embedding_type");
        assert!(parse_config(&absent).is_ok());
    }

    #[test]
    fn rejects_unknown_hidden_act() {
        for (act, expected) in [
            ("gelu", HiddenAct::Gelu),
            ("gelu_approximate", HiddenAct::GeluApproximate),
            ("relu", HiddenAct::Relu),
        ] {
            let mut v = tiny_json();
            v["hidden_act"] = json!(act);
            assert_eq!(
                parse_config(&v)
                    .expect("subset activation parses")
                    .hidden_act(),
                expected,
                "{act}"
            );
        }

        let mut unknown = tiny_json();
        unknown["hidden_act"] = json!("gelu_new");
        assert!(rejects(&unknown));
    }

    #[test]
    fn rejects_nonpositive_and_indivisible_sizes() {
        let mut zero_hidden = tiny_json();
        zero_hidden["hidden_size"] = json!(0);
        assert!(rejects(&zero_hidden));

        let mut negative_vocab = tiny_json();
        negative_vocab["vocab_size"] = json!(-1);
        assert!(rejects(&negative_vocab));

        let mut indivisible = tiny_json();
        indivisible["hidden_size"] = json!(9);
        assert!(rejects(&indivisible));

        let mut zero_eps = tiny_json();
        zero_eps["layer_norm_eps"] = json!(0.0);
        assert!(rejects(&zero_eps));

        // serde_json rejects the bare `NaN` token before the finite
        // guard is reached; the config still rejects either way.
        let nan_config = include_str!("../tests/fixtures/tiny-config.json").replace("1e-12", "NaN");
        assert!(matches!(
            BertSpec::from_config_json(&nan_config),
            Err(KatakuriError::InvalidConfigJson { .. })
        ));
    }

    #[test]
    fn validates_tiny_weights_roundtrip() {
        let bytes = include_bytes!("../tests/fixtures/tiny-model.safetensors");
        let st = SafeTensors::parse(bytes).expect("parse tiny model");
        tiny()
            .validate_weights(&st)
            .expect("tiny weights match spec");
        assert_eq!(st.tensor_count(), 40);
    }

    #[test]
    fn weight_layout_collects_all_divergences() {
        let mut widened = tiny_json();
        widened["hidden_size"] = json!(16);
        let spec = parse_config(&widened).expect("H=16, heads=4 parses");

        let bytes = include_bytes!("../tests/fixtures/tiny-model.safetensors");
        let st = SafeTensors::parse(bytes).expect("parse tiny model");
        match spec.validate_weights(&st) {
            Err(KatakuriError::WeightLayoutMismatch { detail }) => {
                assert!(
                    detail.lines().count() >= 20,
                    "diagnostics-first collects many lines: {detail}"
                );
                assert!(detail.contains("wrong shape"), "{detail}");
                assert!(detail.contains("embeddings.LayerNorm.weight"), "{detail}");
            }
            other => panic!("expected WeightLayoutMismatch, got {other:?}"),
        }
    }

    #[test]
    fn weight_layout_flags_missing_unexpected_and_dtype() {
        let bytes = include_bytes!("../tests/fixtures/tiny-model.safetensors");
        let st = SafeTensors::parse(bytes).expect("parse tiny model");

        // (a) More expectations than the file satisfies: layer 2 missing.
        let mut three = tiny_json();
        three["num_hidden_layers"] = json!(3);
        let spec3 = parse_config(&three).expect("L=3 parses");
        match spec3.validate_weights(&st) {
            Err(KatakuriError::WeightLayoutMismatch { detail }) => {
                assert!(detail.contains("encoder.layer.2."), "{detail}");
                assert!(detail.contains("missing"), "{detail}");
                assert_eq!(detail.lines().count(), 16, "{detail}");
            }
            other => panic!("expected WeightLayoutMismatch, got {other:?}"),
        }

        // (b) Fewer expectations than the file carries: layer 1 unexpected.
        let mut one = tiny_json();
        one["num_hidden_layers"] = json!(1);
        let spec1 = parse_config(&one).expect("L=1 parses");
        match spec1.validate_weights(&st) {
            Err(KatakuriError::WeightLayoutMismatch { detail }) => {
                assert!(detail.contains("encoder.layer.1."), "{detail}");
                assert!(detail.contains("unexpected"), "{detail}");
                assert_eq!(detail.lines().count(), 16, "{detail}");
            }
            other => panic!("expected WeightLayoutMismatch, got {other:?}"),
        }

        // (c) One F16 tensor: dtype divergence.
        let entries: Vec<(String, String, Vec<usize>)> = tiny()
            .expected_tensors()
            .into_iter()
            .map(|(name, shape)| {
                let dtype = if name == "embeddings.word_embeddings.weight" {
                    "F16"
                } else {
                    "F32"
                };
                (name, dtype.to_string(), shape)
            })
            .collect();
        let file = build_file(&entries);
        let st = SafeTensors::parse(&file).expect("hand-built file parses");
        match tiny().validate_weights(&st) {
            Err(KatakuriError::WeightLayoutMismatch { detail }) => {
                assert!(
                    detail.contains("embeddings.word_embeddings.weight"),
                    "{detail}"
                );
                assert!(detail.contains("wrong dtype"), "{detail}");
            }
            other => panic!("expected WeightLayoutMismatch, got {other:?}"),
        }
    }

    #[test]
    fn position_ids_optional_but_typed() {
        let bytes = include_bytes!("../tests/fixtures/tiny-model.safetensors");
        let st = SafeTensors::parse(bytes).expect("parse tiny model");
        assert_eq!(
            st.tensor("embeddings.position_ids")
                .expect("position_ids present")
                .dtype(),
            Dtype::I64
        );
        tiny()
            .validate_weights(&st)
            .expect("I64 position_ids is sanctioned");

        // All-F32, no position_ids: the buffer is optional.
        let entries = all_f32_entries(&tiny());
        assert!(!entries.iter().any(|(n, _, _)| n == POSITION_IDS_NAME));
        let file = build_file(&entries);
        let st = SafeTensors::parse(&file).expect("hand-built file parses");
        tiny()
            .validate_weights(&st)
            .expect("position_ids is optional");

        // position_ids in the wrong dtype is a typed divergence.
        let mut wrong_dtype = all_f32_entries(&tiny());
        wrong_dtype.push((
            POSITION_IDS_NAME.to_string(),
            "F32".to_string(),
            vec![1, 16],
        ));
        let file = build_file(&wrong_dtype);
        let st = SafeTensors::parse(&file).expect("hand-built file parses");
        match tiny().validate_weights(&st) {
            Err(KatakuriError::WeightLayoutMismatch { detail }) => {
                assert!(detail.contains("embeddings.position_ids"), "{detail}");
                assert!(detail.contains("wrong dtype"), "{detail}");
            }
            other => panic!("expected WeightLayoutMismatch, got {other:?}"),
        }
    }

    /// A valid BERT-class `tokenizer.json` with a tiny 9-entry vocab
    /// (ids 0..=8), for the tokenizer-compatibility tests. The vocab
    /// object lives at `["model"]["vocab"]`; mutate it per test.
    fn small_tokenizer_json() -> Value {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "aff", "hello",
        ];
        let mut vocab_obj = Map::new();
        for (i, t) in vocab.iter().enumerate() {
            vocab_obj.insert((*t).to_string(), Value::from(i as u32));
        }
        let special = |content: &str, id: u32| {
            json!({"id": id, "content": content, "single_word": false,
                   "lstrip": false, "rstrip": false, "normalized": false,
                   "special": true})
        };
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                special("[PAD]", 0), special("[UNK]", 1),
                special("[CLS]", 2), special("[SEP]", 3),
            ],
            "normalizer": {"type": "BertNormalizer", "clean_text": true,
                "handle_chinese_chars": true, "strip_accents": null,
                "lowercase": true},
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": {"type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                ],
                "pair": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
                ]},
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
            "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": vocab_obj},
        })
    }

    #[test]
    fn tokenizer_compatibility() {
        let minilm =
            BertSpec::from_config_json(include_str!("../tests/fixtures/minilm-config.json"))
                .expect("MiniLM config parses");
        let tk =
            Tokenizer::from_bert_vocab_txt(include_bytes!("../tests/fixtures/minilm-vocab.txt"))
                .expect("MiniLM vocab parses");
        assert_eq!(tk.vocab_size(), 30522);
        assert_eq!(tk.max_token_id(), 30521);
        minilm
            .check_tokenizer_compatibility(&tk)
            .expect("ids fit the table");

        // Tiny table: the MiniLM tokenizer overflows it.
        match tiny().check_tokenizer_compatibility(&tk) {
            Err(KatakuriError::InvalidConfigJson { detail }) => {
                assert!(detail.contains("30521"), "{detail}");
                assert!(detail.contains("32"), "{detail}");
            }
            other => panic!("expected InvalidConfigJson, got {other:?}"),
        }

        // Smaller-than-table tokenizer is legal (vocab-padded rows).
        let mut small_json = small_tokenizer_json();
        small_json["model"]["vocab"] = json!({
            "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
            "un": 4, "##aff": 5, "##able": 6, "aff": 7, "hello": 8
        });
        let small = Tokenizer::from_tokenizer_json(small_json.to_string().as_bytes())
            .expect("small tokenizer parses");
        tiny()
            .check_tokenizer_compatibility(&small)
            .expect("ids fit the table");
    }

    #[test]
    fn sparse_tokenizer_id_rejects() {
        // Cardinality preserved (9 entries), one id moved past the
        // table: capacity, not count, is the compatibility invariant
        // (review round 1).
        let mut sparse = small_tokenizer_json();
        sparse["model"]["vocab"]["hello"] = json!(40);
        let tk = Tokenizer::from_tokenizer_json(sparse.to_string().as_bytes())
            .expect("sparse tokenizer parses");
        match tiny().check_tokenizer_compatibility(&tk) {
            Err(KatakuriError::InvalidConfigJson { detail }) => {
                assert!(detail.contains("40"), "{detail}");
                assert!(detail.contains("32"), "{detail}");
            }
            other => panic!("expected InvalidConfigJson, got {other:?}"),
        }
    }

    #[test]
    fn u32_max_token_id_rejects_without_overflow() {
        // `u32::MAX` is the wrap point that made `+ 1` unsound on
        // 32-bit targets (review round 2); the `>=` form rejects it
        // on every target.
        let mut extreme = small_tokenizer_json();
        extreme["model"]["vocab"]["hello"] = json!(u32::MAX);
        let tk = Tokenizer::from_tokenizer_json(extreme.to_string().as_bytes())
            .expect("extreme tokenizer parses");
        assert!(matches!(
            tiny().check_tokenizer_compatibility(&tk),
            Err(KatakuriError::InvalidConfigJson { .. })
        ));
    }

    #[test]
    fn layer_count_bounded_at_parse_time() {
        // `5 + 16·L + 2` overflows 32-bit usize near `u32::MAX` —
        // a parse-valid spec must never trap in `expected_tensors()`
        // (review round 3). Boundary: 1024 parses, 1025 rejects.
        let mut huge = tiny_json();
        huge["num_hidden_layers"] = json!(1025);
        assert!(rejects(&huge));

        let mut max = tiny_json();
        max["num_hidden_layers"] = json!(1024);
        let spec = parse_config(&max).expect("1024 layers parses");
        assert_eq!(spec.expected_tensors().len(), 5 + 16 * 1024 + 2);
    }

    // ── MLP (ia-mlp) specs (S5) ───────────────────────────────

    fn mlp_json() -> Value {
        json!({
            "model_type": "ia-mlp",
            "schema_version": 1,
            "input_width": 2,
            "hidden": [{"width": 3}, {"width": 4}],
            "head": {"width": 1},
            "hidden_activation": "relu",
        })
    }

    fn parse_mlp(v: &Value) -> Result<MlpSpec, KatakuriError> {
        MlpSpec::from_json(&v.to_string())
    }

    fn mlp_rejects(v: &Value) -> bool {
        matches!(parse_mlp(v), Err(KatakuriError::MlpSpecParse { .. }))
    }

    #[test]
    fn mlp_parses_and_derives_canonical_order() {
        let spec = parse_mlp(&mlp_json()).expect("legal ia-mlp spec parses");
        assert_eq!(spec.input_width, 2);
        assert_eq!(spec.hidden.len(), 2);
        assert_eq!(spec.head.width, 1);
        assert_eq!(spec.hidden_activation, Activation::Relu);
        assert_eq!(spec.expected_tensors().len(), 6);

        // Depth-first, layer order: hidden.N.weight, hidden.N.bias, …,
        // head.weight, head.bias.
        let names: Vec<String> = spec
            .expected_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            names,
            vec![
                "hidden.0.weight",
                "hidden.0.bias",
                "hidden.1.weight",
                "hidden.1.bias",
                "head.weight",
                "head.bias",
            ]
        );
        let shapes: Vec<Vec<usize>> = spec
            .expected_tensors()
            .into_iter()
            .map(|(_, shape)| shape)
            .collect();
        assert_eq!(
            shapes,
            vec![
                vec![3, 2],
                vec![3],
                vec![4, 3],
                vec![4],
                vec![1, 4],
                vec![1],
            ]
        );
    }

    #[test]
    fn mlp_accepts_empty_hidden_stack() {
        let mut v = mlp_json();
        v["hidden"] = json!([]);
        let spec = parse_mlp(&v).expect("zero hidden layers is legal");
        let tensors = spec.expected_tensors();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].0, "head.weight");
        assert_eq!(tensors[0].1, vec![1, 2]);
        assert_eq!(tensors[1].0, "head.bias");
        assert_eq!(tensors[1].1, vec![1]);
    }

    #[test]
    fn mlp_model_type_gate_names_both_families() {
        let mut bert = mlp_json();
        bert["model_type"] = json!("bert");
        match parse_mlp(&bert) {
            Err(KatakuriError::MlpSpecParse { detail }) => {
                assert!(detail.contains("\"bert\""), "{detail}");
                assert!(detail.contains("\"ia-mlp\""), "{detail}");
            }
            other => panic!("expected MlpSpecParse, got {other:?}"),
        }

        let mut unknown = mlp_json();
        unknown["model_type"] = json!("gpt2");
        assert!(mlp_rejects(&unknown));

        let mut absent = mlp_json();
        absent.as_object_mut().unwrap().remove("model_type");
        match parse_mlp(&absent) {
            Err(KatakuriError::MlpSpecParse { detail }) => {
                assert!(detail.contains("\"bert\""), "{detail}");
                assert!(detail.contains("\"ia-mlp\""), "{detail}");
            }
            other => panic!("expected MlpSpecParse, got {other:?}"),
        }
    }

    #[test]
    fn mlp_schema_version_is_required_and_pinned() {
        for bad in [json!(null), json!(2), json!(0), json!("1")] {
            let mut v = mlp_json();
            v["schema_version"] = bad.clone();
            assert!(mlp_rejects(&v), "schema_version {bad} must reject");
        }

        let mut absent = mlp_json();
        absent.as_object_mut().unwrap().remove("schema_version");
        assert!(mlp_rejects(&absent));
    }

    #[test]
    fn mlp_rejects_unknown_activation_and_missing_fields() {
        for act in ["relu", "tanh", "gelu_erf", "gelu_tanh"] {
            let mut v = mlp_json();
            v["hidden_activation"] = json!(act);
            parse_mlp(&v).unwrap_or_else(|e| panic!("{act} should parse: {e}"));
        }

        let mut unknown = mlp_json();
        unknown["hidden_activation"] = json!("gelu_new");
        assert!(mlp_rejects(&unknown));

        for field in ["hidden_activation", "input_width", "hidden", "head"] {
            let mut missing = mlp_json();
            missing.as_object_mut().unwrap().remove(field);
            assert!(mlp_rejects(&missing), "missing {field} must reject");
        }
    }

    #[test]
    fn mlp_bounds_reject_at_the_edges() {
        // input_width: 1..=4096
        for ok in [1, 4096] {
            let mut v = mlp_json();
            v["input_width"] = json!(ok);
            assert!(parse_mlp(&v).is_ok(), "input_width {ok} should parse");
        }
        for bad in [0, 4097, 1_u64 << 40] {
            let mut v = mlp_json();
            v["input_width"] = json!(bad);
            assert!(mlp_rejects(&v), "input_width {bad} must reject");
        }

        // layer width: 1..=16384
        for ok in [1, 16384] {
            let mut v = mlp_json();
            v["hidden"] = json!([{"width": ok}]);
            assert!(parse_mlp(&v).is_ok(), "width {ok} should parse");
        }
        for bad in [0, 16385] {
            let mut v = mlp_json();
            v["hidden"] = json!([{"width": bad}]);
            assert!(mlp_rejects(&v), "width {bad} must reject");
        }

        // hidden.len(): 0..=8
        let eight: Vec<Value> = (0..8).map(|_| json!({"width": 4})).collect();
        let mut ok = mlp_json();
        ok["hidden"] = Value::Array(eight);
        assert!(parse_mlp(&ok).is_ok(), "8 hidden layers should parse");

        let nine: Vec<Value> = (0..9).map(|_| json!({"width": 4})).collect();
        let mut bad_layers = mlp_json();
        bad_layers["hidden"] = Value::Array(nine);
        assert!(mlp_rejects(&bad_layers), "9 hidden layers must reject");

        // The head is a scalar score: width is exactly 1.
        for bad in [0, 2, 16384] {
            let mut v = mlp_json();
            v["head"] = json!({"width": bad});
            assert!(mlp_rejects(&v), "head width {bad} must reject");
        }

        // Non-integer widths reject (no silent coercion).
        let mut negative = mlp_json();
        negative["input_width"] = json!(-1);
        assert!(mlp_rejects(&negative));
        let mut float = mlp_json();
        float["input_width"] = json!(2.5);
        assert!(mlp_rejects(&float));
        let mut stringy = mlp_json();
        stringy["hidden"] = json!([{"width": "3"}]);
        assert!(mlp_rejects(&stringy));
    }

    /// Boundary values at the maximum legal sizes: `expected_tensors()` must
    /// not wrap a 32-bit `usize` (the wasm32 lesson). The widest derived
    /// dimension is `16384 × 16384 = 268_435_456` (a hidden→hidden weight),
    /// reached here even with the maximum legal input width of 4096.
    #[test]
    fn mlp_max_bounds_derive_without_wrapping() {
        let hidden: Vec<Value> = (0..8).map(|_| json!({"width": 16384})).collect();
        let mut v = mlp_json();
        v["input_width"] = json!(4096);
        v["hidden"] = Value::Array(hidden);
        let spec = parse_mlp(&v).expect("maximum-size spec parses");
        let tensors = spec.expected_tensors();
        assert_eq!(tensors.len(), 2 + 2 * 8);
        assert_eq!(tensors[0].1, vec![16384, 4096]);
        // hidden.1.weight is [16384, 16384]: the largest derived element
        // count, 268_435_456, which fits a 32-bit usize by a wide margin.
        assert_eq!(tensors[2].1, vec![16384, 16384]);
        let widest_elements = tensors
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .max()
            .expect("derived shapes are never empty");
        assert_eq!(widest_elements, 268_435_456);
        assert!(widest_elements <= u32::MAX as usize);
    }

    #[test]
    fn mlp_ignores_unknown_keys() {
        // The consumed surface is strict; unknown training metadata is
        // tolerated (the S3 discipline). Unknown `ia.*` metadata keys are
        // where strictness lives — the checkpoint convention, not the spec.
        let mut v = mlp_json();
        v["training_notes"] = json!("irrelevant");
        v["hidden"][0]["notes"] = json!("also irrelevant");
        assert!(parse_mlp(&v).is_ok());
    }

    #[test]
    fn mlp_canonical_json_is_order_and_space_independent() {
        let a = parse_mlp(&mlp_json()).expect("legal spec");
        let reordered = r#"{ "hidden_activation": "relu",
            "head": { "width": 1 }, "hidden": [ {"width": 3}, {"width": 4} ],
            "input_width": 2, "schema_version": 1, "model_type": "ia-mlp" }"#;
        let b = MlpSpec::from_json(reordered).expect("same spec, different order");
        assert_eq!(a, b);
        assert_eq!(a.canonical_json(), b.canonical_json());
        assert_eq!(
            a.canonical_json(),
            "{\"head\":{\"width\":1},\"hidden\":[{\"width\":3},{\"width\":4}],\
             \"hidden_activation\":\"relu\",\"input_width\":2,\"model_type\":\"ia-mlp\",\
             \"schema_version\":1}"
        );
        // The canonical form re-parses to the same spec (round-trip).
        assert_eq!(
            MlpSpec::from_json(&a.canonical_json()).expect("canonical form parses"),
            a
        );
    }

    #[test]
    fn activation_names_are_the_grammar_strings() {
        assert_eq!(Activation::Relu.as_str(), "relu");
        assert_eq!(Activation::Tanh.as_str(), "tanh");
        assert_eq!(Activation::GeluErf.as_str(), "gelu_erf");
        assert_eq!(Activation::GeluTanh.as_str(), "gelu_tanh");
    }
}

#[cfg(kani)]
#[allow(non_snake_case)]
#[kani::proof]
fn expected_tensors_count_is_5_16L_2() {
    let layers: usize = kani::any();
    kani::assume((1..=3).contains(&layers));
    let json = format!(
        r#"{{"model_type":"bert","architectures":["BertModel"],"hidden_act":"gelu",
            "position_embedding_type":"absolute","hidden_size":8,
            "num_attention_heads":4,"num_hidden_layers":{layers},
            "intermediate_size":16,"vocab_size":32,"type_vocab_size":2,
            "max_position_embeddings":16,"layer_norm_eps":1e-12,"pad_token_id":0}}"#
    );
    let spec = BertSpec::from_config_json(&json).expect("generated config parses");
    let tensors = spec.expected_tensors();
    assert_eq!(tensors.len(), 5 + 16 * layers + 2);

    let mut names: Vec<&str> = tensors.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), tensors.len());
}
