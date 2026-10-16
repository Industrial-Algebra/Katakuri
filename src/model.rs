// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Architecture specs as typed data: strict BERT `config.json` parsing
//! ([`BertSpec`]) and weight-layout manifests validated against parsed
//! safetensors ([`crate::safetensors::SafeTensors`]).
//!
//! Specs are data, never code — no forward implementations live here,
//! and no numerics: weight *values* are never inspected, only layout
//! (names, dtypes, shapes, set completeness). The layer replaces
//! candle-transformers `models::bert` config parsing at Ijima's swap
//! boundary (see `docs/plans/` S3 port plan); candle stays behind
//! Ijima's non-default `embeddings-candle` feature throughout.

use crate::error::KatakuriError;
use crate::safetensors::Dtype;
use serde_json::{Map, Value};

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
        if max_id as usize + 1 > self.vocab_size {
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
