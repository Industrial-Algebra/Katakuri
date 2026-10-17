# Katakuri S3 — Model specs + swap boundary

**Status:** contract for generation dispatch (ia-moment AUTHOR step).
**Branch:** `feature/s3-model-specs` (fixtures committed `0867088`).
**Predecessors:** S1 `safetensors` (merged, PR #2), S2 `tokenizer` (merged, PR #3).

## 1. Goal

Turn `src/model.rs` from a doc-only placeholder into the typed **model-spec
layer**: a strict `config.json` parser (`BertConfig` → validated `BertSpec`),
a weight-layout manifest derived from the spec, and validation of a parsed
safetensors weight set against that manifest. Katakuri stays a **model-
ecosystem layer that never computes**: S3 produces validated *data* (specs,
expected tensor shapes) — no forward pass, no candle dependency.

The slice ends with the **swap-boundary port plan** (§8): how Ijima's
`embeddings_candle.rs` adopts Katakuri for config parsing + weight
validation while keeping candle for compute (their slice, not ours).

## 2. Non-goals

- No network/hub access (S4). No forward pass, no tensor math, no candle.
- No half-precision (F16/BF16) support — F32-only strict subset; reject
  with a typed error naming the subset.
- No non-BERT architectures (`model_type != "bert"` rejects).
- No numerics validation of weight *values* — layout only (names, dtypes,
  shapes, set completeness). Values are S1's bytes and candle's math.
- No new dependencies. Zero. (`serde_json` and this crate's own
  `safetensors`/`tokenizer`/`error` modules only.)

## 3. Reference intel (observed, this machine)

Real `sentence-transformers/all-MiniLM-L6-v2`, cached at
`~/.cache/katakuri-test/` (`MiniLM-config.json`, `MiniLM-L6-v2-model.safetensors`):

**config.json** (verbatim shape):

```json
{
  "_name_or_path": "nreimers/MiniLM-L6-H384-uncased",
  "architectures": ["BertModel"],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 384,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 6,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.8.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```

**model.safetensors: 104 manifest entries**, pattern (per-dtype shape):

| count | name pattern | dtype | shape (MiniLM) |
|---|---|---|---|
| 1 | `embeddings.word_embeddings.weight` | F32 | [vocab, H] = [30522, 384] |
| 1 | `embeddings.position_embeddings.weight` | F32 | [max_pos, H] = [512, 384] |
| 1 | `embeddings.token_type_embeddings.weight` | F32 | [type_vocab, H] = [2, 384] |
| 1 | `embeddings.LayerNorm.weight` | F32 | [H] |
| 1 | `embeddings.LayerNorm.bias` | F32 | [H] |
| 1 | `embeddings.position_ids` | **I64** | **[1, max_pos] = [1, 512]** — **non-weight buffer, NOT a parameter** |
| L | `encoder.layer.N.attention.self.query.weight` | F32 | [H, H] |
| L | `encoder.layer.N.attention.self.query.bias` | F32 | [H] |
| L | `encoder.layer.N.attention.self.key.weight` | F32 | [H, H] |
| L | `encoder.layer.N.attention.self.key.bias` | F32 | [H] |
| L | `encoder.layer.N.attention.self.value.weight` | F32 | [H, H] |
| L | `encoder.layer.N.attention.self.value.bias` | F32 | [H] |
| L | `encoder.layer.N.attention.output.dense.weight` | F32 | [H, H] |
| L | `encoder.layer.N.attention.output.dense.bias` | F32 | [H] |
| L | `encoder.layer.N.attention.output.LayerNorm.weight` | F32 | [H] |
| L | `encoder.layer.N.attention.output.LayerNorm.bias` | F32 | [H] |
| L | `encoder.layer.N.intermediate.dense.weight` | F32 | [I, H] |
| L | `encoder.layer.N.intermediate.dense.bias` | F32 | [I] |
| L | `encoder.layer.N.output.dense.weight` | F32 | [H, I] |
| L | `encoder.layer.N.output.dense.bias` | F32 | [H] |
| L | `encoder.layer.N.output.LayerNorm.weight` | F32 | [H] |
| L | `encoder.layer.N.output.LayerNorm.bias` | F32 | [H] |
| 1 | `pooler.dense.weight` | F32 | [H, H] |
| 1 | `pooler.dense.bias` | F32 | [H] |

Layer indices `N` are `0..num_hidden_layers`. Weight count invariant:
`5 + 16·L + 2` parameters, plus the optional `position_ids` buffer
(MiniLM: 103 + 1 = 104 entries). `position_ids` presence is **optional**
(candle derives an arange when absent); when present it must be exactly
I64 `[1, max_position_embeddings]`.

## 4. Public API (exact)

`src/model.rs` gains, in order:

```rust
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
    /* private: hidden_size, num_attention_heads, num_hidden_layers,
       intermediate_size, vocab_size, type_vocab_size,
       max_position_embeddings, layer_norm_eps: f64, pad_token_id: u32,
       hidden_act: HiddenAct */
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
    pub fn from_config_json(json: &str) -> Result<Self, KatakuriError>;

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
    pub fn validate_weights(&self, weights: &crate::safetensors::SafeTensors<'_>)
        -> Result<(), KatakuriError>;

    /// Cross-checks the spec against a tokenizer: the tokenizer's
    /// vocabulary size must equal the config's `vocab_size`.
    ///
    /// # Errors
    ///
    /// [`crate::error::KatakuriError::InvalidConfigJson`] on mismatch
    /// (detail carries both numbers).
    pub fn check_tokenizer_compatibility(
        &self,
        tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Result<(), KatakuriError>;

    // Accessors (all #[must_use], returning copies):
    pub fn hidden_size(&self) -> usize;
    pub fn num_attention_heads(&self) -> usize;
    pub fn num_hidden_layers(&self) -> usize;
    pub fn intermediate_size(&self) -> usize;
    pub fn vocab_size(&self) -> usize;
    pub fn type_vocab_size(&self) -> usize;
    pub fn max_position_embeddings(&self) -> usize;
    pub fn layer_norm_eps(&self) -> f64;
    pub fn pad_token_id(&self) -> u32;
    pub fn hidden_act(&self) -> HiddenAct;

    /// The expected weight tensor set as `(name, shape)` pairs —
    /// `5 + 16·L + 2` entries, layer indices `0..L`, shapes derived
    /// from the accessors. Excludes the optional `position_ids`
    /// buffer. Order: embeddings, then layers in index order (q/k/v →
    /// attn-out → intermediate → output per §3's table), then pooler.
    #[must_use]
    pub fn expected_tensors(&self) -> Vec<(String, Vec<usize>)>;
}
```

Accessors are trivial field reads. `expected_tensors` is the single
derivation point — `validate_weights` must be implemented ON TOP of it
(derive expectations once, compare against `weights.names()` +
`weights.tensor(name)`, collecting every divergence).

## 5. Errors (`src/error.rs`, verbatim additions)

After the existing `UnsupportedTruncationDirection` variant, matching the
surrounding doc-comment style:

```rust
    /// A `config.json` value is malformed, missing, or outside the
    /// supported BERT subset. `detail` names the field and the problem.
    #[error("invalid model config: {detail}")]
    InvalidConfigJson {
        /// Field-level description of the divergence.
        detail: String,
    },

    /// A parsed weight set diverges from the spec's layout manifest.
    /// `detail` lists every divergence found (missing, unexpected,
    /// wrong-shape, wrong-dtype), one per line.
    #[error("weight layout mismatch: {detail}")]
    WeightLayoutMismatch {
        /// All divergences, one per line.
        detail: String,
    },
```

(Adjust the `#[error]`/doc phrasing to match the enum's existing style
exactly — inspect the file first; the variant names and fields are fixed.)

## 6. Module docs (`src/model.rs` header, verbatim shape)

Replace the placeholder header comment block (keep the license header)
with — mind the S1-r2 lesson: module-level `//!` docs resolve at
**crate-root scope**, so every intra-doc link to an item inside a sibling
module needs the full `crate::` path:

```rust
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
```

`src/lib.rs` crate docs: update the `model` bullet from "Lands with S3"
to landed phrasing ("typed BERT config + weight-layout manifests,
validated against parsed safetensors; the Ijima swap boundary") — same
one-line shape as S2's bullet.

## 7. Test plan

`src/model.rs` `#[cfg(test)] mod tests` (unit, in-file), helper
`fn tiny() -> BertSpec` reading `tests/fixtures/tiny-config.json` via
`include_str!` (fixtures committed at `0867088`; tiny model =
L=2, H=8, V=32, P=16, T=2, I=16, heads=4 → 39 weights + position_ids):

1. `parses_tiny_and_real_shapes` — tiny parses; all accessors return
   the fixture values; `expected_tensors().len() == 39`; spot-check
   three shapes (`embeddings.word_embeddings.weight` == [32, 8],
   `encoder.layer.1.intermediate.dense.weight` == [16, 8],
   `pooler.dense.weight` == [8, 8]).
2. `parses_real_minilm_config` — `include_str!` a copy of the real
   config committed as `tests/fixtures/minilm-config.json` (see §9);
   H=384, L=6, heads=12, I=1536, V=30522, eps=1e-12, act=Gelu,
   `expected_tensors().len() == 103`.
3. `rejects_non_bert_model_type` — `model_type: "gpt2"` (and absent
   `model_type`) reject with `InvalidConfigJson`.
4. `rejects_bad_architectures_and_position_type` — `architectures:
   ["AutoModel"]` rejects; `position_embedding_type: "relative_key"`
   rejects; both fields absent still parse.
5. `rejects_unknown_hidden_act` — `"gelu_new"` rejects; the three
   subset strings parse (parametrize one config each).
6. `rejects_nonpositive_and_indivisible_sizes` — `hidden_size: 0`;
   `vocab_size: -1`; `hidden_size: 9` with heads 4 (indivisible);
   `layer_norm_eps: 0.0`; `layer_norm_eps: NaN` — all reject.
7. `validates_tiny_weights_roundtrip` —
   `SafeTensors::parse(include_bytes!("fixtures/tiny-model.safetensors"))`
   then `tiny().validate_weights(&st)` is `Ok`; also assert
   `st.tensor_count() == 40`.
8. `weight_layout_collects_all_divergences` — parse tiny weights, then
   validate against a MUTATED spec? No — mutate the JSON config instead
   is awkward for shape divergence; instead build the spec from a
   hand-written config with `hidden_size: 16` (same L=2/V=32/...): every
   [8]-shaped tensor is now expected [16] → error `detail` mentions
   multiple lines; assert `detail.lines().count() >= 20` and that it
   mentions at least one `wrong shape` and the `embeddings.LayerNorm.weight`
   name. (Constructing a real mismatched safetensors is unnecessary —
   the comparison path is exercised; shape mismatch against the SAME
   fixture is the cleanest multi-divergence driver.)
9. `weight_layout_flags_missing_unexpected_and_dtype` — from the tiny
   spec's expected set vs the tiny file: (a) simulate missing by
   validating a spec with `num_hidden_layers: 3` → detail mentions
   `encoder.layer.2...` as missing... wait, that ADDS expectations →
   missing lines for all 16 layer-2 tensors; (b) unexpected: spec with
   `num_hidden_layers: 1` → 16 `unexpected` lines for layer 1; (c)
   dtype: hand-construct a minimal safetensors bytes (like S1 tests do,
   header JSON + zero payload) with one F16 tensor named
   `embeddings.word_embeddings.weight` + everything else per tiny spec
   → `wrong dtype` line. Do (c) via a small helper that clones the tiny
   header from the parsed SafeTensors? The manifest isn't serializable
   back — instead hand-write the full 39-entry header JSON in the test
   via `expected_tensors()` from the spec (names+shapes), all F32,
   except the one F16 mutation, zero payload — this doubles as a
   round-trip check of `expected_tensors` naming.
10. `position_ids_optional_but_typed` — the tiny file HAS position_ids
    (I64 [1,16]) and validates Ok; the hand-written-header variant from
    test 9 WITHOUT position_ids also validates Ok; hand-written with
    `position_ids` F32 → `wrong dtype`-style line mentioning it.
11. `tokenizer_compatibility` — `BertSpec::check_tokenizer_compatibility`
    with the S2 fixture tokenizer (`minilm-vocab.txt` has 30522 lines →
    need a spec whose `vocab_size == 30522`, i.e. the real-MiniLM
    config from test 2) → Ok; with tiny spec (V=32) → `InvalidConfigJson`
    mentioning both numbers.

Env-gated `tests/model_spec.rs` (integration, same shape as S2's
env-gated file):

12. `real_file` (`#[ignore]`): reads `$KATAKURI_TEST_CONFIG_JSON` and
    `$KATAKURI_TEST_SAFETENSORS` (both `~/.cache/katakuri-test/MiniLM-*`),
    parses spec, validates the real 104-tensor file Ok,
    `check_tokenizer_compatibility` against
    `$KATAKURI_TEST_TOKENIZER_JSON` (or `MiniLM-vocab.txt` —
    `from_bert_vocab_txt`) Ok, `expected_tensors().len() == 103`,
    `hidden_act() == HiddenAct::Gelu`. Skips with a pass-style message
    when the env vars are unset (same pattern as the S2 real-file test).

`#[cfg(kani)]` harness (verify feature pattern, `src/model.rs` bottom):

13. `expected_tensors_count_is_5_16L_2`: for arbitrary small
    `L in 1..=3` (construct spec fields directly via a
    `#[cfg(kani)]`-gated test constructor or by parsing a generated
    JSON string — prefer parsing a generated string: public API only):
    `expected_tensors().len() == 5 + 16·L + 2` and every name is
    unique (collect into set, compare lens). Keep proofs small: bound
    L, pre-build the JSON with format!.

Expected counts: 11 unit tests in `mod tests` + 1 env-gated integration
+ kani under cfg(kani). Lib default ≈ 56 (45 + 11), verify ≈ 57 (+kani's
cargo test doesn't count; verify adds the cfg(feature="verify")
harnesses compile). Integration file: 0 run / 1 ignored without env.

## 8. Swap-boundary port plan (Ijima's slice — lands in this doc only)

`embeddings_candle.rs::load` today: hand-downloads three files via
hf-hub, parses `config.json` into candle's `Config` (serde, tolerant),
**overrides `hidden_act = GeluApproximate` unconditionally**, mmaps
weights into `VarBuilder::from_mmaped_safetensors`, `BertModel::load`.
After S3+S4, the port:

1. Replace the config parse with `BertSpec::from_config_json` (strict:
   the consumer gains typed rejection of non-BERT/malformed configs
   where today serde-lenience would pass garbage downstream).
2. Replace the three ad-hoc downloads with Katakuri hub (S4) once it
   lands; until then keep hf-hub downloads, but feed the bytes through
   `SafeTensors::parse` + `BertSpec::validate_weights` before mmap —
   catches layout divergence (wrong revision, truncated repo, dtype
   drift) BEFORE `BertModel::load`'s opaque key-lookup errors.
3. Keep candle's `Config` for `BertModel::load` (compute stays in
   candle — Katakuri never computes); construct it field-by-field from
   the `BertSpec` accessors. **Pin the existing override**: the
   reference consumer forces `GeluApproximate` regardless of config —
   the port must preserve that behavior bit-for-bit (parity with the
   pi-mempalace migration embeddings) rather than "fixing" it to the
   config's `gelu`; changing it is a deliberate, separately-reviewed
   embedding-semantics change.
4. `check_tokenizer_compatibility(spec, tokenizer)` at construction —
   vocab-size drift between tokenizer.json and config.json becomes an
   early typed error instead of an embedding-quality bug.

## 9. Files the dispatch touches

- `src/model.rs` — full implementation + tests + kani (as §4/§7).
- `src/error.rs` — the two variants (§5).
- `src/lib.rs` — the one-line doc bullet update (§6).
- `tests/model_spec.rs` — new env-gated integration test (§7.12).
- `tests/fixtures/minilm-config.json` — verbatim real config (§3) —
  the generator (this session) commits it with the other fixture.
- `CHANGELOG.md` — one bullet under `## [Unreleased]` → `### Added`:
  "`model`: typed BERT `BertSpec` (strict config parsing), weight-layout
  manifests validated against parsed safetensors, tokenizer
  compatibility check, and the Ijima swap-boundary port plan."
- `Cargo.lock` — only if `cargo` insists; no `Cargo.toml` changes.

## 10. Gate matrix (all must pass, outputs piped — never full logs)

```
cargo test -p katakuri                              # ~56 lib + goldens + 1 ignored
cargo test -p katakuri --features verify            # ~57 lib
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps --features verify
# kani (workspace has the harness pattern):
cargo test -p katakuri --features verify --_cfg kani 2>/dev/null | tail -3 \
  # or per-repo convention from S1 — inspect how S1's kani harnesses run
# env-gated (manual, this session only):
KATAKURI_TEST_CONFIG_JSON=~/.cache/katakuri-test/MiniLM-config.json \
KATAKURI_TEST_SAFETENSORS=~/.cache/katakuri-test/MiniLM-L6-v2-model.safetensors \
KATAKURI_TEST_TOKENIZER_JSON=~/.cache/katakuri-test/MiniLM-tokenizer.json \
  cargo test --test model_spec -- --ignored
```

## 11. Ambiguity protocol

Resolve in favor of strictness and document the resolution in the
completion report; never silently guess a semantic. If any §4 signature
conflicts with what `src/error.rs`/`src/safetensors.rs` actually expose,
adapt the call sites to the real API and record the delta.
