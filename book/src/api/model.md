# `model`

Architecture specs as typed data: strict BERT `config.json` parsing
(`BertSpec`) and weight-layout manifests validated against a parsed
safetensors file.

Specs are data, never code — no forward implementations live here, and no
numerics: weight *values* are never inspected, only layout (names, dtypes,
shapes, set completeness).

## Parsing a config

```rust
use katakuri::model::{BertSpec, HiddenAct};

let spec = BertSpec::from_config_json(&std::fs::read_to_string("config.json")?)?;

assert_eq!(spec.hidden_size(), 384);
assert_eq!(spec.num_hidden_layers(), 6);
assert_eq!(spec.num_attention_heads(), 12);
assert_eq!(spec.intermediate_size(), 1536);
assert_eq!(spec.vocab_size(), 30522);
assert_eq!(spec.max_position_embeddings(), 512);
assert_eq!(spec.hidden_act(), HiddenAct::Gelu);
```

Real `all-MiniLM-L6-v2` values shown; the same code parses any BERT-class
config in the supported subset.

### What is enforced

- `model_type` must be the string `"bert"` — required; absent rejects.
- `architectures`, when present, must contain `"BertModel"`.
- `position_embedding_type`, when present, must be `"absolute"`.
- `hidden_act` must be `Gelu`, `GeluApproximate`, or `Relu`
  (`"gelu"`, `"gelu_approximate"`, `"relu"`). `"gelu_new"` and friends reject.
- Every size field (`hidden_size`, `num_attention_heads`,
  `num_hidden_layers`, `intermediate_size`, `vocab_size`, `type_vocab_size`,
  `max_position_embeddings`) must be a positive integer that fits `u32`;
  `num_hidden_layers` is additionally bounded at 1024.
- `hidden_size` must be divisible by `num_attention_heads`.
- `layer_norm_eps` must be finite and `> 0.0`; `pad_token_id` must fit `u32`.

Tolerated-and-ignored (inference-irrelevant training metadata):
`_name_or_path`, `attention_probs_dropout_prob`, `gradient_checkpointing`,
`hidden_dropout_prob`, `initializer_range`, `transformers_version`,
`use_cache`, and any other unknown keys. Documented, never read.

## Validating weights against the spec

```rust
use katakuri::safetensors::SafeTensors;

let bytes = std::fs::read("model.safetensors")?;
let weights = SafeTensors::parse(&bytes)?;

spec.validate_weights(&weights)?;
```

`validate_weights` derives the full layout manifest from the spec's numbers
(`expected_tensors()`) and checks that every expected tensor is present with
that exact shape and dtype `F32`; nothing unexpected is present, except
`embeddings.position_ids`, the sole sanctioned non-parameter tensor, and then
only as `I64 [1, max_position_embeddings]`.

Diagnostics are collected, not first-error-only: one error lists every
missing / unexpected / wrong-shape / wrong-dtype line.

For MiniLM the manifest is 103 parameters; a real file carries 104 tensors
(the extra is the position-ids buffer).

## Cross-checking the tokenizer

```rust
spec.check_tokenizer_compatibility(&tokenizer)?;
```

The rule is **capacity**, not cardinality: no id the tokenizer can emit may
index past the embedding table —

```text
max_token_id() + 1 ≤ vocab_size      (checked as: max_token_id() < vocab_size)
```

The `>=` form is deliberate. On a 32-bit target (wasm32)
`max_id as usize + 1` wraps at `u32::MAX`, which would admit an id needing
`2^32` embedding rows; `max_id >= vocab_size` is target-independent and
algebraically identical on 64-bit (review finding 12).

A tokenizer *smaller* than the table is legal — vocab-padded configs simply
leave rows unused. Only overflow rejects.

## The error surface

All malformed, missing, or out-of-subset values return
`KatakuriError::InvalidConfigJson` with a `detail` naming the field and the
problem; a weight-set divergence returns
`KatakuriError::WeightLayoutMismatch` with the full list. Both are constructed
via `thiserror`, so the `Display` text is the diagnostic.

## MLP specs (0.2.0)

Beside `BertSpec`, the model module carries the `ia-mlp` feedforward family —
`MlpSpec`, `MlpLayerSpec`, and `Activation` (named semantics;
approximations are named, never inherited). Full grammar, bounds, and the
canonical JSON form are covered in [MLP Specs & Validated
Checkpoints](../guide/mlp-specs.md).
