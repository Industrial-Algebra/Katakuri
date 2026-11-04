# Katakuri optimizer-state checkpoint convention (S6 / wishlist W4)

- **Status:** frozen convention, version 1 (`optimizer_state_version = 1`)
- **Date:** 2026-11-04
- **Slice:** S6 (W4). Amari Task 42 (bounded AdamW training) is scheduled, so
  the S5 deferral condition for W4 is met.
- **Executable half:** `src/optimizer.rs` (`OptimizerState`,
  `load_optimizer_state`, `expected_optimizer_tensors`,
  `OptimizerStateError`).
- **Depends on:** S5 `checkpoint` (`TrustPolicy`, `load_validated`,
  `ia.*` metadata convention) and S5 `model::MlpSpec`.
- **Non-goals (S5 doctrine stands):** no compute, no autodiff, no optimizer
  math, no Zunesha dependency, no hub/tokenizer changes, no change to
  `SafeTensors::parse`'s value-blindness.

## 1. Goal

Training resumes must be **portable and auditable**. Katakuri never runs
AdamW, but it fixes the artifact's shape so that two independent writers —
Amari's hand-rolled bounded optimizer today, anything later — produce
byte-comparable, mutually loadable state, and so that a loader can say
*exactly* why a resume is unusable instead of silently training from the
wrong moments.

This document is the contract. `src/optimizer.rs` implements it; the two are
versioned together by `optimizer_state_version`.

## 2. Naming

| Tensor role | Name | Shape | Dtype |
|---|---|---|---|
| parameter (weight/bias) | `<param-name>` (unchanged) | as the spec | F32 |
| AdamW first moment `m` | `<param-name>.adamw.m` | **identical to the parameter** | **identical to the parameter** (F32) |
| AdamW second moment `v` | `<param-name>.adamw.v` | **identical to the parameter** | **identical to the parameter** (F32) |

For an `ia-mlp` spec of input width 2 and one hidden layer of width 3, the
canonical moment set is exactly:

```
head.bias.adamw.m            [1]
head.bias.adamw.v            [1]
head.weight.adamw.m          [1, 3]
head.weight.adamw.v          [1, 3]
hidden.0.bias.adamw.m        [3]
hidden.0.bias.adamw.v        [3]
hidden.0.weight.adamw.m      [3, 2]
hidden.0.weight.adamw.v      [3, 2]
```

(`expected_optimizer_tensors(&spec)` returns exactly this vector; a test pins
it byte for byte.)

**Reserved suffixes.** `.adamw.m` and `.adamw.v` are reserved by the
convention. A tensor name ending in either suffix is **optimizer state, never
a parameter**. Consequences, all enforced:

- A moment name must be `<parameter name>` + suffix. A suffix-named tensor
  whose stem is not a parameter of the spec (`hidden.7.bias.adamw.m` for a
  one-hidden-layer spec) is `UnexpectedTensor`.
- Nested suffixes (`hidden.0.weight.adamw.m.adamw.m`) are `UnexpectedTensor`:
  they are neither a parameter nor a moment of this spec.
- A *parameter* name that ends in a reserved suffix would collide with the
  moment namespace. It cannot arise from the `ia-mlp` grammar (parameter names
  are `hidden.<i>.weight|bias` and `head.weight|bias`), and the naming rule
  makes it illegal for any future spec family: classify with
  `split_moment_name` before treating a name as a parameter.

## 3. Dtype and layout

- **Dtype:** a moment's dtype is **identical to its parameter's** dtype. For
  the `ia-mlp` family that is F32, row-major. No f64 accumulators at rest, no
  bf16/f16 narrowing: a resume must reload the exact elements that were
  updated, and a dtype change is a semantic change the loader cannot verify.
- **Layout:** a moment's shape is **identical to its parameter's** shape. No
  transposes, no reshapes, no flattening at rest.
- **Encoding:** safetensors native — little-endian element bytes, contiguous
  spans, no padding, `data_offsets` in the header (S1 validates all of this).
- **Values are not inspected by `parse`.** `OptimizerState::parse` is
  value-blind like S1. Values are only scanned when a `TrustPolicy` asks for
  it (`reject_nonfinite`), and subnormals are kept — they are legal f32
  values (S5 doctrine).

## 4. Metadata (`__metadata__`)

| Key | Type | Value |
|---|---|---|
| `optimizer_state_version` | `u32` (decimal string) | `1` — this convention |
| `optimizer` | string | `"adamw"` |
| `step` | `u64` (decimal string) | the training step the state belongs to |
| `ia.spec` | string | *optional*: the parameter checkpoint's `MlpSpec` JSON |
| `ia.schema-hash`, `ia.model-hash`, `ia.provenance.*` | — | S5 convention, unchanged |

Rules:

- All three optimizer keys are **required**. A missing key is
  `MissingMetadata`; a value that does not parse is `MalformedMetadata`; a
  version this crate does not implement is `UnsupportedStateVersion`; an
  optimizer name other than `adamw` is `UnsupportedOptimizer`. Forward
  compatibility is a **version bump**, never a guess.
- `step` is **metadata only** — never a per-tensor, never inferred from a
  tensor's contents. `step = 0` is legal (a resume before the first update).
- Every `ia.*` key obeys the S5 metadata convention: unknown `ia.*` keys
  reject (`UnknownIaMetadata`). When `ia.spec` is present it must denote the
  caller's spec, otherwise `SpecMismatch` — state that belongs to a different
  architecture is a silent-failure generator. Agreement is **semantic**: the
  declared document is parsed and compared in canonical form (the same
  canonical JSON `ia.schema-hash` digests), so whitespace, key order, and any
  tolerated extra field are irrelevant. A declaration that is not a valid spec
  at all is `MalformedSpecMetadata` — its own typed failure, never a
  mismatch.
- Declared hashes are verified when the trust policy asks:
  `verify_model_hash` checks `ia.model-hash` against the data section,
  `verify_schema_hash` checks `ia.schema-hash` against the caller's spec's
  canonical JSON. An artifact that carries neither cannot satisfy those flags
  and rejects with `<absent>` (S5 behavior).

## 5. Canonical order and byte reproducibility

Tensors are read into **lexicographic order of their names** — the
safetensors canonical form, and the order of S1's `SafeTensors::names()`.
`ValidatedOptimizerState::tensors` is in that order, and
`canonical_data_bytes()` concatenates the tensors' bytes in that order.

The guarantee, stated exactly (the S5 r1 P2 lesson): canonical ordering is a
property of the **validated view**, not of the input file. A file whose
physical tensor order differs still yields identical `canonical_data_bytes()`
and identical tensor order. Round-tripping a checkpoint therefore means
*canonicalizing* it, not preserving its physical layout.

Lexicographic (not numeric) order is deliberate: it is byte-comparable across
languages without a numeric-name parser, and it is what the header's JSON map
already sorts by.

## 6. Completeness: all-or-nothing

An optimizer-state checkpoint is **valid iff every parameter tensor of the
spec has both moments present**. Partial state is never accepted, never
partially trusted, never repaired:

- one moment of a parameter missing → `MissingMoment { param, moment }`
- any tensor outside the derived moment set → `UnexpectedTensor { name }`

Accepted artifact shapes (exactly two):

1. **Moments only** — the tensor set is exactly
   `expected_optimizer_tensors(&spec)`. The parameters live in the model
   checkpoint; the spec is the shape/dtype oracle.
2. **Weights alongside moments** — the full parameter set *plus* the full
   moment set (the wishlist's "alongside model weights" artifact). Parameter
   tensors keep their plain names and must match the spec's dtype/shape
   (`ParameterMismatch`); a partial weight set is `PartialParameters` because
   full moments + incomplete weights is ambiguous.

Rationale: a resume that silently ignores one missing moment would restart
that parameter's moment at zero — a training bug that looks like a
hyperparameter change weeks later. The typed error names the parameter and
which moment is absent, so the fix is mechanical.

## 7. Validation pipeline

`load_optimizer_state(path, policy, spec)` — ordered, failure is total (no
partial loads), and it **composes the S5 trust seam rather than bypassing
it**:

1. **Trust policy** — `regular_file_only` (symlinks/FIFOs/devices/directories
   reject), `byte_cap`. Same code path as `load_validated`.
2. **S1 parse** — value-blind; layout, spans, dtype strings validated.
3. **`ia.*` metadata convention** (S5 §5.5) — unknown `ia.*` keys reject.
4. **Optimizer metadata** — §4 above.
5. **All-or-nothing moments** — §6.
6. **No unexpected tensors** — §6, including the collision rules of §2.
7. **Dtype/shape agreement** — §3.
8. **Hashes** — `verify_model_hash`, `verify_schema_hash` (before the value
   scan, so a tampered file reports a hash mismatch, not a value diagnostic).
9. **Nonfinite scan** — `reject_nonfinite` only; NaN/±Inf in any F32 moment
   rejects (`NonfiniteTensorValue`), subnormals pass.

`OptimizerState::parse(bytes)` stops after step 2 plus canonical ordering, and
`OptimizerState::validate_optimizer_state(&spec)` runs steps 4–7. Use those
two when the bytes and the trust policy are managed elsewhere; use
`load_optimizer_state` for the whole contract.

## 8. Errors

`OptimizerStateError` is the module's typed error. S5 failures arrive as
`OptimizerStateError::Checkpoint(KatakuriError)` — the convention adds no
second vocabulary for I/O, policy, parse, metadata, hash, or nonfinite
failures.

| Variant | Rule |
|---|---|
| `Checkpoint(KatakuriError)` | trust policy, S1 parse, `ia.*` metadata, hashes, nonfinite |
| `MissingMoment { param, moment }` | all-or-nothing (§6) |
| `UnexpectedTensor { name }` | outside the derived set; nested/stray suffixes (§2, §6) |
| `PartialParameters { detail }` | incomplete parameter set alongside moments (§6) |
| `ParameterMismatch { tensor, detail }` | carried parameter ≠ spec dtype/shape |
| `MomentShapeMismatch { tensor, expected, actual }` | moment ≠ parameter shape (§3); `expected`/`actual` are bounded renderings (§8.1) |
| `MomentDtypeMismatch { tensor, expected, actual }` | moment ≠ parameter dtype (§3) |
| `MissingMetadata { key }` | required optimizer key absent (§4) |
| `MalformedMetadata { key, detail }` | version/step does not parse (§4) |
| `UnsupportedStateVersion { found }` | version ≠ 1 (§4) |
| `UnsupportedOptimizer { found }` | optimizer ≠ `adamw` (§4) |
| `SpecMismatch { detail }` | declared `ia.spec` denotes a different spec (§4) |
| `MalformedSpecMetadata { detail }` | declared `ia.spec` is not a valid spec (§4) |

### 8.1 Bounded diagnostics

A rejection must never copy an unbounded input just to say "rejected". The
header is allowed to be 100 MB and S1 bounds neither a tensor name's length
nor a shape's rank (`[1, 1, …]` with 100,000 dims is a legal shape with a
4-byte span), so every *untrusted* value this module quotes is bounded:

| Untrusted input | Rendering |
|---|---|
| metadata value, tensor name, malformed spec detail | `bounded_excerpt`: ≤ `MAX_DIAGNOSTIC_EXCERPT_BYTES` (64) input bytes, truncated on a UTF-8 boundary, control characters escaped (a diagnostic stays one line), true byte length stamped when truncated |
| shape | `shape_excerpt`: ≤ `MAX_DIAGNOSTIC_SHAPE_DIMS` (8) dims, true rank stamped when truncated |
| list of names | `name_list_excerpt`: ≤ `MAX_DIAGNOSTIC_NAME_LIST` (4) entries, each bounded, then `... and N more` with the true remainder |

All three helpers are public: consumers writing their own diagnostics against
this convention use the same caps rather than inventing their own.

Scope of the guarantee: **every field of every `OptimizerStateError`
constructor is bounded, with no provenance assumption.** Two lessons are
folded in here:

- *Names and shapes are untrusted even when they look structural.* S1 bounds
  neither a tensor name's length nor a shape's rank.
- *A spec is not necessarily parse-validated.* `MlpSpec`'s fields are public,
  so hand-built specs are legal input to this API; a 10,000-layer spec is
  constructible and derives 20,002 parameter names. "Bounded by the spec
  grammar" is true only of parse-validated specs, so the missing-name list is
  capped and remainder-stamped instead (PR #22 round 2: that message was
  377,901 bytes before the cap).

S5 failures pass through as `Checkpoint(KatakuriError)` and keep S5's own
diagnostics, which are outside this convention's guarantee.

## 9. Reference API

```rust
use katakuri::checkpoint::TrustPolicy;
use katakuri::model::MlpSpec;
use katakuri::optimizer::{
    MomentKind, expected_optimizer_tensors, load_optimizer_state, moment_name,
};

fn load(
    spec: &MlpSpec,
    policy: &TrustPolicy,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // The name set a writer must emit (lexicographic canonical order).
    let names = expected_optimizer_tensors(spec);

    // One moment's name.
    assert_eq!(
        moment_name("hidden.0.weight", MomentKind::First),
        "hidden.0.weight.adamw.m"
    );

    // Load a complete artifact under the S5 trust policy.
    let state = load_optimizer_state(path, policy, spec)?;
    let _ = (&names, state.step, state.moments_for("hidden.0.weight"));
    Ok(())
}
```

## 10. Versioning

This is version `1`. What a version bump means: a change to the naming rule,
the metadata keys or their types, the dtype/layout identity, the ordering
rule, or the completeness rule. Adding a second optimizer (SGD with momentum,
say) is a new `optimizer` value *and* a new suffix namespace, delivered as a
new version — v1 loaders reject it by `optimizer_state_version` or
`optimizer` rather than misreading it.

Additive, non-breaking work that does **not** bump the version: new error
variants for existing rules, new accessors, documentation.
