# MLP Specs & Validated Checkpoints

Katakuri 0.2.0 adds a second architecture family — the `ia-mlp` feedforward
spec — and the validated checkpoint loader Amari's learned rewrite needs
(Amari 0.25.0 Tasks 40–45, wishlist W1 + W2 + W5). They land together because
they are coupled: the loader's shape/schema check and its canonical parameter
order both come from the spec.

## `MlpSpec` — the `ia-mlp` family

`BertSpec` and `MlpSpec` are **separate families sharing invariants**, not one
generalized config parser. A BERT-shaped model still rejects under `MlpSpec`,
and an MLP rejects under `BertSpec` — each error names the families it knows
(`"bert"` / `"ia-mlp"`) so a caller can route.

```json
{
  "model_type": "ia-mlp",
  "schema_version": 1,
  "input_width": 64,
  "hidden": [{"width": 256}],
  "head": {"width": 1},
  "hidden_activation": "gelu_erf"
}
```

Strict grammar, parsed by `MlpSpec::from_json`:

| Field | Rule |
|---|---|
| `model_type` | required, exactly `"ia-mlp"` |
| `schema_version` | required, exactly `1` (forward-compat gate) |
| `input_width` | required, `1..=4096` |
| `hidden` | required array, length `0..=8`; each `{"width": w}`, `1..=16384` |
| `head` | required, `{"width": 1}` — the head produces a scalar score |
| `hidden_activation` | required: `relu`, `tanh`, `gelu_erf`, `gelu_tanh` |

Every bound is checked with plain comparisons — no `+1`, no subtraction, no
multiplication at parse time — so the checks cannot wrap on a 32-bit target
(the wasm32 lesson from S3, re-applied). The largest derived weight is
`16384 × 16384 = 268_435_456` elements, which fits a 32-bit `usize`;
`cargo check --target wasm32-unknown-unknown` keeps that true.

Unrecognized keys inside the spec are tolerated and ignored (the S3
discipline: the consumed surface is strict, training metadata is not ours).
The *checkpoint metadata* namespace is where unknown keys reject — see below.

### Canonical parameter order

`expected_tensors()` returns names and shapes **depth-first, layer order**:

```text
hidden.0.weight  [w0, input]
hidden.0.bias    [w0]
hidden.1.weight  [w1, w0]
hidden.1.bias    [w1]
…
head.weight      [1, w_last]
head.bias        [1]
```

That order *is* the canonical ordering the checkpoint round-trip guarantee
relies on — it is not an implementation detail.

### Activations are named semantics, not code

`Activation` names the function; Katakuri never evaluates it:

- `Relu` — `max(0, x)`
- `Tanh`
- `GeluErf` — exact `0.5x(1+erf(x/√2))`; erf is the consumer's
- `GeluTanh` — candle's `GeluApproximate` semantics (the exact function
  Ijima pinned), now a *named* choice instead of an inherited backend detail

## `load_validated` — the trust policy

0.1.0's `SafeTensors::parse` takes `&[u8]` and is value-blind. The checkpoint
module is the scoped, named reversal: it opens a file, applies an explicit
`TrustPolicy`, parses through S1 unchanged, and optionally scans values.
`parse` itself stays value-blind.

Every policy field is an explicit choice — nothing is silently on:

```rust
use katakuri::checkpoint::{TrustPolicy, load_validated};

let policy = TrustPolicy {
    regular_file_only: true,   // symlinks/FIFOs/devices/dirs reject; hardlinks pass
    byte_cap: 64 << 20,        // checked before reading, read bounded by the same cap
    verify_schema_hash: true,  // ia.schema-hash vs the parsed spec's canonical JSON
    verify_model_hash: true,   // ia.model-hash vs the data section
    reject_nonfinite: true,    // NaN / ±Inf in F32 tensors reject
};

match load_validated(std::path::Path::new("model.safetensors"), &policy) {
    Ok(checkpoint) => {
        // spec, manifest, metadata, and tensors in canonical order
        let _ = checkpoint.spec.input_width;
    }
    Err(error) => eprintln!("checkpoint rejected: {error}"),
}
```

The pipeline is ordered and failure is total (no partial loads):

1. regular-file check (`symlink_metadata` + `O_NOFOLLOW` on Unix + `fstat` on
   the opened descriptor; plain metadata elsewhere)
2. byte cap
3. read
4. S1 parse (value-blind)
5. metadata convention (`ia.*`)
6. `ia.spec` → `MlpSpec` (strict)
7. shape/schema agreement with `expected_tensors()` (diagnostics-first)
8. schema/model hash verification — **before** the value scan, so a tampered
   file reports a hash mismatch rather than a value diagnostic
9. optional nonfinite scan
10. manifest assembly in canonical order

Hardlinks are regular files and are accepted. Subnormals are finite and are
**kept**: they are legal f32 values, and whether the compute layer tolerates
them is its question, not the loader's. A failed load returns no partial
result — `ValidatedCheckpoint` exists only when every enabled check passed.

## The metadata convention (`ia.*`)

| Key | Meaning |
|---|---|
| `ia.spec` | the spec JSON itself (required) |
| `ia.schema-hash` | sha256 of the parsed spec's canonical JSON |
| `ia.model-hash` | sha256 of the safetensors data section |
| `ia.provenance.training-data` | free-form replay/training-data binding, `1..=4096` bytes |

Any other `ia.*` key **rejects** (`UnknownIaMetadata`): the namespace is
strict, and forward compatibility is a `schema_version` bump, not a silent
ignore. Metadata outside the `ia.` namespace is the writer's business and is
carried through untouched.

`MlpSpec::canonical_json()` is the deterministic, key-sorted serialization the
schema hash covers, so semantically identical specs hash identically
regardless of input whitespace or key order.

## The layout manifest (T0 co-sign, unstable)

`TensorManifest` is the shared Katakuri ⇄ `borsalino-tensor` contract
(Borsalino PR #61 §4.4): canonical order, F32 pinned, row-major contiguous, no
padding, plus the determinism metadata the compute layer needs (no
bit-pattern-changing fast paths — wishlist W3's Katakuri side).

It is **data, not buffers**: no `zunesha` dependency, and binding entries to
device buffers is the consumer's concern. The type is **not frozen** in 0.2.0
— it freezes only after the Borsalino tensor-extension session reviews it
against T0's zero-copy needs. That gate is named in the module docs.

## Feature gates

| Surface | Requires |
|---|---|
| `MlpSpec`, `Activation`, `TensorManifest` (metadata layer) | nothing — pure data, no `sha2`, no file I/O |
| `katakuri::digest` (`artifact_sha256`, `sha256_hex`) | `digest` |
| `katakuri::checkpoint` (`load_validated`, `TrustPolicy`) | `std` + `digest` |
| `katakuri::hub` (`sha256_hex` re-exported) | `hub` (**implies** `digest`) |

A consumer that only describes architectures never links the hash or file-I/O
surfaces (wishlist W6). The `hub` feature carries `digest` in its dependency
list; a hand-rolled `--cfg feature="hub"` build without `digest` is a compile
error, and the crate-root docs assert the graph in a doctest.

## What this is not

Katakuri does not compute the forward pass. The MLP's matmuls, the GELU, and
the determinism question all live in the consuming engine — Amari's
hand-rolled tiny matmul in 0.25.0, Borsalino's kernels later. This page
describes weights, shapes, ordering, and trust decisions, and stops there.

## Canonical form and physical layout

Two independent canonicalizations keep checkpoints stable:

- **`MlpSpec::canonical_json()`** — a hand-rolled, fixed-key-order document
  (pinned byte-for-byte by a unit test). Its bytes are what `ia.schema-hash`
  digests. It deliberately does not iterate any `serde_json` map: a
  downstream consumer enabling `serde_json`'s `preserve_order` feature would
  otherwise change the digest through feature unification.
- **`ValidatedCheckpoint::canonical_data_bytes()`** — tensor data
  concatenated in canonical parameter order (`expected_tensors()` order,
  depth-first by layer). Loading honors the file's physical offsets and does
  **not** require canonical arrangement; the bit-stable guarantee is over
  this canonical concatenation, not the input bytes.

The trust-policy open uses `libc`'s per-ABI `O_NOFOLLOW`/`O_NONBLOCK`
(unix-only dependency) — hand-derived flag tables were wrong twice during
S5's development, which is exactly the failure mode a dependency on the
verified source eliminates.
