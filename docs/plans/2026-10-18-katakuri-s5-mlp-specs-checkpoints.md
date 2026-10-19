# Katakuri S5 — MLP specs + validated checkpoints (+ digest re-homing)

- **Date:** 2026-10-18
- **Status:** plan contract — awaiting operator approval before dispatch
- **Source:** `docs/amari-learned-rewrite-wishlist.md` W1+W2+W5 (must-haves plus
  the coupled should-have); cross-cuts `Industrial-Algebra/Borsalino` PR #61
  (tensor extension handover, T0 manifest co-sign)
- **Replaces:** nothing. Extends S3 (model specs) and S1 (safetensors) seams.
- **Target release:** 0.2.0 (additive features; honest minor bump)

## 1. Goal

Amari's learned rewrite (0.25.0 Tasks 40–45) needs Katakuri to describe and
trust-load a **small MLP ranker** (input width ~64, one hidden layer ~256,
scalar score). Three wishlist items land together because they are coupled:

1. **W1 — `MlpSpec`**: a feedforward architecture-spec family alongside
   `BertSpec`. This is **code, not documentation** (PR #7 review finding):
   0.1.0's parser rejects every `model_type` but `"bert"`, and no layer-list
   representation exists.
2. **W2 — validated checkpoint loading**: a std-gated `load_validated` seam
   carrying Amari's Task-43 trust policy (regular-file check, byte cap,
   declared-metadata hash verification, shape/schema mismatch rejection,
   nonfinite rejection as **explicit policy**, canonical parameter ordering).
   0.1.0 has no file-opening API at all — parse takes `&[u8]` (PR #7 review
   finding) — so this is a new module, not an extension of `parse`.
3. **W5 — `digest` feature**: `sha256_hex` re-homed out of `hub` behind a
   new default-off `digest` feature (hub depends on it). Moved into S5
   because W2's hash verification cannot live behind `hub` — the coupling
   the wishlist missed.

Plus the **T0 co-sign**: the typed shape/stride/dtype **layout manifest**
Borsalino's tensor extension (PR #61 §4.4, slice T0) requires — "co-design
it with the Katakuri session before either side freezes an API." Katakuri
emits it; `borsalino-tensor` consumes it; neither side freezes it alone.

## 2. Non-goals

- **No compute.** No forward pass, no matmul, no activation evaluation
  (Borsalino PR #61 T1–T4 owns the compute half; the f32 workhorse doctrine).
- **No optimizer-state convention** (W4) — S6, only when Amari Task 42
  (training) is scheduled.
- **No tokenizer, hub-model hosting, autodiff, training, f64** (wishlist
  non-wishes; Borsalino explicit NOT-scope).
- **No change to S1–S3 doctrine.** `parse` stays value-blind and `&[u8]`;
   `BertSpec` is not generalized (separate families, shared invariants).
- **No Zunesha dependency.** The manifest is data (shape/stride/dtype), not
  buffers; `zunesha::Buffer` binding is the consumer's concern.

## 3. Reference intel

- Wishlist: `docs/amari-learned-rewrite-wishlist.md` (W1–W3, decision point).
- Borsalino PR #61 `2026-10-16-borsalino-tensor-extension-handover.md`:
  §3 op inventory (L0 gather — embedding-table row lookup is the manifest's
  first consumer), §4.4 layout contract (row-major `zunesha::Buffer`, typed
  manifest, f32 contract, explicit casts only at the load boundary), §5
  shared golden fixtures, §6 T0 co-sign gate.
- S3 patterns: strict `model_type` gating, bounded derived arithmetic
  (`num_hidden_layers ≤ 1024` precedent), `expected_tensors()` as the
  compatibility oracle, wasm32-overflow lessons (`>=` not `+1`).
- S1 doctrine: parse invariants, values never inspected; the reversal here
  is **scoped and named** (§5.4) — the scan lives at the validated seam.
- Candle lesson (`GeluApproximate` pin): activation semantics are decided,
  named, and versioned — never inherited from a backend.

## 4. Cargo.toml (exact additions, ledger entries in the PR)

```toml
[features]
# S5 digest: canonical content hashing for local artifacts (W5).
# sha2 stays PINNED at 0.11.0 — 0.11.1 requires Rust 1.88, MSRV is 1.85.
digest = ["dep:sha2"]
hub = ["dep:sha2", "digest"]          # hub now depends on digest
```

No other new dependencies. `serde_json` reuse (already a dep? — S3 uses a
strict hand-rolled reader for config; **verify during dispatch**: if S3
parses without serde, `MlpSpec` parsing matches that style — no new deps).
`std` is assumed by `checkpoint` (file I/O); the module is std-gated.

## 5. Public API (exact)

### 5.1 `MlpSpec` — `src/model.rs` (additive, no feature gate)

```rust
/// Activation with pinned, backend-independent semantics.
/// Approximations are NAMED as approximations — never inherited.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activation {
    Relu,
    Tanh,
    /// Exact GELU: 0.5x(1+erf(x/√2)). Erf evaluated by the CONSUMER;
    /// this enum names the semantics, it does not compute.
    GeluErf,
    /// Tanh-approximation GELU (candle's `GeluApproximate` semantics —
    /// the exact function Ijima pinned; the pin becomes a named choice).
    GeluTanh,
}

/// Feedforward (MLP) architecture spec — strict family `model_type: "ia-mlp"`.
/// Layer list, activations, shapes: not a hub model name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlpSpec {
    pub input_width: u32,
    /// Hidden layers, in order. Bounded: `layers.len() <= 8`.
    pub hidden: Vec<MlpLayerSpec>,
    /// Final linear to a scalar score: output width is always 1.
    pub head: MlpLayerSpec,
    pub hidden_activation: Activation,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlpLayerSpec {
    pub width: u32,        // output width of this layer
    // weight: [width, in_width] row-major; bias: [width]
}
```

Strict grammar (mirrors `BertSpec`'s construction discipline):

- `model_type: "ia-mlp"` **required**, anything else rejected with the
  close-matches error naming `"bert"` / `"ia-mlp"`.
- `schema_version: 1` required (forward-compat gate, not a promise).
- Bounds at parse (parse-valid never traps — S3 doctrine):
  `1 <= input_width <= 4096`, `0 <= hidden.len() <= 8`,
  every `1 <= width <= 16384`. Derived maxima (total params, buffer sizes)
  fit `u64`/`usize` on wasm32 without wrapping — **`>=`-style comparisons
  only** (S3 wasm32 lesson re-applied at review).
- `expected_tensors()`: canonical parameter order is **depth-first, layer
  order**: `hidden.0.weight`, `hidden.0.bias`, …, `head.weight`,
  `head.bias`. This order IS the canonical ordering W2's round-trip
  stability relies on. Reuse the S3 `ExpectedTensor` type.

### 5.2 Layout manifest — `src/manifest.rs` (the T0 co-sign type)

```rust
/// The shared Katakuri ⇄ borsalino-tensor layout contract (Borsalino PR #61
/// §4.4, slice T0). Katakuri emits; the tensor layer consumes. Frozen only
/// after both sessions review — this plan proposes, it does not freeze.
///
/// Data, not buffers: no zunesha dependency here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorManifest {
    /// Canonical parameter order (== expected_tensors() order).
    pub entries: Vec<ManifestEntry>,
    /// f32 is the contract (Borsalino §4.3). Casts happen at the consumer's
    /// load boundary, explicitly, never silently.
    pub dtype: ManifestDtype,          // F32 only in v1
    /// Row-major, contiguous, no padding — zero-copy into zunesha::Buffer.
    pub layout: ManifestLayout,        // RowMajorContiguous
    /// Determinism metadata (wishlist W3, Katakuri side): the consumer must
    /// not select fast paths that change bit patterns (e.g. no backend
    /// fast-math, no TF32); activations pinned by name (Activation enum).
    pub determinism: DeterminismPolicy,
}

pub struct ManifestEntry {
    pub name: String,          // canonical: "hidden.0.weight" | ...
    pub shape: Vec<u32>,       // [out, in]; row-major stride implied
    pub byte_offset: u64,      // into the safetensors data section
    pub byte_len: u64,
}
```

`DeterminismPolicy { dtype_pinned: F32, fast_paths: Forbidden }` is the
Katakuri-side answer to W3 — the *metadata* the compute layer needs, while
the compute itself is Borsalino's.

### 5.3 `digest` feature — re-home `sha256_hex`

- Move `sha256_hex` from `src/hub.rs` to `src/digest.rs`,
  `#[cfg(feature = "digest")]`, re-export from hub (no hub API break).
- Add `pub fn artifact_sha256(bytes: &[u8]) -> String` — same function,
  named for its local-file role; the hub internals keep using it.

### 5.4 `checkpoint` module — `src/checkpoint.rs` (std-gated, W2)

```rust
/// Trust policy for checkpoint loading (Amari Task 43). Every field is an
/// explicit choice; nothing is silently on.
#[derive(Clone, Debug)]
pub struct TrustPolicy {
    /// Reject unless the path resolves to a regular file (no symlinks,
    /// no FIFOs, no devices). Implemented via metadata + O_NOFOLLOW.
    pub regular_file_only: bool,
    /// Reject files larger than this (bytes).
    pub byte_cap: u64,
    /// Verify `__metadata__["ia.schema-hash"]` (the MlpSpec canonical hash)
    /// against the parsed spec.
    pub verify_schema_hash: bool,
    /// Verify `__metadata__["ia.model-hash"]` over the data section.
    pub verify_model_hash: bool,
    /// SCAN VALUES: reject NaN/±Inf in any f32 tensor.
    /// — the deliberate, scoped reversal of S1's "values never inspected":
    /// `parse` stays value-blind; this seam scans, because Amari's ordering
    /// decisions must never see nonfinite scores propagate (wishlist W3).
    /// Subnormals are finite and are KEPT (they are legal f32 values; the
    /// determinism question is the compute layer's, not the loader's).
    pub reject_nonfinite: bool,
}

pub struct ValidatedCheckpoint {
    pub spec: MlpSpec,
    pub manifest: TensorManifest,
    pub metadata: BTreeMap<String, String>,   // __metadata__, provenance included
    /// Parsed tensors in canonical order (names + bytes + dtypes),
    /// same representation S1's parse produces.
    pub tensors: Vec<ParsedTensor>,
}

pub fn load_validated(path: &Path, policy: &TrustPolicy) -> Result<ValidatedCheckpoint>;
```

Behavior, in order: regular-file check → size cap → read → S1 parse
(value-blind) → `MlpSpec` from `__metadata__["ia.spec"]` (strict JSON) →
shape/schema mismatch rejection via `expected_tensors()` → hash checks →
optional nonfinite scan → assemble manifest. Failure is total (no partial
loads).

### 5.5 Provenance convention (W2 bonus)

`__metadata__` keys, validated but not interpreted:
`ia.schema-hash`, `ia.model-hash`, `ia.spec` (the spec JSON itself),
`ia.provenance.training-data` (Amari binds replay hashes here — free-form
string, length-bounded 1..=4096). Unknown `ia.*` keys: **reject** (strict
grammar; forward-compat via `schema_version` bump, not silent ignores).

## 6. Errors (`src/error.rs`, additions — match existing style)

`CheckpointNotRegularFile { path }`, `CheckpointTooLarge { size, cap }`,
`SchemaHashMismatch { expected, actual }`, `ModelHashMismatch { expected,
actual }`, `NonfiniteTensorValue { tensor, which: NonfiniteKind }`,
`ProvenanceMalformed { key }`, `UnknownIaMetadata { key }`,
`MlpSpecParse` (strict-grammar suberrors, mirroring `BertSpecParse`).

## 7. lib.rs

`pub mod checkpoint;` (std-gated), `pub mod digest;` (`digest` feature),
`pub mod manifest;` (no gate — pure data), `MlpSpec`/`Activation` in
`model.rs` beside `BertSpec`. Book page `guide/mlp-specs.md` + review-record
continuation. W6 audit rides in the docs: feature map updated — metadata
(spec/manifest types) builds without `digest`/`hub`/`checkpoint`.

## 8. Port-plan addendum (Amari 0.25.0 integration boundary)

Amari Task 43 consumes exactly: `MlpSpec` (strict parse), `load_validated`
+ `TrustPolicy` (all-true profile), `TensorManifest` (hand to its scoring
executor — wherever W3's compute decision lands, Amari 0.25.0 hand-rolls
the tiny matmul over host buffers; Borsalino T1+ replaces it later without
touching Amari's side because the manifest is the stable seam).
Ijima is untouched (BertSpec path unchanged).

**Borsalino T0 co-sign gate:** this plan *proposes* `TensorManifest`; the
type is frozen only after the Borsalino tensor-extension session reviews it
against T0's buffer-interop needs (their §4.4: `zunesha::Buffer`, row-major,
zero-copy). Coordination: cross-reference PRs on both repos.

## 9. Test plan

Adversarial + property, in the S1–S4 style:

- **Grammar:** wrong/missing `model_type` (error names both families),
  missing `schema_version`, unknown `ia.*` metadata key, bounds (width 0,
  width 4097, 9 hidden layers) — all rejected at parse.
- **Wasm32 wrap:** the bound arithmetic provably can't wrap (compile-check
  the expressions; `cargo check --target wasm32-unknown-unknown`).
- **Symlinks:** fixture tree with symlink/FIFO (Unix) — rejected;
  hardlinks (legal) accepted (a hardlink IS a regular file).
- **Byte cap:** file 1 byte over → rejected; exact cap → accepted.
- **Hash mismatch:** tamper one payload byte → `ModelHashMismatch`;
  tamper spec → `SchemaHashMismatch`; both verified before nonfinite scan.
- **Nonfinite scan:** NaN / +Inf / −Inf in any tensor → rejected with tensor
  name; **subnormal kept** (asserted); scan-off → NaN passes (policy is
  explicit).
- **Canonical round-trip:** load → re-serialize tensors in manifest order →
  byte-identical data section (bit-stable ordering proof).
- **Manifest agreement:** `manifest.entries` ≡ `expected_tensors()` (name,
  shape, order) — property test over generated legal specs.
- **Feature matrix:** default / digest / hub / hub-net / verify / all —
  the S1–S4 combos plus `digest` standalone (hub-on-digest-off is now
  impossible; assert the feature graph in a doc test).
- **Goldens:** one committed tiny checkpoint fixture (~KB, synthetic
  weights) exercising the full policy-on path.

## 10. Gate matrix (outputs piped, all must be green)

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
for FEAT in "" verify digest hub hub-net all-features-default-set; do …
cargo test -p katakuri --features "$FEAT"
cargo doc --all-features --no-deps        # zero unresolved links
cargo check --target wasm32-unknown-unknown
```

Plus the PR body checklist: dependency ledger comments (sha2 pin reason),
book page updated, CHANGELOG `[0.2.0] — Unreleased` entry.

## 11. Review protocol

ia-moment path: deepseek dispatch on this contract → in-session verify →
review moments to convergence (zero P1/P2, P3-only converges; 12-round
breaker; settled-stays-settled; findings matched by content/thread, not
commit_id). Operator merges. The Borsalino T0 co-sign review runs in
parallel on their session; `TensorManifest` may not ship frozen in 0.2.0
ahead of it (if their review lags, the type lands `#[doc(unstable)]` —
naming the co-sign as the gate rather than pretending it happened).
