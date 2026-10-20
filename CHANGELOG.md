# Changelog

All notable changes to Katakuri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — Unreleased

Additive minor bump: MLP architecture specs (wishlist W1), validated
checkpoint loading (W2), and the `digest` feature re-homing (W5), plus the
Borsalino T0 layout-manifest proposal. Undated until published; `0.1.0`
stays as-is.

### Added — S5: MLP specs + validated checkpoints (2026-10-19)

- `model`: `MlpSpec`, `MlpLayerSpec`, `Activation` — a strict `ia-mlp`
  feedforward family beside `BertSpec` (separate families, shared
  invariants; `BertSpec` is not generalized). Grammar: `model_type:
  "ia-mlp"`, `schema_version: 1`, `input_width 1..=4096`, `0..=8` hidden
  layers of width `1..=16384`, scalar `head: {width: 1}`, and named
  activations (`relu`, `tanh`, `gelu_erf`, `gelu_tanh` — approximations
  named, never inherited). `expected_tensors()` returns the canonical
  depth-first order (W1).
- `checkpoint` (`std` + `digest`): `load_validated` + `TrustPolicy` carrying
  Amari Task 43's trust policy — regular-file check (`symlink_metadata` +
  `O_NOFOLLOW` + `fstat`; hardlinks accepted), byte cap, `ia.spec`
  (`MlpSpec`) parsing, shape/schema agreement against `expected_tensors()`,
  schema/model hash verification, and an optional nonfinite scan. Ordered,
  failure-total, hashes before values (W2).
- Provenance convention: `ia.schema-hash`, `ia.model-hash`, `ia.spec`,
  `ia.provenance.training-data` (`1..=4096` bytes); unknown `ia.*` keys
  reject. `MlpSpec::canonical_json()` is the deterministic serialization the
  schema hash covers (W2 bonus).
- `manifest`: `TensorManifest`, `ManifestEntry`, `ManifestDtype`,
  `ManifestLayout`, `DeterminismPolicy`, `FastPathPolicy` — the Katakuri side
  of the Borsalino T0 co-sign (typed shape/offset/dtype layout, F32 pinned,
  row-major contiguous, fast paths forbidden). Pure data: no `zunesha`
  dependency. **Unstable until the Borsalino tensor-extension session
  reviews it** (the gate is named in the module docs) (T0).
- `digest` (default-off): `artifact_sha256`, plus the S4-compatible
  `sha256_hex` alias. Moved out of `hub`; `hub` depends on `digest` and
  re-exports both, so no hub API breaks (W5).
- Tests: grammar adversarials and parse-time bounds (including the
  no-32-bit-wrap boundary at maximum legal sizes), symlink/FIFO/hardlink
  fixtures, byte-cap boundaries, both hash-tamper classes, nonfinite
  rejection including subnormal-kept and scan-off NaN, canonical byte-
  identical round-trip, a manifest ≡ `expected_tensors()` property over
  generated legal specs, and one committed tiny synthetic checkpoint
  (`tests/fixtures/tiny-mlp.safetensors`) exercising the full policy-on
  path. Book page: `guide/mlp-specs.md`.

### Changed

- `sha2` pinned to exactly `0.11.0` (`0.11.1` requires Rust 1.88; MSRV is
  1.85). Feature graph: `digest = ["dep:sha2"]`, `hub = ["dep:sha2",
  "digest"]` — hub-on-digest-off is now unrepresentable and asserted at
  compile time.
- The metadata layer (specs, manifest) builds with no `digest`, no `hub`, and
  no `checkpoint` — the W6 feature map is documented in the crate docs.

### Fixed — S5 review round 1

- **P1 (twice-failed class):** per-ABI `O_NOFOLLOW` is now taken from `libc`
  (unix-only target dep) — hand-derived values were wrong twice (all-BSD
  `0x100`; then x86-64's value on aarch64 Linux, whose flag is `0x8000`).
- **P1:** the trust-policy open now includes `O_NONBLOCK`, so a FIFO swapped
  in between the metadata precheck and the open returns immediately for the
  descriptor check instead of blocking indefinitely on the read end.
- **P2:** `MlpSpec::canonical_json` is a hand-rolled fixed-key-order writer —
  iterating `serde_json::Map` made `ia.schema-hash` depend on the map backing
  (any downstream `preserve_order` enablement silently changed the digest);
  the format is pinned byte-for-byte by a unit test.
- **P2:** the round-trip guarantee is now stated and enforced over the
  canonical form: `ValidatedCheckpoint::canonical_data_bytes()` concatenates
  tensor data in canonical parameter order, layout-independent; a
  reversed-physical-layout fixture test proves equal canonical bytes over
  different file bytes.

### Fixed — S5 review round 2

- **P2:** `libc` is now an optional unix-only dependency activated via
  `digest` (the feature that compiles the checkpoint module) — default and
  no-default Unix dependency graphs carry no checkpoint-only dependency
  (verified with `cargo tree` in all four configurations; wasm32 unaffected).

## [0.1.0] — 2026-10-17

Undated on purpose: `cargo publish` and the `v0.1.0` tag make it real, and the
operator's tag push finalizes it. Everything below landed between 2026-10-16
and 2026-10-17 as slices S1–S4 plus release polish.

### Added — S4: hub (2026-10-17)

- `hub`: optional fetch + content-addressed cache (offline-first,
  digest-verified reads); `hub-net` adds ureq/rustls transport.
- Fixed (review round 1): cache layout separates content/digests/tmp
  namespaces (sidecar-shadowing), case-safe path encoding
  (case-insensitive filesystems), redacted Debug for bearer tokens,
  strict final-200 status check, HTTPS-only transport for token requests,
  feature-safe doc links.
- Fixed (review round 2): the ureq agent is `https_only` across the whole
  redirect chain, so an https→http downgrade cannot leak the bearer.

### Added — S3: model specs (2026-10-17)

- `model`: typed BERT `BertSpec` (strict config parsing), weight-layout
  manifests validated against parsed safetensors, tokenizer
  compatibility check, and the Ijima swap-boundary port plan.
- Fixed (review round 1): `check_tokenizer_compatibility` checks token-ID
  capacity (max emittable id + 1 ≤ vocab_size) instead of cardinality —
  sparse vocab ids can no longer overflow the embedding table; vocab-padded
  (smaller-than-table) tokenizers are accepted as legal.
- Fixed (review round 2): the capacity check uses `>=`, immune to 32-bit
  (wasm32) wrap at `u32::MAX`.
- Fixed (review round 3): `num_hidden_layers` bounded at 1024 at parse time —
  `5 + 16·L + 2` overflowed 32-bit `usize` near `u32::MAX`.

### Added — S2: WordPiece tokenizer (2026-10-16)

- `katakuri::tokenizer`: bespoke BERT-class WordPiece engine, replacing
  the `tokenizers` crate dependency for Ijima's embedding surface.
  Pipeline transcribed from tokenizers-0.21.4: raw added-token split
  (leftmost, longest-first), BertNormalizer (clean_text → CJK spacing →
  NFD accent strip following `lowercase` → lowercase), BertPreTokenizer
  (whitespace split + punctuation isolation), greedy longest-match
  WordPiece with `##` continuations, LongestFirst truncation reserving
  the two special slots, `[CLS] … [SEP]` wrapping, BatchLongest
  right-padding.
- Constructors: `from_tokenizer_json` (strict BERT-class subset —
  unsupported kinds reject with typed errors, never silent divergence)
  and `from_bert_vocab_txt` (BERT defaults).
- Golden-vector parity: 26 singles + 3 batch goldens generated from the
  real MiniLM tokenizer with Ijima's exact call shape
  (`tests/fixtures/`, tokenizers 0.21.4); vocab fixture committed so CI
  replays parity without network.
- Dependencies: `unicode-normalization 0.1` (NFD) and
  `unicode_categories 0.1` (is_other/is_mark_nonspacing/is_punctuation)
  — the reference implementation's own Unicode tables.
- Strict-construction hardening (review round 1): per-character
  lowercase (no contextual final sigma — parity with the reference);
  empty added tokens, boundary-flag added tokens (single_word/lstrip/
  rstrip), left-truncation direction, and divergent template
  special_tokens mappings all reject with typed errors; golden reader
  uses checked u32 conversion.
- Fixed (review round 2): present-but-malformed `special_tokens` maps
  (`null`, non-object, wrong-shaped entries) reject instead of falling
  through to defaults.
- Fixed (review round 3): `special_tokens` ids are bounded to `u32`,
  matching HF deserialization.

### Added — S1: safetensors parse/validate (2026-10-16)

- `katakuri::safetensors`: `SafeTensors::parse` — zero-copy, fully
  validated manifest over borrowed bytes. Validation contract: 100 MB
  header cap, strict tensor-entry fields, dtype/shape/offset checks,
  span == shape×dtype (checked u64 arithmetic), no-overlap with gaps
  allowed. `Tensor` views (`dtype`/`shape`/`data_offsets`/`bytes`/
  `elem_count`), `__metadata__` extraction.
- Device seam: `SafeTensors::upload_to_device` over `zunesha::Device`
  (`create_device_buffer`, GPU-resident weights). NoDevice path is
  first-class: parsing needs no device; upload failure maps to
  `KatakuriError::Device(DeviceError::NoDevice)` with host views intact.
- Borsalino verification pattern: `verify` feature with karpal-verify
  obligation bundles (offset-in-bounds, span-consistent,
  manifest-non-overlapping; tier `Impossible`), plus `#[cfg(kani)]`
  proof harnesses for the validation arithmetic.
- MiniLM-shaped round-trip test (103-tensor BERT manifest, byte-exact),
  rejection tests for every validation rule, env-gated real-file
  round-trip (`KATAKURI_TEST_SAFETENSORS`).
- Fixed (review round 1): zero-length spans are excluded from the
  overlap check — empty tensors claim no bytes and can no longer be
  falsely rejected inside another tensor's span.
- Fixed (review round 2): `verify` module doc links resolve explicitly
  under `RUSTDOCFLAGS="-D warnings"`.

### Added — repo scaffold (2026-10-16)

- Library crate skeleton (`katakuri`, lib target; replaces the scaffold's
  binary stub) per IA conventions: Apache-2.0 + SPDX headers on every source
  file, pinned stable toolchain (`rust-toolchain.toml`), central
  `KatakuriError` (thiserror), and module stubs for the four planned
  surfaces — `safetensors` (S1), `tokenizer` (S2), `model` (S3), `hub` (S4).
- Licensing and process stack: Apache-2.0 `LICENSE`, `CONTRIBUTING.md` with
  CLA reference, Forgejo mirror under `industrial-algebra` with dual-push
  wiring.

### Added — release polish (2026-10-17)

- CI gate matrix: test across feature combinations, clippy on default and
  all-features, fmt, rustdoc with `-D warnings` per feature, example
  compile + run, wasm32 build, publish on `v*` tags, book deploy.
- mdBook (`book/`) with the IA theme: introduction, getting started,
  concepts, feature flags, hub guide, per-module API pages, verification,
  and the review record.
- Three runnable examples: `inspect-safetensors`, `encode-text`,
  `fetch-and-validate` (with `--offline`).
- `docs/guide/feature-flags.md`, `docs/ROADMAP.md`, `docs/critique.md`, and
  a rewritten README with status, quick start, and the IS / IS-NOT boundary.

### Dependencies

- `serde_json 1` — the safetensors header is JSON; the ecosystem-standard
  parser (serde/itoa/ryu/memchr transitives only).
- `zunesha 0.1.2` — the device substrate (Buffer upload, NoDevice path).
- `karpal-verify`/`karpal-proof 0.6.1` — verification obligations, behind
  the `verify` feature (pinned to Borsalino's pin).
- `unicode-normalization 0.1`, `unicode_categories 0.1` — the tokenizer
  reference implementation's own Unicode tables (S2).
- `sha2 0.11` (feature `hub`) — sha256 for the content-addressed cache (S4).
- `ureq 3.4` (feature `hub-net`) — blocking HTTPS with rustls, redirects,
  no async runtime (S4).

### Security

- Hub cache poisoning is structurally blocked: content/digest/temp namespace
  separation, sidecars named exactly the filename, digest verification on
  every read, case-safe path encoding, and a strict path grammar that rejects
  `..`/`/`/NUL at construction.
- Transport never sends a bearer token over plaintext: HTTPS check before the
  request and `https_only` across the redirect chain; tokens are redacted in
  `Debug`.
- The safetensors header is capped at 100 MB before parsing (DoS guard), and
  all span arithmetic is checked `u64`.
