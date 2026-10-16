# Changelog

All notable changes to Katakuri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Added — repo scaffold (2026-10-16)

- Library crate skeleton (`katakuri`, lib target; replaces the scaffold's
  binary stub) per IA conventions: Apache-2.0 + SPDX headers on every source
  file, pinned stable toolchain (`rust-toolchain.toml`), central
  `KatakuriError` (thiserror), and module stubs for the four planned
  surfaces — `safetensors` (S1), `tokenizer` (S2), `model` (S3), `hub` (S4).
- Licensing and process stack: Apache-2.0 `LICENSE`, `CONTRIBUTING.md` with
  CLA reference, Forgejo mirror under `industrial-algebra` with dual-push
  wiring.

### Dependencies (S1)

- `serde_json 1` — the safetensors header is JSON; the ecosystem-standard
  parser (serde/itoa/ryu/memchr transitives only).
- `zunesha 0.1.2` — the device substrate (Buffer upload, NoDevice path).
- `karpal-verify`/`karpal-proof 0.6.1` — verification obligations, behind
  the `verify` feature (pinned to Borsalino's pin).
