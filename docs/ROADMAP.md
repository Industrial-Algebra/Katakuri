# Katakuri Roadmap

> **Everything below "Speculative" is speculation.** It is recorded so the
> direction is visible, not promised. Nothing there is scoped, scheduled, or
> contracted; items move up only when a consumer needs them and a written
> contract exists (the `docs/plans/` pattern).

## Implemented — v0.2.0

- **S1 · safetensors** — zero-copy parse + full validation (dtype, shape,
  offsets, span arithmetic, non-overlap, 100 MB header cap), `__metadata__`
  extraction, and the `zunesha::Device` upload seam with the `NoDevice` host
  path first-class.
- **S2 · tokenizer** — bespoke BERT-class WordPiece engine replacing the
  `tokenizers` crate dependency: added-token split, `BertNormalizer`,
  `BertPreTokenizer`, greedy longest-match with `##` continuations,
  LongestFirst truncation, `[CLS] … [SEP]`, BatchLongest padding. Golden
  parity: 26 singles + 3 batch cases from the reference at 0.21.4.
- **S3 · model specs** — `BertSpec` (strict `config.json` subset), weight
  layout manifests validated against parsed safetensors, tokenizer/embedding
  capacity check, and the Ijima swap-boundary port plan.
- **S4 · hub** — `hub` (content-addressed, digest-verified, offline-first
  cache with a dependency-free `Fetch` seam) and `hub-net` (blocking
  ureq/rustls transport, HTTPS-only across redirects).
- **Ijima port** — the embedding path's three `hf-hub` downloads and candle
  surfaces replaced with Katakuri calls; tokenization bit-for-bit.
- **Release polish** — CI gate matrix (test / clippy / fmt / rustdoc /
  examples / wasm32), publish-on-tag, this mdBook, roadmap and critique.
- **S5 · MLP specs + validated checkpoints (0.2.0)** — the `ia-mlp`
  strict feedforward family with named activations, `TensorManifest` (the
  Borsalino T0 co-sign proposal, unstable until their session reviews),
  the `digest` feature re-homing, and `load_validated` under an explicit
  `TrustPolicy` (regular-file-only, byte cap, schema/model hashes, optional
  nonfinite scan; libc-backed trust open on Unix). Three review rounds,
  five findings, converged zero — recorded in the review record.

## Speculative

> Banner applies: the rest of this page is direction, not commitment.

- **Kani execution in CI.** The three `#[cfg(kani)]` harnesses exist but are
  not executed in the pipeline (Kani is not installed in CI); running them
  per-PR would turn the arithmetic proofs into a live gate.
- **Cross-repo-commit revision pinning.** The hub keys the cache by revision
  string as given, so `main` re-resolves. Recording the response's
  `x-repo-commit` header and verifying it stayed stable would close that gap
  (S4's recorded limitation).
- **More architectures — BERT-class only for now.** `BertSpec` covers BERT
  and its descendants (MiniLM-class configs; the strict parser accepts `model_type: "bert"` only). Other families
  would each need their own spec type; nothing generic is planned, because
  "generic model spec" is how loaders grow compute.
- **Minuet consumer measurement.** Minuet was identified as a consumer but its
  surface was never measured (repo not on the authoring machine). Measure
  before promising scope — the same rule that produced S1–S4.
- **Cache file locking.** Serialize concurrent writers sharing one cache root;
  unnecessary for today's single-process deployments, needed if a
  multi-process daemon ever shares `$KATAKURI_HUB_DIR`.
- **Windows reserved-name rejection.** The filename grammar admits `con`,
  `nul`, `aux`; rejecting the device-name set would close the last recorded
  path-safety gap.

## Explicitly out of scope

Not deferred — excluded by the architectural contract
(`docs/plans/2026-10-16-katakuri-architecture-handover.md` §4.1):

- forward implementations, autodiff, optimizers, training;
- kernels, shader compilation, GPU dispatch (that is Borsalino);
- numeric weight inspection;
- an async runtime;
- model metadata APIs and uploads.
