# Katakuri — Architecture Handover

**Date:** 2026-10-16
**From:** Amari core-matrix session (M-REP-C loop)
**Status:** repo skeleton exists (uncommitted); no implementation code yet
**Operator directive:** break out the non-tensor-math functionality that candle
currently provides to Minuet and Ijima into this repo, built over Zunesha.
Amari's rewrite session will defer on candle entirely — adopting Katakuri when
ready, or waiting until it is.

---

## 1. Why this repo exists (provenance — read this first)

On 2026-10-16, `zip 8.6.0` was yanked from crates.io. The entire 8.x line
(8.5.0, 8.5.1, 8.6.0) is yanked while zip iterates the 9.x line daily
(9.0.0–9.0.4 within a week, three of those yanked). The pattern reads as
deliberate retirement of 8.x.

`candle-core 0.11.0` requires `zip ^8.6.0`, whose only satisfiable version is
now yanked — **permanently unsatisfiable** unless zip un-yanks or backfills.
Consequences observed:

- Amari CI: 15/16 workflow jobs failed at dependency *resolution* (2–23s each;
  nothing compiled). Amari's candle usage turned out to be **zero** — a
  pre-declared optional dep from PR #242 ("rewrite research dependencies")
  forced into resolution by `--all-features` CI steps. Fix: remove it
  (separate PR, 2026-10-16).
- A fix PR exists upstream (huggingface/candle #4041, zip-9 compatibility,
  CPU-verified) but candle's release cadence is quarterly (0.9.2 Jan → 0.10.x
  Mar/Apr → 0.11.0 Jun 26 → nothing since) and there was zero maintainer
  response in the first ~12h.
- The operator's conclusion: this incident is precisely why heavy ML-framework
  dependencies were hesitant adoptions in the first place. The strategic
  response is an IA-owned ecosystem layer, not a bespoke candle replacement.

**Doctrine confirmed by the operator during this discussion:**

- **Borsalino is the workhorse; Amari is the precision.** f32 GPU compute and
  exact/verified CPU math are separate layers, not competitors.
- Katakuri is the **model-ecosystem layer** — everything around tensor math.
- Ijima runs on **GPU-less hardware**: the NoDevice/host-memory path is a
  first-class requirement, not a fallback.

## 2. Naming and neighbors

Shandora workspace (all One Piece lineage):

| Crate | Role |
|---|---|
| **Zunesha** | Shared GPU device substrate — device, queues, memory, `zunesha::Buffer`. Vulkan complete, Metal planned. **No shader compilation** (that's Borsalino/Goldenweek). Has `NoDevice` stub. v0.1.2 on crates.io. |
| **Borsalino** | Thin GPU compute abstraction (GpuBackend trait, 7 methods, raw-FFI Metal, zero-dep philosophy). Compiles shaders; dispatches. v0.6.0 on crates.io. |
| **Goldenweek** | Graphics (render) layer over Zunesha. |
| **Katakuri** | **This repo.** Model ecosystem: weights, tokenization, model specs, hub access. Over Zunesha. Never computes. |

## 3. Measured consumer surface (the ground truth for scope)

`Anima/Ijima/ijima-server/src/embeddings_candle.rs` — 287 lines, the entire
candle surface in Ijima, behind non-default feature `embeddings-candle`
(candle-core/nn/transformers 0.11, default-features = false, CPU-only):

Workload: tokenize → BERT forward → attention-masked mean-pool → L2-normalize
→ cosine similarity (384-dim MiniLM-class embeddings).

Symbol census from that file:

| Surface element | Count | Katakuri module that owns it |
|---|---|---|
| `tokenizer` references | 13 | `tokenizer` |
| safetensors loading | 4 | `safetensors` |
| `BertModel` (architecture) | 4 | `model` (as *spec*, not code) |
| `VarBuilder` | 3 | `safetensors` |
| `Tensor::new` | 3 | — stays in consumer (compute) |
| `hf_hub` | 1 | `hub` |
| `Device::Cpu` | 1 | Zunesha NoDevice path |

Amari measurement (for contrast): candle declared 2026-06 (#242), **zero
imports in any crate ever** — removed 2026-10-16. Sakuya does not depend on
candle (operator-verified). Quantizon/possum carry *heavier* use (model
loaders, planned MDLM **training**) — explicitly **out of initial scope**;
Katakuri does not attempt autodiff, optimizers, or training.

## 4. Architecture

### 4.1 The hard boundary

**Katakuri stops precisely where candle starts doing math.** No forward
implementations, no autodiff, no kernels, no shader compilation. Consumers
implement forward passes over their tensor engine of choice (CPU today;
Borsalino-class compute later — weights already live in Zunesha buffers,
zero-copy ready).

This is the structural form of the operator's workhorse/precision separation:
the same weights feed a precision CPU path or a Borsalino GPU path, and
Katakuri itself takes no side.

### 4.2 Layering over Zunesha

- Weight data lands in `zunesha::Buffer` when a device exists; host memory
  via `NoDevice` otherwise (Ijima's hardware).
- Katakuri is a *consumer* of Zunesha, never a wrapper of it — buffers flow
  through, alignment/lifecycle obligations are Zunesha's (its `verify`
  feature bundles karpal-verify obligations for buffer alignment/quiescence;
  Katakuri should inherit, not restate).

### 4.3 Module decomposition

| Module | Owns | Replaces (candle surface) |
|---|---|---|
| `safetensors` | Weight parse + validate (dtype/shape/layout/manifest) → Zunesha buffer or host memory. No math. | VarBuilder + safetensors glue |
| `tokenizer` | WordPiece (BERT-class) tokenization | the `tokenizers` crate |
| `hub` | Model fetch + content-addressed cache over plain HTTPS | `hf-hub` |
| `model` | Architecture **specs as typed data**: BERT config (layers/heads/dims/hidden act), named-tensor layout manifests | candle-transformers `models::bert` |

### 4.4 The one open dependency decision: the tokenizer

Wrap HF's `tokenizers` crate (pure Rust, self-contained tree, but the same
adoption-risk pattern that motivated this repo) vs. **bespoke WordPiece**
covering BERT-class models actually used (small, fully auditable, IA-owned).

Session recommendation, pending operator confirmation: **bespoke**. The
incident that created this repo argues for owning the surface; WordPiece for
the used model family is small. If exotic tokenizers appear later, the `hub`
layer can carry converter tooling rather than runtime deps.

## 5. Slice plan (bounded, reviewable contracts)

- **S1 — scaffold + safetensors.** Fix the skeleton (crate name lowercase
  `katakuri`, lib target not bin, Apache-2.0 + SPDX headers, ia-rust-stack
  conventions, pinned toolchain, workspace inheritance if Shandora becomes a
  workspace). Implement safetensors parse/validate → `zunesha::Buffer` +
  NoDevice host path. Acceptance: round-trip a real MiniLM safetensors file;
  dtype/shape validation rejects corrupted tensors; Ijima's weight-loading
  half is expressible.
- **S2 — tokenizer.** WordPiece matching Ijima's exact tokenization contract
  (special tokens, padding, truncation semantics used by `embeddings_candle.rs`).
  Acceptance: byte-identical token id sequences vs. HF tokenizers on a fixed
  test corpus (golden vectors committed).
- **S3 — model specs + swap boundary.** Typed BERT config + weight-layout
  manifests; a port plan for `embeddings_candle.rs` → Katakuri-backed
  (consumer keeps forward until/if it moves). candle stays behind Ijima's
  non-default feature throughout.
- **S4 — hub.** Optional feature; plain HTTPS fetch, content-addressed cache,
  offline-first.

## 6. Consumers and integration notes

- **Ijima (first consumer).** `embeddings-candle` remains non-default; add
  `embeddings-katakuri` alongside; swap when S1–S3 cover the file's needs.
  Blast-radius containment (non-default, single file, trait boundary) is the
  pattern that kept Ijima's default builds safe during the yank — keep it.
- **amari-rewrite (deferred adopter).** The `neural` feature is trait-only
  (21 lines) and stays that way. Operator directive 2026-10-16: adopt
  Katakuri if ready in time, otherwise defer until it is. Never re-introduce
  a pre-declared tensor-framework dep.
- **Minuet.** Operator-identified consumer; surface not measured in this
  session (repo not on this machine). Measure before promising scope.
- **Quantizon/possum.** Training-class use — out of scope for now; their
  candle stays until an IA autodiff story exists (a future, larger decision).

## 7. Conventions (non-negotiable)

- ia-rust-stack ten-point checklist; ia-coding-standards; ia-licensing
  (Apache-2.0 + CLA).
- Gitflow per skill `ia-gitflow` (feature/* → develop; this repo's protected
  branch conventions TBD by operator on first PR).
- Review loops before production code (12-round cap, inline PR posting).
- Evidence hygiene: `$SHANDORA`/`$HOME` path scrub for committed artifacts.
- No silent dependency growth: every new dep gets a written justification in
  the PR; `--all-features` resolvability is a CI-visible property now.

## 8. Paths and artifacts (as of handover)

- Repo: `/home/lucien/working/industrial-algebra/Shandora/Katakuri/`
  (skeleton: `Cargo.toml` binary-crate stub, `src/main.rs` hello-world — both
  need replacement in S1; git initialized, history empty, all untracked).
- Zunesha local: `../Zunesha` (read its `lib.rs` docs before designing buffer
  integration).
- Measured consumer: `~/working/industrial-algebra/Anima/Ijima/ijima-server/src/embeddings_candle.rs`.
- Incident evidence: Amari PR #311 CI failures; upstream huggingface/candle
  PR #4041; zip version/yank table (§1).

## 9. Session context

Drafted in the Amari core-matrix session immediately after M-REP-C converged
(R4, PR #311) and the candle incident was diagnosed. The same session is
landing the Amari candle removal (zero-usage dep) in parallel. Katakuri work
continues in a fresh session per operator instruction.
