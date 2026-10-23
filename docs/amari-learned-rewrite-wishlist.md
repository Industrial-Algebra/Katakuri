# Wish list: Amari learned rewrite guidance

Date: 2026-10-16
Status: input for Katakuri 0.1.x planning
Source: Amari 0.25.0 inverse-expansion plan, Tasks 40–45
(`amari-rewrite` neural guidance). Amari's `neural` feature is
trait-only today; the candle dependency was removed after the
zip 8.6.0 yank and Katakuri (over Zunesha) is the designated
backend.

## The shape of the need

Amari's learned-rewrite work is **candidate ranking for inverse
rewrite search**, not language modeling: score symbolic predecessor
candidates, order them, optionally beam-prune under an explicit
authority discipline. The moral equivalent of a tiny scoring MLP
over fixed-width feature vectors — no tokenizer, no hub-scale
models, no GPU, no floating-point nondeterminism tolerated.

Amari keeps: the search, the authority discipline, the symbolic
score fallback, the replay/rejection semantics, and all metering.
The backend (Katakuri + Zunesha) should own: weights, architecture
description, checkpoint I/O, and the tensor compute the scoring
needs.

## Wishes, in priority order

### W1 — Small custom feedforward architecture specs (must-have)

Katakuri 0.1.0 architecture specs target named model families
(BERT-shaped). Amari needs to describe a **small MLP ranker**
(input width ~64, one hidden layer ~256, scalar score) as a
first-class architecture spec: layer list, activations, shapes —
not a hub model name. If the spec format already generalizes to
"arbitrary feedforward stack", this is documentation, not code.

### W2 — Validated checkpoint loading with Amari's trust policy (must-have)

Amari's Task 43 (safe-tensor checkpoints) wants, at load time:
regular-file validation (symlink rejection), byte caps, declared
metadata (schema hash, model hash), nonfinite-value rejection,
shape/schema mismatch rejection, and **canonical parameter
ordering** (score round trips must be bit-stable). Katakuri owns
safetensors parsing; the wish is a `load_validated`-style API
carrying these policies, so Amari does not re-implement them over
raw bytes. Bonus: a metadata sidecar convention for **training-data
provenance hashes** (Amari binds guidance to replay hashes).

### W3 — Deterministic CPU inference for scalar scoring (must-have, via Zunesha)

Scoring must be deterministic on CPU: fixed input width, fixed
batch canonicalization, defined behavior on nonfinite inputs
(reject, never propagate NaN into an ordering decision). Katakuri
never computes, so this is a Zunesha capability — the Katakuri-side
wish is that the architecture spec carries whatever determinism
metadata the compute layer needs (dtype pinning, no
backend-selected fast paths).

### W4 — Pairwise-ranking training story (should-have)

Amari's Task 42 trains with bounded AdamW over pairwise margin
loss, seeded deterministically, with deadline checks at epoch
boundaries and nonfinite-gradient rejection. Open seam to decide
together: does autodiff/optimization live in Zunesha (Katakuri
persists optimizer state — AdamW moments — alongside model weights
in safetensors), or does Amari hand-roll the tiny optimizer over
Zunesha tensors? The wish: Katakuri defines the **optimizer-state
checkpoint convention** either way, so training resumes are
portable and auditable.

### W5 — Deterministic artifact identity (should-have)

Every artifact Katakuri loads or stores should have a canonical
content hash Amari can bind into its guidance/replay hashes
(checkpoint digest, schema digest). If hub access already provides
content addressing (S4 hub: sha2), expose it for local files too.

### W6 — no_std-considerate layering (nice-to-have)

Amari's rewrite core is no_std + alloc; the neural seam is
std-gated today and that is acceptable. The wish is only that
Katakuri's weight/spec parsing not force std into consumers that
only want metadata (feature-split metadata from I/O, as the
`hub`/`hub-net` split already models).

## Explicit non-wishes

- No tokenizer needs (Amari encodes terms structurally).
- No hub-scale model hosting needs in 0.25.0.
- No GPU/backend selection: CPU-only, deterministic.
- No training-data pipeline needs: Amari derives pairwise examples
  from its own replay-validated traces (Task 41).

## What Amari brings to the integration

- The `neural` trait seam (`DifferentiableRule`, guidance scoring
  hook) already isolated behind a feature gate.
- The authority contract the backend must not weaken: ordering-only
  mode never drops candidates; beam mode reports dropped counts and
  can never claim exhaustion; every learned-guidance witness still
  replays against the symbolic system.
- Differential/property tests the integration must pass (Tasks
  40–44 RED lists), which can serve as Katakuri-side acceptance
  fixtures for the seams above.

## Decision point for Amari 0.25.0

When the plan reaches Tasks 40–45, reassess: if Katakuri covers
W1–W3 (and ideally W4), learned rewrite lands in 0.25.0 on
Katakuri; otherwise those tasks defer to 0.26.0 with the symbolic
guidance stack (already merged) carrying the release.
