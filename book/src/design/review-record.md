# Review Record

Every slice (S1–S4) ran a review loop before landing, capped at twelve rounds,
with findings posted inline on the PR. **Twenty verified findings** were found
and fixed across those four loops; this page is the record, because a review
that only leaves "LGTM" behind leaves nothing to learn from.

Three notes on what "verified" means here:

- Every entry below was reproduced by a regression test (or, in one case, by a
  behavioral repro that cannot be hermetic — noted inline).
- Severity labels are the reviewer's where the review record states one
  (**P1** correctness/security, **P2** contract divergence, **P3** hygiene).
  An entry labelled only `(round N)` is one the record did not grade.
- Not every finding was in shipped behaviour: finding 8 was in the *test
  harness*. That counts — a golden reader that silently truncates can make a
  parity proof lie.

## S1 — safetensors

| # | Class | Finding | Fix |
|---|---|---|---|
| 1 | Empty-span overlap (P2) | Zero-length tensors inside another tensor's span were falsely rejected as overlapping | Filter zero-length spans before the overlap check; empty tensors claim no bytes |
| 2 | Doc-link breakage (P2) | `verify` module docs failed `RUSTDOCFLAGS="-D warnings"`: module-level `//!` links resolve at crate-root scope | Explicit `crate::…` paths; the doc gate now joins the matrix in **both** feature configs |

Deferred to upstream: karpal-verify 0.6.1's SMT exporter has no operator
mapping for the terms Katakuri's obligations emit (its own `gpu.rs` uses the
same `Term::app` convention). Recorded, dispositioned on the PR, not fixed
here.

## S2 — tokenizer

| # | Class | Finding | Fix |
|---|---|---|---|
| 3 | Per-character lowercase (P1) | `str::to_lowercase` applies the contextual final sigma rule, so `ΟΣ` became `ος` where HF produces `οσ` | `chars().flat_map(char::to_lowercase)`; regression pins `ΟΔΟΣ → οδοσ` |
| 4 | Infinite loop (P1) | An empty added-token entry matched at every position, so the split never advanced | Reject empty added tokens at construction; defensive skip in the scan |
| 5 | Unsupported flags accepted (P2) | `single_word` / `lstrip` / `rstrip` boundary flags were parsed and ignored | Reject with `InvalidTokenizerJson` — silently ignoring them changes ids |
| 6 | Truncation direction (P2) | `direction: "Left"` was accepted and then silently right-truncated | `UnsupportedTruncationDirection` |
| 7 | Special-token mapping divergence (P2) | A `post_processor.special_tokens` map disagreeing with the vocab silently changed emitted ids | Validate every mapping when present: singleton expansion, id/token agreement with the vocab |
| 8 | Unchecked conversion (P3) | The golden reader used an unchecked cast, so an out-of-range golden would truncate instead of failing | `u32::try_from`, fail closed |
| 9 | Malformed maps accepted (round 2) | A present-but-malformed `special_tokens` map (`null`, wrong shape) fell through to a default | Field-presence match: absent → vocab lookup (documented), present → fully validated |
| 10 | Id-width divergence (round 3) | Ids above `u32::MAX` were accepted, but HF's `Vec<u32>` deserialization rejects those bytes | Require every id to fit `u32` (`4294967296` regression) |

## S3 — model specs

| # | Class | Finding | Fix |
|---|---|---|---|
| 11 | Capacity vs cardinality (round 1) | The tokenizer/spec check compared vocabulary *counts*; a sparse id (entry 40000 in a 30522-entry table) passed while encoding could index past the embedding table | Check `max_token_id() + 1 ≤ vocab_size` (max over vocab, added-token ids, specials); add `Tokenizer::max_token_id`; accept the legal vocab-padded class |
| 12 | wasm32 wrap (round 2) | `max_id as usize + 1` wraps to 0 at `u32::MAX` on 32-bit, admitting an id needing `2^32` rows | Compare `max_id >= vocab_size` — target-independent, algebraically identical on 64-bit |
| 13 | 32-bit tensor-count overflow (round 3) | `5 + 16·L + 2` overflows `usize` near `u32::MAX` layers: a parse-valid spec trapped in `expected_tensors()` on wasm32 | Bound `num_hidden_layers ≤ 1024` at parse time (>10× the largest known BERT-family encoder) |

## S4 — hub

| # | Class | Finding | Fix |
|---|---|---|---|
| 14 | Namespace shadowing (P1) | Content and digest sidecars shared a namespace, so a fetched file named `x.sha256` could shadow a sidecar | Separate `content/`, `digests/`, and temp namespaces; sidecars named exactly the filename |
| 15 | Case-fold aliasing (P1) | On case-insensitive filesystems, revisions `V1`/`v1` or files `Config.json`/`config.json` aliased to one cache entry | Case-safe encoding (`X → ^x`) on every case-bearing component; `^` is outside the grammar, so the map is injective |
| 16 | Secret in `Debug` (P1) | `UreqFetcher`'s derived `Debug` printed the bearer token | Manual impl: `Some(<redacted>)` / `None` only |
| 17 | Status laxity (P2) | Any 2xx (including 204/304) was accepted as file content to be cached | `require_final_status`: exactly 200 after redirects |
| 18 | Scheme laxity (P2) | A non-HTTPS initial URL was accepted, so a bearer could transit plaintext | `require_https` before building the request |
| 19 | Feature-unsafe doc links (P2) | Docs linking hub items broke under `--features hub` vs `hub-net` combinations | Feature-safe link targets; `--features hub` doc gate added to the matrix |
| 20 | Redirect downgrade (round 2) | The HTTPS check gated only the initial URL: an `https → http` redirect mid-chain could leak the bearer and accept plaintext bytes | `https_only(true)` on the agent itself, so ureq refuses the plaintext hop during redirect resolution. Behavioral repro re-run in round 3 (needs a live server; not hermetically testable) |

## The classes, distilled

Seven distinct failure classes account for all twenty findings. They are worth
naming because each is a *recurring* way strict data-plane code goes wrong:

1. **Unicode case folding is context-sensitive where the reference is not.**
   Parity surfaces must transcribe the reference's exact per-character
   behavior, not Rust's idiomatic equivalent (finding 3).
2. **A matcher that can match zero-length input loops forever.** Anyone
   scanning for literal tokens must reject empty needles at construction *and*
   guard the scan (finding 4).
3. **Cardinality is not capacity.** "How many entries" and "how large an index
   can this emit" diverge exactly when ids are sparse — and sparse ids are
   legal input (finding 11).
4. **32-bit arithmetic is where "obviously correct" checks fail.** `+ 1` after
   a `u32` at its maximum, and `16·L` at a large `L`, are both fine on 64-bit
   and broken on wasm32 (findings 12, 13). Compile the target, don't reason
   about it.
5. **Shared namespaces create aliasing.** Content vs sidecar, uppercase vs
   lowercase — any two things a filesystem might fold together must be
   separated structurally, not by convention (findings 14, 15).
6. **Ignored configuration is silent divergence.** Accepting a flag you do not
   honour (`lstrip`, `Left`) changes output without an error; reject instead
   (findings 5, 6, 7).
7. **A security check on the first request is not a check on the chain.**
   Redirects are part of the transport; enforcing a scheme at the entry point
   and letting the client follow a downgrade is a leak (findings 17, 18, 20).

## Why this is published

The review loops were not a formality: several findings were P1s in security
(namespace shadowing, token redaction, redirect downgrade) or correctness
(final sigma, sparse ids, wasm32 wraps). Publishing them makes the *classes*
reusable across the ecosystem rather than relearning them one crate at a time
— and it keeps an honest record of what `0.1.0` survived before it shipped.

The remaining weaknesses that review did **not** close are recorded in
`docs/critique.md`.
