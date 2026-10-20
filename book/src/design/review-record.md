# Review Record

Every slice (S1–S4) ran a review loop before landing, capped at twelve rounds,
with findings posted inline on the PR. **Twenty-two root findings** across
those four loops; **twenty were fixed here with a regression** (severity
split: 9× P1, 10× P2, 1× P3, all as graded on the PR threads). This page is
the record, because a review that only leaves "LGTM" behind leaves nothing
to learn from.

Three notes on what is counted:

- **Two findings are excluded from the twenty** (both real, both dispositioned
  on the PR, both documented in the S1 section): a P3 that strengthened a weak
  test *assertion* — a test defect, not shipped-code behavior — and a P2
  deferred upstream (the karpal-verify SMT exporter).
- Every retained entry below was reproduced by a regression test (or, in one
  case, by a behavioral repro that cannot be hermetic — noted inline).
- Not every finding was in shipped behaviour: finding 8 was in the *test
  harness*. That counts — a golden reader that silently truncates can make a
  parity proof lie.

## S1 — safetensors

| # | Class | Finding | Fix |
|---|---|---|---|
| 1 | Empty-span overlap (P2) | Zero-length tensors inside another tensor's span were falsely rejected as overlapping | Filter zero-length spans before the overlap check; empty tensors claim no bytes |
| 2 | Doc-link breakage (P2) | `verify` module docs failed `RUSTDOCFLAGS="-D warnings"`: module-level `//!` links resolve at crate-root scope | Explicit `crate::…` paths; the doc gate now joins the matrix in **both** feature configs |

Excluded from the retained count (both real, both dispositioned on the PR):

- **(P3, test assertion)** the unknown-field rejection test's assertion was
  too weak to establish that the diagnostic names the offending field —
  adopted and strengthened, but a test defect rather than shipped code.
- **(P2, deferred upstream)** karpal-verify 0.6.1's SMT exporter has no
  operator mapping for the terms Katakuri's obligations emit (its own
  `gpu.rs` uses the same `Term::app` convention) —
  [Karpal#176](https://github.com/Industrial-Algebra/Karpal/issues/176);
  the 0.6.1 pin is retained pending the upstream fix.

## S2 — tokenizer

| # | Class | Finding | Fix |
|---|---|---|---|
| 3 | Per-character lowercase (P1) | `str::to_lowercase` applies the contextual final sigma rule, so `ΟΣ` became `ος` where HF produces `οσ` | `chars().flat_map(char::to_lowercase)`; regression pins `ΟΔΟΣ → οδοσ` |
| 4 | Infinite loop (P1) | An empty added-token entry matched at every position, so the split never advanced | Reject empty added tokens at construction; defensive skip in the scan |
| 5 | Unsupported flags accepted (P2) | `single_word` / `lstrip` / `rstrip` boundary flags were parsed and ignored | Reject with `InvalidTokenizerJson` — silently ignoring them changes ids |
| 6 | Truncation direction (P2) | `direction: "Left"` was accepted and then silently right-truncated | `UnsupportedTruncationDirection` |
| 7 | Special-token mapping divergence (P2) | A `post_processor.special_tokens` map disagreeing with the vocab silently changed emitted ids | Validate every mapping when present: singleton expansion, id/token agreement with the vocab |
| 8 | Unchecked conversion (P3) | The golden reader used an unchecked cast, so an out-of-range golden would truncate instead of failing | `u32::try_from`, fail closed |
| 9 | Malformed maps accepted (P2, round 2) | A present-but-malformed `special_tokens` map (`null`, wrong shape) fell through to a default | Field-presence match: absent → vocab lookup (documented), present → fully validated |
| 10 | Id-width divergence (P2, round 3) | Ids above `u32::MAX` were accepted, but HF's `Vec<u32>` deserialization rejects those bytes | Require every id to fit `u32` (`4294967296` regression) |

## S3 — model specs

| # | Class | Finding | Fix |
|---|---|---|---|
| 11 | Capacity vs cardinality (P1, round 1) | The tokenizer/spec check compared vocabulary *counts*; a sparse id (entry 40000 in a 30522-entry table) passed while encoding could index past the embedding table | Check `max_token_id() + 1 ≤ vocab_size` (max over vocab, added-token ids, specials); add `Tokenizer::max_token_id`; accept the legal vocab-padded class |
| 12 | wasm32 wrap (P1, round 2) | `max_id as usize + 1` wraps to 0 at `u32::MAX` on 32-bit, admitting an id needing `2^32` rows | Compare `max_id >= vocab_size` — target-independent, algebraically identical on 64-bit |
| 13 | 32-bit tensor-count overflow (P1, round 3) | `5 + 16·L + 2` overflows `usize` near `u32::MAX` layers: a parse-valid spec trapped in `expected_tensors()` on wasm32 | Bound `num_hidden_layers ≤ 1024` at parse time (>10× the largest known BERT-family encoder) |

## S4 — hub

| # | Class | Finding | Fix |
|---|---|---|---|
| 14 | Namespace shadowing (P1) | Content and digest sidecars shared a namespace, so a fetched file named `x.sha256` could shadow a sidecar | Separate `content/`, `digests/`, and temp namespaces; sidecars named exactly the filename |
| 15 | Case-fold aliasing (P1) | On case-insensitive filesystems, revisions `V1`/`v1` or files `Config.json`/`config.json` aliased to one cache entry | Case-safe encoding (`X → ^x`) on every case-bearing component; `^` is outside the grammar, so the map is injective |
| 16 | Secret in `Debug` (P1) | `UreqFetcher`'s derived `Debug` printed the bearer token | Manual impl: `Some(<redacted>)` / `None` only |
| 17 | Status laxity (P2) | Any 2xx (including 204/304) was accepted as file content to be cached | `require_final_status`: exactly 200 after redirects |
| 18 | Scheme laxity (P2) | A non-HTTPS initial URL was accepted, so a bearer could transit plaintext | `require_https` before building the request |
| 19 | Feature-unsafe doc links (P2) | Docs linking hub items broke under `--features hub` vs `hub-net` combinations | Feature-safe link targets; `--features hub` doc gate added to the matrix |
| 20 | Redirect downgrade (P1, round 2) | The HTTPS check gated only the initial URL: an `https → http` redirect mid-chain delivered attacker-controlled plaintext **content** that was accepted and cached (the round-2 repro ran without a bearer; combined with finding 18’s verified plaintext-bearer path on the initial URL, the downgrade risk follows) | `https_only(true)` on the agent itself, so ureq refuses the plaintext hop during redirect resolution. Behavioral repro re-run in round 3 (needs a live server; not hermetically testable) |

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

The review loops were not a formality: nine retained findings were P1s in
security (namespace shadowing, token redaction, redirect downgrade) or
correctness (final sigma, the empty-token loop, sparse ids, both wasm32
arithmetic traps). Publishing them makes the *classes*
reusable across the ecosystem rather than relearning them one crate at a time
— and it keeps an honest record of what `0.1.0` survived before it shipped.

The remaining weaknesses that review did **not** close are recorded in
`docs/critique.md`.

## S5 — MLP specs + validated checkpoints (PR #15)

Three rounds: **2×P1 + 2×P2 → 1×P2 → converged zero.** Five root findings,
all fixed:

| # | Sev | Round | Finding | Fix |
|---|---|---|---|---|
| 1 | P1 | r1 | Per-ABI `O_NOFOLLOW` hand table wrong twice — dispatch grouped all BSD-family under `0x100`; the in-session re-fix spread x86-64's value to aarch64 Linux (whose flag is `0x8000`) | Table deleted; `libc` (optional, unix-only, via `digest`) owns the flags |
| 2 | P1 | r1 | A FIFO swapped in between the metadata precheck and the open blocks the loader forever (read-end open waits for a writer) | Probe open is `O_NONBLOCK`; descriptor check rejects; hang-pinning mkfifo test |
| 3 | P2 | r1 | `ia.schema-hash` depended on `serde_json`'s map backing — downstream `preserve_order` enablement silently changed the digest (two fixture hashes reproduced) | Hand-rolled fixed-key-order `canonical_json`, no map iteration; byte-pinned by unit test |
| 4 | P2 | r1 | "Byte-identical round-trip" overstated — it held only for canonically-written files; reversed physical layouts loaded with no canonicalization witness | `canonical_data_bytes()` (layout-independent canonical form) + reversed-layout fixture test |
| 5 | P2 | r2 | `libc` sat in the default and no-default Unix dependency graphs though neither compiles the checkpoint module | `libc` optional, activated by `digest`; verified via `cargo tree` in all graphs |

**Classes worth reusing:**

1. **Hand-copied ABI constants are a twice-failed class.** Two independent
   hand-derivations of one flag table were both wrong, each in a different
   way, and both passed every local gate (the targets that would catch them
   weren't compiled). The dependency carrying the verified per-arch values is
   cheaper than the third attempt.
2. **Blocking syscalls are denial of service in a validator.** A trust-policy
   loader must bound its own wait: any open that can park (FIFO read end)
   needs the nonblocking flag even when the *expected* failure is later in
   the pipeline.
3. **Canonical serialization must not iterate unordered containers —
   including ones that are "usually" ordered.** Feature unification in a
   *downstream* crate can flip the backing map; a canonical form is a
   protocol, and protocols do not inherit defaults.
4. **State guarantees over the canonical form, not the input bytes.** A
   loader that accepts arbitrary physical layout owes a named, tested
   canonicalization — "round-trip byte-identical" without that qualifier was
   an overstatement review correctly rejected.
5. **A new dependency must be feature-scoped to the code that needs it.**
   Target gating excludes platforms; only feature gating excludes
   configurations of your own crate.
