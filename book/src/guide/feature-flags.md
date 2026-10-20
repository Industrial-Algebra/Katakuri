# Feature Flags

Katakuri is additive-feature-gated. The default build is the offline core —
weights, tokenization, specs — and every network or verification surface is
opt-in.

| Feature | Default | Enables | Pulls in |
|---|---|---|---|
| `std` | ✅ | Standard-library support (file I/O for `checkpoint`) | — |
| `verify` | | karpal-verify obligation bundles for the parse invariants | `karpal-verify`, `karpal-proof` |
| `digest` | | Canonical sha256 artifact identities (`artifact_sha256`) | `sha2` (pinned `=0.11.0`) |
| `hub` | | Model fetch + content-addressed cache (offline-first), `Fetch` seam | `digest` + `sha2` |
| `hub-net` | | Blocking HTTPS transport (`UreqFetcher`) on top of `hub` | `hub` + `ureq` (rustls) |

`checkpoint` is a surface, not a feature name: it requires `std` **and**
`digest` (file I/O plus hash verification). The metadata layer — specs and the
layout manifest — needs neither.

`default = ["std"]`.

## The zero-dependency default (for the hub surface)

The base crate depends on five small crates — `thiserror`, `serde_json`,
`zunesha`, and the two Unicode tables the reference tokenizer itself uses.
The hub adds **nothing** to that: the `Fetch` trait is a dependency-free seam,
so the cache, path validation, and digest machinery build and test with no
network client at all. `sha2` arrives only with `digest` (and `hub`, which
depends on it — one hash implementation, re-exported rather than duplicated),
and `ureq` only with `hub-net`. There is no async runtime anywhere.

`sha2` is pinned to exactly `0.11.0`: `0.11.1` raised its MSRV to Rust 1.88,
above this crate's declared 1.85.

## Combinations

| Goal | Features | Command |
|---|---|---|
| Parse weights, tokenize, validate specs — no hashing, no file I/O | (none) | `cargo build` |
| The above + karpal obligation bundles | `verify` | `cargo build --features verify` |
| Local artifact digests + validated checkpoint loading | `digest` | `cargo build --features digest` |
| Cache + custom transport (embed a non-ureq client) | `hub` | `cargo build --features hub` |
| Cache + batteries-included HTTPS | `hub-net` | `cargo build --features hub-net` |
| Everything | all | `cargo build --all-features` |

`hub-net` implies `hub`, which implies `digest`; you never enable them by
hand. There is no hub-on-digest-off configuration — the crate asserts that in
a doctest and a `compile_error!`.

## What each gate buys

- **`verify`** — three `ObligationBundle`s (`IsOffsetInBounds`,
  `IsSpanConsistent`, `IsManifestNonOverlapping`) exported for the karpal
  SMT/Lean/Kani backends, tier `Impossible`: the validator makes the illegal
  state unrepresentable. See [Verification](../design/verification.md).
- **`digest`** — `katakuri::digest`: `artifact_sha256` (and the S4-compatible
  `sha256_hex` alias). Local artifact identity for checkpoint metadata
  (`ia.schema-hash`, `ia.model-hash`) without pulling in the hub.
- **`checkpoint`** (`std` + `digest`) — `katakuri::checkpoint`:
  `load_validated`, `TrustPolicy`, `ValidatedCheckpoint`. The ordered trust
  pipeline; see [MLP Specs & Validated Checkpoints](mlp-specs.md).
- **`hub`** — `katakuri::hub`: `RepoId`, `Revision`, `Hub`, `Policy`, the
  `Fetch` trait, and the re-exported `sha256_hex`. Offline-first: with a warm
  cache, no network code path is reachable.
- **`hub-net`** — `UreqFetcher` (blocking, rustls, HTTPS-only including the
  redirect chain, optional bearer token). The only surface in this crate that
  opens a socket.

## CI matrix

Every gate is exercised in both configurations, plus a wasm32 check:

```sh
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --features hub-net
cargo test --features digest
cargo check --target wasm32-unknown-unknown --features hub
```
