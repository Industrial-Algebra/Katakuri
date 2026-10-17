# Feature Flags

Katakuri is additive-feature-gated. The default build is the offline core —
weights, tokenization, specs — and every network or verification surface is
opt-in.

| Feature | Default | Enables | Pulls in |
|---|---|---|---|
| `std` | ✅ | Standard-library support | — |
| `verify` | | karpal-verify obligation bundles for the parse invariants | `karpal-verify`, `karpal-proof` |
| `hub` | | Model fetch + content-addressed cache (offline-first), `Fetch` seam | `sha2` |
| `hub-net` | | Blocking HTTPS transport (`UreqFetcher`) on top of `hub` | `hub` + `ureq` (rustls) |

`default = ["std"]`.

## The zero-dependency default (for the hub surface)

The base crate depends on five small crates — `thiserror`, `serde_json`,
`zunesha`, and the two Unicode tables the reference tokenizer itself uses.
The hub adds **nothing** to that: the `Fetch` trait is a dependency-free seam,
so the cache, path validation, and digest machinery build and test with no
network client at all. `sha2` arrives only with `hub`, and `ureq` only with
`hub-net`. There is no async runtime anywhere.

## Combinations

| Goal | Features | Command |
|---|---|---|
| Parse weights, tokenize, validate specs — no network | (none) | `cargo build` |
| The above + karpal obligation bundles | `verify` | `cargo build --features verify` |
| Cache + custom transport (embed a non-ureq client) | `hub` | `cargo build --features hub` |
| Cache + batteries-included HTTPS | `hub-net` | `cargo build --features hub-net` |
| Everything | all | `cargo build --all-features` |

`hub-net` implies `hub`; you never enable both by hand.

## What each gate buys

- **`verify`** — three `ObligationBundle`s (`IsOffsetInBounds`,
  `IsSpanConsistent`, `IsManifestNonOverlapping`) exported for the karpal
  SMT/Lean/Kani backends, tier `Impossible`: the validator makes the illegal
  state unrepresentable. See the book's [Verification](../book/src/design/verification.md) page.
- **`hub`** — `katakuri::hub`: `RepoId`, `Revision`, `Hub`, `Policy`, the
  `Fetch` trait, `sha256_hex`. Offline-first: with a warm cache, no network
  code path is reachable.
- **`hub-net`** — `UreqFetcher` (blocking, rustls, HTTPS-only including the
  redirect chain, optional bearer token). The only surface in this crate that
  opens a socket.

## CI matrix

Every gate is exercised in both configurations, plus a wasm32 check:

```sh
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --features hub-net
cargo check --target wasm32-unknown-unknown --features hub
```
