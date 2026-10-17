# Verification

Katakuri carries the Borsalino verification pattern in two independent forms:
**obligation bundles** (machine-checked properties, feature `verify`) and
**Kani harnesses** (bounded model checking of the validation arithmetic).

## Obligation bundles (`verify`)

Every property the safetensors validator enforces is exported as a karpal
obligation, tier
[`Impossible`](https://docs.rs/karpal-verify) — the validator makes the
illegal state unrepresentable, so "a manifest that parses is a manifest that
satisfies them".

| Property | What it prevents | Enforced by |
|---|---|---|
| `IsOffsetInBounds` | offsets escaping the data region | `OffsetOutOfBounds` rejection |
| `IsSpanConsistent` | span ≠ shape × dtype bytes | `SizeMismatch` / `SizeOverflow` rejection |
| `IsManifestNonOverlapping` | two tensors claiming the same bytes | `OverlappingTensors` rejection |

```rust
use katakuri::verify::safetensors_parse_obligations;

let bundle = safetensors_parse_obligations();
assert_eq!(bundle.name, "katakuri_safetensors_parse");
assert_eq!(bundle.obligations().len(), 3);
for obligation in bundle.obligations() {
    assert!(matches!(obligation.tier, karpal_verify::VerificationTier::Impossible));
}
```

`verify` is optional and pulls `karpal-verify` / `karpal-proof` at the version
Borsalino pins. Bundles export to the karpal SMT / Lean / Kani backends.

Deferred upstream: the karpal-verify 0.6.1 SMT exporter has no operator
mapping for the terms these bundles emit, so SMT export is not exercised from
Katakuri yet (recorded on S1 review round 1; karpal-verify's own `gpu.rs`
uses the same `Term::app` convention).

## Kani harnesses

`src/kani_harnesses.rs` is compiled only under `cargo kani` (inert in normal
builds, declared via `check-cfg = ['cfg(kani)']` so `-D warnings` stays clean):

```sh
cargo kani --harness span_consistency_arithmetic
cargo kani --harness span_overflow_rejected
cargo kani --harness offset_containment_implies_valid_slice
```

| Harness | Property proved |
|---|---|
| `span_consistency_arithmetic` | under bounded dims (≤ 65536 each), the span is exactly product × dtype size and a dtype multiple |
| `span_overflow_rejected` | `u64::MAX` dims are detected as overflow, never wrapped |
| `offset_containment_implies_valid_slice` | `start ≤ end ≤ data_len` implies a non-panicking slice of exactly `end − start` bytes |

The arithmetic they cover lives in a pure function (`span_for`) deliberately
separated from parsing, so the proofs target the arithmetic rather than the
JSON plumbing.

**Status:** the harnesses are written and gated by `#[cfg(kani)]`, but Kani is
not installed in CI — executing them is a follow-up, recorded in
`docs/critique.md` and the roadmap.

## Width correctness (32-bit targets)

Review loops found two 32-bit-only failures that no 64-bit test would catch,
both from arithmetic that is correct when `usize` is 64 bits wide:

| Finding | Failure on wasm32 | Fix |
|---|---|---|
| `max_token_id() + 1 ≤ vocab_size` | `+ 1` wraps to 0 at `u32::MAX` | compare `max_token_id() >= vocab_size` |
| `5 + 16·L + 2` tensor count | overflows near `u32::MAX` layers | reject `num_hidden_layers > 1024` at parse time |

The CI matrix therefore includes a real 32-bit compile:

```sh
cargo check --target wasm32-unknown-unknown --features hub
```

This is a compile-level guard, not a proof: it keeps the width-sensitive
code path building, while the two findings above are held by regressions
(`u32_max_token_id_rejects_without_overflow`, the 1024/1025 boundary test).
