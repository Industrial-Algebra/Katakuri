# Katakuri

[![crates.io](https://img.shields.io/crates/v/katakuri.svg)](https://crates.io/crates/katakuri)
[![docs](https://img.shields.io/badge/docs-mdBook_sources-00cccc)](book/src/SUMMARY.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Model ecosystem layer over [Zunesha](https://github.com/Industrial-Algebra/Zunesha) — safetensors weight loading, WordPiece tokenization, BERT architecture specs, and hub access.

**Katakuri never computes**: no forward passes, no autodiff, no kernels, no shader compilation. It loads, validates, and describes the data consumers compute over. The same weights feed an exact CPU path (Amari-class precision) or a Borsalino GPU path; Katakuri takes no side.

## Layers

| Crate | Role |
|---|---|
| Zunesha | GPU device substrate — device, queues, memory, `zunesha::Buffer` |
| Borsalino | GPU compute abstraction — compiles shaders, dispatches |
| Goldenweek | Graphics (render) layer over Zunesha |
| **Katakuri** | **This crate.** Model ecosystem: weights, tokenization, specs, hub. Never computes. |

Weight data lands in `zunesha::Buffer` when a device exists, and in host memory via Zunesha's `NoDevice` path otherwise — GPU-less hardware is a first-class requirement, not a fallback.

## Status

`0.1.0` — slices S1–S4 complete: safetensors parse/validate, the WordPiece tokenizer, BERT model specs, and the hub. Adopted by Ijima for its embedding path, bit-for-bit on tokenization against the reference implementation's golden vectors. Ijima keeps the forward pass, as designed.

## Quick start

```toml
[dependencies]
katakuri = "0.1"
```

```rust
use katakuri::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = std::fs::read("vocab.txt")?;          // line i = token id i
    let tokenizer = Tokenizer::from_bert_vocab_txt(&vocab)?;

    let encoding = tokenizer.encode("Katakuri never computes.", true);
    println!("{:?}", encoding.ids);
    // [101, 29354, 18569, 2072, 2196, 24134, 2015, 1012, 102]  (MiniLM vocab)
    Ok(())
}
```

Runnable examples: `cargo run --example inspect-safetensors -- tests/fixtures/tiny-model.safetensors`, `cargo run --example encode-text`, `cargo run --example fetch-and-validate --features hub-net` (first run ~90 MB, then cached).

## Feature flags

| Feature | Default | Enables | Pulls in |
|---|---|---|---|
| `std` | ✅ | Standard-library support | — |
| `verify` | | karpal-verify obligation bundles for the parse invariants | `karpal-verify`, `karpal-proof` |
| `hub` | | Model fetch + content-addressed cache (offline-first) | `sha2` |
| `hub-net` | | Blocking HTTPS transport on top of `hub` | `hub` + `ureq` (rustls) |

The hub adds nothing to the default dependency tree: `Fetch` is a dependency-free seam, so cache/digest/path code builds and tests without a network client. No async runtime anywhere.

## What it is — and is not

**IS** — safetensors parse + full validation (dtype/shape/offset/overlap), upload to `zunesha::Buffer`; BERT-class WordPiece tokenization with golden parity; typed `config.json` specs and weight-layout validation; model fetch with a digest-verified content-addressed cache.

**IS NOT** — forward passes, autodiff, optimizers, kernels/shaders, training, numeric weight inspection, metadata/model listing APIs, or async.

## Documentation

Book: mdBook sources in [`book/`](book/) (hosting planned; not yet deployed) ·  · API: <https://docs.rs/katakuri> · Contracts: [`docs/plans/`](docs/plans/) · [Roadmap](docs/ROADMAP.md) · [Critique](docs/critique.md) · [Changelog](CHANGELOG.md).

## Development

```sh
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md); all contributors must sign the [CLA](https://github.com/Industrial-Algebra/.github/blob/main/CLA.md).

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
