# Katakuri

Model ecosystem layer over [Zunesha](https://github.com/Industrial-Algebra/Zunesha) — safetensors weight loading, WordPiece tokenization, architecture specs, and hub access. **Katakuri never computes**: no forward passes, no autodiff, no kernels, no shader compilation. Consumers implement forward passes on their tensor engine of choice (exact CPU math today; Borsalino-class GPU compute later). The same weights feed either path; Katakuri takes no side.

## Layers

| Crate | Role |
|---|---|
| Zunesha | GPU device substrate — device, queues, memory, `zunesha::Buffer` |
| Borsalino | GPU compute abstraction — compiles shaders, dispatches |
| Goldenweek | Graphics (render) layer over Zunesha |
| **Katakuri** | **This repo.** Model ecosystem: weights, tokenization, specs, hub. Never computes. |

Weight data lands in `zunesha::Buffer` when a device exists, and in host memory via Zunesha's `NoDevice` path otherwise — GPU-less hardware is a first-class requirement, not a fallback.

## Status

Pre-`0.1.0` scaffold. The slice plan (S1 safetensors → S2 tokenizer → S3 model specs → S4 hub) and full architecture live in [`docs/plans/2026-10-16-katakuri-architecture-handover.md`](docs/plans/2026-10-16-katakuri-architecture-handover.md).

## Development

```sh
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributors must sign the [CLA](https://github.com/Industrial-Algebra/.github/blob/main/CLA.md).

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
