# Contributing to Katakuri

Thank you for your interest in contributing to Katakuri!

## CLA

All contributors must sign the [Contributor License Agreement](https://github.com/Industrial-Algebra/.github/blob/main/CLA.md) before contributions can be merged.

## Development

```sh
git clone https://github.com/Industrial-Algebra/Katakuri.git
cd Katakuri
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Git Flow

- Feature branches: `feature/<description>` off `develop`
- Hotfixes: `hotfix/<description>` off `main`
- PRs target `develop` for features, `main` for hotfixes
- Release PRs go from `develop` to `main`

## Scope discipline

Katakuri **never computes** — no forward implementations, no autodiff, no
kernels, no shader compilation. If a change needs tensor math, it belongs in
a consumer (CPU today, Borsalino-class compute later), not here. New
dependencies require a written justification in the PR; `--all-features`
resolvability is a CI-visible property.

## License

All contributions are licensed under [Apache-2.0](./LICENSE).
