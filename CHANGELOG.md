# Changelog

All notable changes to Katakuri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — repo scaffold (2026-10-16)

- Library crate skeleton (`katakuri`, lib target; replaces the scaffold's
  binary stub) per IA conventions: Apache-2.0 + SPDX headers on every source
  file, pinned stable toolchain (`rust-toolchain.toml`), central
  `KatakuriError` (thiserror), and module stubs for the four planned
  surfaces — `safetensors` (S1), `tokenizer` (S2), `model` (S3), `hub` (S4).
- Licensing and process stack: Apache-2.0 `LICENSE`, `CONTRIBUTING.md` with
  CLA reference, Forgejo mirror under `industrial-algebra` with dual-push
  wiring.
