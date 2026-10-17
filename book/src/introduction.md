# Introduction

**Katakuri** is the model-ecosystem layer over
[Zunesha](https://github.com/Industrial-Algebra/Zunesha): safetensors weight
loading, BERT-class WordPiece tokenization, architecture specs as typed data,
and hub access.

> The one who sees ahead — the weights are ready before the compute begins.

## What it does

Katakuri owns everything *around* tensor math:

- **`safetensors`** — parse and fully validate a weight file into a
  zero-copy manifest, then hand tensor bytes either to host memory or to a
  `zunesha::Buffer` on a device.
- **`tokenizer`** — a bespoke WordPiece engine that reproduces the reference
  tokenizer byte-for-byte on committed golden vectors.
- **`model`** — a typed BERT `config.json` (`BertSpec`) plus weight-layout
  manifests validated against a parsed weight file.
- **`hub`** — model-file fetch and a content-addressed, digest-verified local
  cache; offline-first, with the network transport behind its own feature.

## The hard boundary: Katakuri never computes

**No forward implementations. No autodiff. No kernels. No shader
compilation.**

Katakuri stops precisely where tensor math starts. It loads, validates, and
describes the data consumers compute over. Consumers implement forward passes
on their tensor engine of choice — exact CPU math today, Borsalino-class GPU
compute later. The same weights feed either path; Katakuri takes no side.

This is the structural form of the workhorse / precision doctrine:
**Borsalino is the workhorse, Amari is the precision** — and Katakuri is the
data plane both read from, so neither has to own a model loader.

The boundary is enforced by the dependency ledger as much as by policy: no
tensor framework, no BLAS, no GPU compute crate, and no async runtime appears
anywhere in the tree.

## Why it exists

On 2026-10-16 `zip 8.6.0` was yanked from crates.io, making
`candle-core 0.11.0`'s `zip ^8.6.0` requirement permanently unsatisfiable.
Amari's CI lost 15 of 16 workflow jobs at dependency *resolution* — before a
single line compiled — over a candle dependency that repo never used.

The strategic response was not a bespoke candle replacement. It was an
IA-owned layer for the parts that are *not* tensor math, so the ecosystem does
not inherit a heavy ML framework's dependency risk for weight parsing,
tokenization, config validation, and file fetching. That is Katakuri.

## Layering over Zunesha

| Crate | Role |
|---|---|
| Zunesha | GPU device substrate — device, queues, memory, `zunesha::Buffer` |
| Borsalino | GPU compute abstraction — compiles shaders, dispatches |
| Goldenweek | Graphics (render) layer over Zunesha |
| **Katakuri** | **This crate.** Model ecosystem: weights, tokenization, specs, hub. Never computes. |

Weight data lands in `zunesha::Buffer` when a device exists, and in host
memory via Zunesha's `NoDevice` path otherwise — GPU-less hardware (Ijima's
deployment class) is a first-class requirement, not a fallback. Katakuri is a
*consumer* of Zunesha, never a wrapper: buffers flow through, and alignment
and lifecycle obligations remain Zunesha's.

## Status

`0.1.0`: slices S1 (safetensors), S2 (tokenizer), S3 (model specs), and S4
(hub) are complete. These are the four surfaces that replaced candle and
`hf-hub` in Ijima's embedding path; the tokenizer is bit-for-bit identical
to the reference on the committed golden vectors, and the forward pass stays
in the consumer, as designed.
