# Test fixtures

## `minilm-vocab.txt`

The WordPiece vocabulary of `sentence-transformers/all-MiniLM-L6-v2`
(30522 lines; line *i* is token id *i*). Source:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt

The model is Apache-2.0 licensed; the vocab is reproduced here as a test
fixture with provenance intact.

## `golden_minilm.json`

Golden tokenization vectors generated with the HF Rust `tokenizers`
crate, version 0.21.4 (the exact crate and version Ijima pins), using
Ijima's exact call shape:

```rust
Tokenizer::from_file("MiniLM-tokenizer.json")?
    .with_padding(Some(PaddingParams { strategy: PaddingStrategy::BatchLongest, ..Default::default() }))
// per text: encode(text, /* add_special_tokens */ true)
```

- `singles` (26): one entry per corpus text — `text`, `ids`,
  `attention_mask`. Includes: case folding, NFD accent stripping,
  punctuation isolation, subword segmentation, CJK per-character
  handling, emoji → `[UNK]`, literal `[SEP]`/`[CLS]` raw-text matching,
  `##` literals, empty/whitespace inputs, and an over-length input
  truncating to exactly 128 tokens ending `[SEP]`.
- `batch` (3): BatchLongest right-padding with pad id 0 and attention
  mask 0 (observable only in batch).

Regenerate (only if the corpus changes — committed outputs are the
contract):

```sh
# scratch project with tokenizers = "0.21", replicating the call shape above
```

Katakuri's golden test (`tests/tokenizer_golden.rs`) builds a
tokenizer.json from the committed vocab and asserts byte-identical ids
and masks — which simultaneously proves vocab.txt + typed defaults ≡
tokenizer.json semantics for this model.

## `tiny-mlp.safetensors`

A 769-byte synthetic `ia-mlp` checkpoint exercising the full validated-load
policy (`tests/checkpoint.rs`). Not model weights — a committed fixture with
no third-party provenance, generated deterministically:

- spec: `input_width 2`, one hidden layer of width 3, scalar head,
  `hidden_activation: "relu"` (`ia.spec`, canonical JSON)
- tensors, in canonical order and contiguous: `hidden.0.weight [3, 2]`,
  `hidden.0.bias [3]`, `head.weight [1, 3]`, `head.bias [1]` — 52 bytes of
  F32 payload
- metadata: `ia.schema-hash` (sha256 of the spec's canonical JSON),
  `ia.model-hash` (sha256 of the 52-byte data section), and a synthetic
  `ia.provenance.training-data` digest
- one bias element is a subnormal (`1e-42`), so the policy-on path proves
  subnormals survive the nonfinite scan

Regenerate only by changing the spec or values deliberately: the schema and
model hashes are baked into the metadata, so the fixture is self-verifying.
