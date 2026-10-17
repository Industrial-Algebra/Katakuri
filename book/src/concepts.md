# Concepts & Doctrine

Four ideas explain every design decision in this crate.

## 1. The never-computes boundary

Katakuri loads, validates, and describes. It never computes.

| Katakuri does | Katakuri does not |
|---|---|
| Parse and validate a safetensors manifest | Forward passes |
| Hand tensor bytes to host memory or `zunesha::Buffer` | Autodiff, gradients, optimizers |
| Tokenize text into ids | Kernels, shaders, dispatch |
| Validate a config + weight layout against each other | Numeric weight inspection |

The boundary is not modesty; it is the point. A model loader that also
contains a compute graph drags in a tensor framework, and a tensor framework
drags in a dependency tree nobody in the IA ecosystem audited — which is
exactly the failure that created this repo. Consumers implement the forward
pass on the engine they already trust:

```text
Katakuri:  config.json → BertSpec
           model.safetensors → validated manifest
           tokenizer.json → Tokenizer
              ↓  (typed data: ids, shapes, names, dtypes, byte views)
Consumer:  forward pass on exact CPU math (Amari-style) or Borsalino GPU
```

Because `upload_to_device` produces a `zunesha::Buffer` while the host view
stays valid, the same weights feed either path with no re-loading and no
conversion step. Katakuri takes no side in the workhorse/precision split.

## 2. Strict construction: typed rejection over silent divergence

Wherever an input format is bigger than the subset Katakuri implements, the
implemented subset is **strict** and everything outside it is a typed error —
never a best-effort interpretation.

The rule matters most for parity surfaces. A tokenizer that silently
approximates `BertNormalizer` produces embeddings that are *almost* right,
which is worse than an error, because nothing downstream can tell. So:

| Input | Outside the subset | Result |
|---|---|---|
| `tokenizer.json` model | anything but `WordPiece` | `UnsupportedModel` |
| normalizer | anything but `BertNormalizer` | `UnsupportedNormalizer` |
| pre-tokenizer | anything but `BertPreTokenizer` | `UnsupportedPreTokenizer` |
| post-processor | templates other than `[CLS] A [SEP]` | `UnsupportedPostProcessor` |
| truncation strategy | anything but `LongestFirst` | `UnsupportedTruncationStrategy` |
| truncation direction | anything but `Right` | `UnsupportedTruncationDirection` |
| added-token flags | `single_word` / `lstrip` / `rstrip` | `InvalidTokenizerJson` |
| `config.json` `hidden_act` | anything but gelu / gelu_approximate / relu | `InvalidConfigJson` |
| `config.json` `model_type` | anything but `"bert"` | `InvalidConfigJson` |

The same doctrine governs geometry. `BertSpec` rejects non-positive or
`u32`-overflowing size fields, a `hidden_size` not divisible by
`num_attention_heads`, a non-finite or non-positive `layer_norm_eps`, and an
impractical `num_hidden_layers > 1024` (the last one exists because
`5 + 16L + 2` overflows a 32-bit `usize` and would panic on wasm32 —
strict construction rejects it at parse time instead).

Validation is also **diagnostics-first**: `validate_weights` collects *every*
divergence (missing, unexpected, wrong shape, wrong dtype) into one error
rather than reporting the first. One round trip tells you everything wrong
with a weight set.

## 3. Golden-parity methodology

Parity claims are cheap; parity evidence is not. The tokenizer is verified
against the reference implementation, not against a reading of its source:

1. **Generate goldens with the reference.** 26 single-sequence cases plus 3
   batch cases were produced by the real HF Rust `tokenizers` crate
   (0.21.4 — the exact crate and version Ijima pinned), using Ijima's exact
   call shape (`BatchLongest` padding, `encode(text, add_special_tokens = true)`).
2. **Commit the goldens and the vocab.** `tests/fixtures/golden_minilm.json`
   and `tests/fixtures/minilm-vocab.txt` live in the repo, so CI replays
   parity with **no network and no model download**.
3. **Pin the corpus to the hard cases.** Case folding, NFD accent stripping,
   punctuation isolation, subword segmentation, CJK per-character handling,
   emoji → `[UNK]`, literal `[SEP]`/`[CLS]` in raw text, `##` literals,
   empty/whitespace inputs, and an over-length input truncating to exactly 128
   tokens ending in `[SEP]`.
4. **Assert ids and masks byte-for-byte.** The committed goldens *are* the
   contract; regenerating them is only legitimate when the corpus changes.

A useful side effect: the test builds a tokenizer from the plain vocab with
typed defaults and asserts it matches the goldens. That proves
`vocab.txt` + BERT defaults ≡ `tokenizer.json` semantics for this model, so
both constructors are exercised against the same evidence.

## 4. Content-addressed cache, digest on read

The hub's cache is trustworthy only as far as its digests. Every read
re-hashes:

```text
{root}/{owner}~{model}/{revision}/content/{filename}    ← bytes
{root}/{owner}~{model}/{revision}/digests/{filename}    ← lowercase-hex sha256 + "\n"
```

- A path returned by `Hub::get` held bytes whose sha256 matched the sidecar
  **at return time**.
- Every later `Hub::cached` hit re-reads the file, re-hashes it, and compares.
  A tampered or truncated entry reads as a **miss**, not as data; under
  `CacheFirst` it is refetched and atomically overwritten.
- Writes are atomic: full write to a same-directory temp file, then `rename`.
  No reader ever sees a partial file, and a crashed write cannot leave a
  half-written entry that later verifies.

Three structural choices make the cache safe rather than merely correct:

1. **Namespace separation.** Content, digest sidecars, and temporary files
   occupy separate directories, and a sidecar is named *exactly* the filename.
   A fetched file can therefore never be mistaken for a sidecar — even a file
   literally named `x.sha256` lands under `content/`.
2. **Case-safe encoding.** Every case-bearing component is encoded
   (uppercase `X` → `^` + lowercase `x`); `^` is outside the segment grammar,
   so encoded names contain no uppercase and cannot alias on a
   case-insensitive filesystem. Revision `V1` and `v1` are different cache
   entries on macOS and Windows too.
3. **Strict path grammar.** Owner, model, revision, and filename are single
   segments of ASCII alphanumerics plus `-._`. `..`, `/`, NUL, and everything
   else reject at construction, so a hostile repo id cannot escape the cache
   root.

The cache is **revision-keyed**, not commit-keyed: the revision string is used
exactly as given. `main` re-resolves and can name different bytes over time;
pin a commit sha when you need immutability. The known limitations (no file
locking, Windows reserved device names, revision-string keying) are recorded
in [The Hub](./guide/hub.md#known-limitations) — recorded, not hidden.
