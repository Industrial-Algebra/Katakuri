# `tokenizer`

A bespoke WordPiece (BERT-class) engine: a transcription of the reference
pipeline, verified byte-for-byte against committed golden vectors.

## Constructing

From a line-per-token vocabulary (BERT defaults: `##` continuation,
`[UNK]`/`[CLS]`/`[SEP]`/`[PAD]` resolved from the vocab,
`max_input_chars_per_word` 100, no truncation):

```rust
use katakuri::tokenizer::Tokenizer;

let vocab = std::fs::read("vocab.txt")?;
let tokenizer = Tokenizer::from_bert_vocab_txt(&vocab)?;
```

From a `tokenizer.json` (strict BERT-class subset):

```rust
let json = std::fs::read("tokenizer.json")?;
let tokenizer = Tokenizer::from_tokenizer_json(&json)?;
```

`from_tokenizer_json` accepts `BertNormalizer`, `BertPreTokenizer`,
`TemplateProcessing` with the single-sequence template `[CLS] A [SEP]`, a
`WordPiece` model, an added-tokens list, and `LongestFirst` truncation
(optional). The `decoder` and `padding` fields are accepted and ignored —
single-sequence `encode` never pads and `encode_batch` always uses
`BatchLongest`.

Everything else is a typed rejection, because Katakuri never silently diverges
from a tokenizer configuration it does not implement:

| Field | Accepted | Rejected as |
|---|---|---|
| `normalizer` | `BertNormalizer` | `UnsupportedNormalizer` |
| `pre_tokenizer` | `BertPreTokenizer` | `UnsupportedPreTokenizer` |
| `post_processor` | `[CLS] A [SEP]` template | `UnsupportedPostProcessor` |
| `model` | `WordPiece` | `UnsupportedModel` |
| truncation `strategy` | `LongestFirst` | `UnsupportedTruncationStrategy` |
| truncation `direction` | `Right` | `UnsupportedTruncationDirection` |
| added tokens | non-empty, no boundary flags | `InvalidTokenizerJson` |
| `special_tokens` map | present → fully validated; absent → vocab lookup | `InvalidTokenizerJson` |

## Encoding

```rust
let encoding = tokenizer.encode("Katakuri never computes.", true);
assert_eq!(
    encoding.ids,
    [101, 29354, 18569, 2072, 2196, 24134, 2015, 1012, 102]
);
assert_eq!(encoding.attention_mask, [1, 1, 1, 1, 1, 1, 1, 1, 1]);
```

Against the committed MiniLM vocab. `encode` is infallible: unknown pieces
become `[UNK]`, and truncation (when configured) keeps `max_length − 2`
content tokens before wrapping in `[CLS] … [SEP]`.

Batching right-pads to the batch longest:

```rust
let batch = tokenizer.encode_batch(&["a dog", "the weights are ready"], true);
// [0] ids:  [101, 1037, 3899, 102, 0, 0]      mask: [1, 1, 1, 1, 0, 0]
// [1] ids:  [101, 1996, 15871, 2024, 3201, 102]
```

`Encoding` carries `ids: Vec<u32>` and `attention_mask: Vec<u32>`.

## Introspection

```rust
println!("{} tokens", tokenizer.vocab_size());   // 30522 for MiniLM
println!("max id {}", tokenizer.max_token_id()); // 30521
```

`max_token_id()` is the max over the vocab table, added-token ids, and the four
special ids. It is what the model spec checks against the embedding table —
**capacity**, not cardinality: a sparse vocab with entry id 40000 in a
30522-entry table preserves the count but would index past the table (review
finding 11).

## Pipeline

`encode` runs, in order:

1. **added-token split** on the raw input — leftmost match, longest first;
2. **`BertNormalizer`** — `clean_text`, CJK spacing, accent strip
   (`strip_accents.unwrap_or(lowercase)`), then per-character lowercase;
3. **`BertPreTokenizer`** — whitespace split (whitespace removed) plus
   punctuation isolation;
4. **WordPiece** — greedy longest-match-first with `##` continuations; failure
   or an over-length word emits a whole-word `[UNK]`;
5. **truncation** to `max_length − 2` content tokens (LongestFirst,
   single-sequence), then `[CLS] … [SEP]`;
6. **padding** — only in `encode_batch` (BatchLongest).

Step 2 lowercases **per character**, not via `str::to_lowercase`. That is not a
performance choice: Rust's `str::to_lowercase` applies the contextual final
sigma rule, so `ΟΣ` would become `ος` where the reference produces `οσ`. The
golden corpus pins the per-character behavior (review finding 3):

```text
ΟΔΟΣ → "οδοσ" → [101, 1169, 29722, 29730, 29733, 102]
       ("ο" + "##δ" + "##ο" + "##σ")
```

## Parity evidence

The test suite asserts ids and masks against
`tests/fixtures/golden_minilm.json` — 26 single-sequence cases and 3 batch
cases generated from HF Rust `tokenizers` 0.21.4 with Ijima's exact call
shape. The vocab fixture is committed, so parity replays offline. See
[Concepts](../concepts.md) for the methodology.
