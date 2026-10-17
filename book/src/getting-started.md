# Getting Started

## Add the dependency

```toml
[dependencies]
katakuri = "0.1"
```

The default feature set is `std` only. No tensor framework, no HTTP client,
no crypto crate: parsing, tokenization, and spec validation build and run
offline on GPU-less hardware. Network fetch is opt-in (`hub-net`) — see
[Feature Flags](./guide/feature-flags.md).

For a local checkout:

```toml
[dependencies]
katakuri = { path = "../Katakuri" }
```

## First code: encode text

The tokenizer needs only a vocab. This is the real
`sentence-transformers/all-MiniLM-L6-v2` vocabulary, but any BERT-class
line-per-token `vocab.txt` works — line *i* is token id *i*.

```rust
use katakuri::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = std::fs::read("vocab.txt")?;
    let tokenizer = Tokenizer::from_bert_vocab_txt(&vocab)?;

    let encoding = tokenizer.encode("Katakuri never computes.", true);
    println!("ids:  {:?}", encoding.ids);
    println!("mask: {:?}", encoding.attention_mask);
    Ok(())
}
```

Against the MiniLM vocab this prints:

```text
ids:  [101, 29354, 18569, 2072, 2196, 24134, 2015, 1012, 102]
mask: [1, 1, 1, 1, 1, 1, 1, 1, 1]
```

`encode` never pads and never fails: unknown pieces become `[UNK]` by design.
Use `encode_batch` when you want BatchLongest right-padding (that is the only
place padding is observable).

From a `tokenizer.json` instead of a bare vocab:

```rust
use katakuri::tokenizer::Tokenizer;

let json = std::fs::read("tokenizer.json")?;
let tokenizer = Tokenizer::from_tokenizer_json(&json)?;
```

Only the BERT-class subset is accepted (`BertNormalizer`, `BertPreTokenizer`,
`TemplateProcessing` with `[CLS] A [SEP]`, `WordPiece`, `LongestFirst`
truncation). Anything else rejects with a typed error rather than silently
diverging from a configuration Katakuri does not implement.

## Next: read weights

The rest of this page shows fragments; they assume the surrounding function is
`fn main() -> Result<(), Box<dyn std::error::Error>>`.

```rust
use katakuri::safetensors::SafeTensors;

let bytes = std::fs::read("model.safetensors")?;
let weights = SafeTensors::parse(&bytes)?;

println!("{} tensors", weights.tensor_count());
for name in weights.names() {
    let tensor = weights.tensor(name)?;
    println!("{name}: {:?} {:?}", tensor.dtype(), tensor.shape());
}
```

Parsing validates the whole manifest up front: dtype, shape, offsets, span
arithmetic, and non-overlap. A file that parses is a file whose every tensor
can be sliced safely.

## Next: validate a model end to end

```rust
use katakuri::model::BertSpec;

let spec = BertSpec::from_config_json(&std::fs::read_to_string("config.json")?)?;
spec.validate_weights(&weights)?;          // names, dtypes, shapes, completeness
spec.check_tokenizer_compatibility(&tokenizer)?; // ids fit the embedding table
```

## Runnable examples

The repository ships three, each runnable against a checkout:

```sh
cargo run --example inspect-safetensors -- tests/fixtures/tiny-model.safetensors
cargo run --example encode-text
cargo run --example fetch-and-validate --features hub-net   # first run downloads ~90 MB
cargo run --example fetch-and-validate --features hub-net -- --offline
```

`fetch-and-validate` fetches `all-MiniLM-L6-v2` through the hub, parses the
config into a `BertSpec`, validates the real 90 MB weight file against the
derived layout, and cross-checks tokenizer/embedding capacity. `--offline`
accepts cache hits only and never touches the network.
