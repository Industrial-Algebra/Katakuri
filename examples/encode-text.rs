// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Encode text with the committed MiniLM WordPiece vocabulary.
//!
//! ```sh
//! cargo run --example encode-text
//! ```
//!
//! The vocab is the real `sentence-transformers/all-MiniLM-L6-v2` fixture
//! committed under `tests/fixtures/` (line *i* is token id *i*), so this
//! example needs no network and no model download. The Greek pair shows the
//! per-character lowercase rule: `ΟΔΟΣ` lowercases to `οδοσ` (no contextual
//! final sigma), matching the HF reference exactly.

use katakuri::tokenizer::Tokenizer;

fn main() {
    let tokenizer =
        Tokenizer::from_bert_vocab_txt(include_bytes!("../tests/fixtures/minilm-vocab.txt"))
            .expect("committed MiniLM vocab parses");
    println!(
        "vocab: {} tokens, max emittable id {}\n",
        tokenizer.vocab_size(),
        tokenizer.max_token_id()
    );

    let sentences = [
        "Katakuri never computes.",
        "The weights are ready before the compute begins.",
        "ΟΔΟΣ",
        "Embeddings, punctuation, and 日本語.",
    ];

    for text in sentences {
        let encoding = tokenizer.encode(text, true);
        println!("text:  {text}");
        println!("ids:   {:?}", encoding.ids);
        println!("mask:  {:?}", encoding.attention_mask);
        println!();
    }

    // BatchLongest right-padding: the only place padding is observable.
    let batch = tokenizer.encode_batch(&["a dog", "the weights are ready"], true);
    println!("batch (right-padded to the batch longest):");
    for (i, encoding) in batch.iter().enumerate() {
        println!("  [{i}] ids:  {:?}", encoding.ids);
        println!("  [{i}] mask: {:?}", encoding.attention_mask);
    }
}
