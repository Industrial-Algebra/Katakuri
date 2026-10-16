// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Golden-vector parity: `katakuri::tokenizer` vs HF Rust `tokenizers
//! 0.21.4`, byte-identical token id sequences.
//!
//! The goldens (`tests/fixtures/golden_minilm.json`) were generated from
//! the real `sentence-transformers/all-MiniLM-L6-v2` tokenizer with
//! Ijima's exact call shape. This test builds a tokenizer.json from the
//! committed vocab fixture (line *i* = id *i*) and replays every golden —
//! which also proves vocab.txt + typed defaults ≡ tokenizer.json
//! semantics for this model.

use katakuri::tokenizer::Tokenizer;

fn load_golden() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/golden_minilm.json")).expect("golden json")
}

fn build_from_committed_vocab() -> Tokenizer {
    let vocab_lines: Vec<&str> = include_str!("fixtures/minilm-vocab.txt").lines().collect();
    assert_eq!(vocab_lines.len(), 30_522, "MiniLM vocab size");
    let mut vocab_obj = serde_json::Map::new();
    for (i, tok) in vocab_lines.iter().enumerate() {
        vocab_obj.insert((*tok).to_string(), serde_json::Value::from(i as u32));
    }
    let special = |content: &str, id: u32| {
        serde_json::json!({
            "id": id, "content": content, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false,
            "special": true,
        })
    };
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": {"max_length": 128, "strategy": "LongestFirst", "stride": 0},
        "padding": null,
        "added_tokens": [
            special("[PAD]", 0), special("[UNK]", 100), special("[CLS]", 101),
            special("[SEP]", 102), special("[MASK]", 103),
        ],
        "normalizer": {"type": "BertNormalizer", "clean_text": true,
            "handle_chinese_chars": true, "strip_accents": null,
            "lowercase": true},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {"type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
            ]},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab_obj},
    });
    let bytes = serde_json::to_vec(&tokenizer_json).expect("serialize");
    Tokenizer::from_tokenizer_json(&bytes).expect("parse committed vocab")
}

fn ids_of(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("ids array")
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("u32 id")).expect("u32 id"))
        .collect()
}

#[test]
fn golden_singles_match_hf_tokenizers() {
    let tk = build_from_committed_vocab();
    let golden = load_golden();
    let singles = golden["singles"].as_array().expect("singles");
    assert_eq!(singles.len(), 26);
    for s in singles {
        let text = s["text"].as_str().expect("text");
        let expect_ids = ids_of(&s["ids"]);
        let expect_mask = ids_of(&s["attention_mask"]);
        let enc = tk.encode(text, true);
        assert_eq!(enc.ids, expect_ids, "ids diverge from HF on {text:?}");
        assert_eq!(
            enc.attention_mask, expect_mask,
            "mask diverges from HF on {text:?}"
        );
    }
}

#[test]
fn golden_batch_matches_hf_tokenizers() {
    let tk = build_from_committed_vocab();
    let golden = load_golden();
    let batch = golden["batch"].as_array().expect("batch");
    assert_eq!(batch.len(), 3);
    let texts: Vec<&str> = batch
        .iter()
        .map(|b| b["text"].as_str().expect("text"))
        .collect();
    let encs = tk.encode_batch(&texts, true);
    for (enc, b) in encs.iter().zip(batch.iter()) {
        assert_eq!(enc.ids, ids_of(&b["ids"]), "batch ids diverge");
        assert_eq!(enc.attention_mask, ids_of(&b["attention_mask"]));
    }
}

#[test]
#[ignore = "set KATAKURI_TEST_TOKENIZER_JSON=<path to a real tokenizer.json> to run"]
fn real_tokenizer_json_file_matches_goldens() {
    let path = std::env::var("KATAKURI_TEST_TOKENIZER_JSON")
        .expect("KATAKURI_TEST_TOKENIZER_JSON not set");
    let bytes = std::fs::read(&path).expect("read file");
    let tk = Tokenizer::from_tokenizer_json(&bytes).expect("parse real file");
    let golden = load_golden();
    for s in golden["singles"]
        .as_array()
        .expect("singles")
        .iter()
        .take(5)
    {
        let text = s["text"].as_str().expect("text");
        let enc = tk.encode(text, true);
        assert_eq!(
            enc.ids,
            ids_of(&s["ids"]),
            "real-file ids diverge on {text:?}"
        );
    }
}
