// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Env-gated integration test: parse a real BERT `config.json`, validate a
//! real safetensors weight set against the derived layout manifest, and
//! cross-check the tokenizer vocabulary size.
//!
//! ```sh
//! KATAKURI_TEST_CONFIG_JSON=~/.cache/katakuri-test/MiniLM-config.json \
//! KATAKURI_TEST_SAFETENSORS=~/.cache/katakuri-test/MiniLM-L6-v2-model.safetensors \
//! KATAKURI_TEST_TOKENIZER_JSON=~/.cache/katakuri-test/MiniLM-tokenizer.json \
//!   cargo test --test model_spec -- --ignored
//! ```
//!
//! `KATAKURI_TEST_TOKENIZER_JSON` is optional; without it the committed
//! MiniLM `vocab.txt` fixture is parsed via `from_bert_vocab_txt`.

use katakuri::model::{BertSpec, HiddenAct};
use katakuri::safetensors::SafeTensors;
use katakuri::tokenizer::Tokenizer;

#[test]
#[ignore = "set KATAKURI_TEST_CONFIG_JSON + KATAKURI_TEST_SAFETENSORS to run"]
fn real_file() {
    let (Ok(config_path), Ok(weights_path)) = (
        std::env::var("KATAKURI_TEST_CONFIG_JSON"),
        std::env::var("KATAKURI_TEST_SAFETENSORS"),
    ) else {
        eprintln!(
            "skipping real_file: KATAKURI_TEST_CONFIG_JSON / KATAKURI_TEST_SAFETENSORS not set"
        );
        return;
    };

    let config = std::fs::read_to_string(&config_path).expect("read config.json");
    let spec = BertSpec::from_config_json(&config).expect("parse config.json");

    let weights = std::fs::read(&weights_path).expect("read model.safetensors");
    let st = SafeTensors::parse(&weights).expect("parse model.safetensors");
    spec.validate_weights(&st).expect("weights match spec");

    let tokenizer = match std::env::var("KATAKURI_TEST_TOKENIZER_JSON") {
        Ok(path) => {
            let bytes = std::fs::read(&path).expect("read tokenizer.json");
            Tokenizer::from_tokenizer_json(&bytes).expect("parse tokenizer.json")
        }
        Err(_) => Tokenizer::from_bert_vocab_txt(include_bytes!("fixtures/minilm-vocab.txt"))
            .expect("parse committed MiniLM vocab"),
    };
    spec.check_tokenizer_compatibility(&tokenizer)
        .expect("tokenizer vocabulary matches config");

    assert_eq!(spec.expected_tensors().len(), 103);
    assert_eq!(spec.hidden_act(), HiddenAct::Gelu);
}
