// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Fetch all-MiniLM-L6-v2 through the hub, then validate the whole
//! model-ecosystem chain: config → spec, weights → layout, tokenizer →
//! embedding capacity.
//!
//! ```sh
//! # first run downloads ~90 MB into the cache; later runs are offline hits
//! cargo run --example fetch-and-validate --features hub-net
//!
//! # only accept cache hits — never touch the network
//! cargo run --example fetch-and-validate --features hub-net -- --offline
//! ```
//!
//! Cache root: `$KATAKURI_HUB_DIR`, else `$HOME/.cache/katakuri/hub`.
//! `HF_TOKEN`, when set, is sent as a bearer token (gated repos / rate
//! limits); it is never printed.

fn main() {
    #[cfg(feature = "hub-net")]
    net::run();

    #[cfg(not(feature = "hub-net"))]
    {
        eprintln!(
            "this example needs the network transport: \
             cargo run --example fetch-and-validate --features hub-net"
        );
        std::process::exit(2);
    }
}

#[cfg(feature = "hub-net")]
mod net {
    use katakuri::hub::{Hub, RepoId, Revision, UreqFetcher};
    use katakuri::model::BertSpec;
    use katakuri::safetensors::SafeTensors;
    use katakuri::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";

    pub fn run() {
        let offline = std::env::args().skip(1).any(|a| a == "--offline");

        if let Err(err) = validate(offline) {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }

    fn validate(offline: bool) -> Result<(), Box<dyn std::error::Error>> {
        let repo = RepoId::parse(REPO)?;
        let revision = Revision::main();

        let fetcher: Arc<dyn katakuri::hub::Fetch> = match std::env::var("HF_TOKEN") {
            Ok(token) if !token.is_empty() => Arc::new(UreqFetcher::with_token(token)),
            _ => Arc::new(UreqFetcher::new()),
        };
        let hub = Hub::with_default_root(fetcher)?;

        println!("repo:    {REPO} @ {}", revision.as_str());
        let rev_dir = hub.cache_path(&repo, &revision, "config.json");
        let rev_dir = rev_dir.parent().and_then(Path::parent).unwrap_or(&rev_dir);
        println!("cache:   {}", rev_dir.display());
        println!(
            "mode:    {}",
            if offline {
                "offline (cache hits only)"
            } else {
                "online (fetch on miss)"
            }
        );
        println!();

        let config_path = resolve(&hub, &repo, &revision, "config.json", offline)?;
        let weights_path = resolve(&hub, &repo, &revision, "model.safetensors", offline)?;
        println!();

        // config.json → typed spec (strict BERT subset).
        let config = std::fs::read_to_string(&config_path)?;
        let spec = BertSpec::from_config_json(&config)?;
        println!(
            "spec:      hidden_size={} layers={} heads={} intermediate={}",
            spec.hidden_size(),
            spec.num_hidden_layers(),
            spec.num_attention_heads(),
            spec.intermediate_size()
        );
        println!(
            "           vocab_size={} type_vocab={} max_positions={} act={:?} eps={}",
            spec.vocab_size(),
            spec.type_vocab_size(),
            spec.max_position_embeddings(),
            spec.hidden_act(),
            spec.layer_norm_eps()
        );
        println!(
            "           expected layout: {} tensors",
            spec.expected_tensors().len()
        );

        // model.safetensors → validated manifest, then layout validation.
        let weights_bytes = std::fs::read(&weights_path)?;
        let weights = SafeTensors::parse(&weights_bytes)?;
        println!(
            "weights:   parsed {} tensors, {} bytes of data",
            weights.tensor_count(),
            weights.data_len()
        );
        match spec.validate_weights(&weights) {
            Ok(()) => println!("           layout OK — every expected name/dtype/shape present"),
            Err(err) => {
                println!("           layout MISMATCH:\n{err}");
                return Err(err.into());
            }
        }

        // tokenizer → embedding-table capacity (max emittable id, not count).
        let tokenizer =
            Tokenizer::from_bert_vocab_txt(include_bytes!("../tests/fixtures/minilm-vocab.txt"))?;
        spec.check_tokenizer_compatibility(&tokenizer)?;
        println!(
            "tokenizer: compatible — max emittable id {} < vocab_size {}",
            tokenizer.max_token_id(),
            spec.vocab_size()
        );

        println!();
        println!("all checks passed — the weights are ready before the compute begins");
        Ok(())
    }

    /// Cache hit (digest-verified) → path; otherwise fetch unless `offline`.
    fn resolve(
        hub: &Hub,
        repo: &RepoId,
        revision: &Revision,
        filename: &str,
        offline: bool,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(path) = hub.cached(repo, revision, filename) {
            report(filename, "cache hit (digest verified)", &path);
            return Ok(path);
        }
        if offline {
            return Err(format!("{filename}: not in cache (--offline refuses to fetch)").into());
        }
        let path = hub.get(repo, revision, filename)?;
        report(filename, "fetched", &path);
        Ok(path)
    }

    fn report(filename: &str, how: &str, path: &Path) {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        println!("{filename:<20} {how} — {size} bytes");
        println!("{:<20} {}", "", path.display());
    }
}
