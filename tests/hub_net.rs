// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Env-gated integration test: the full hub → spec loop against the real
//! Hugging Face Hub, with the network pulled after the first fetch.
//!
//! ```sh
//! KATAKURI_TEST_HUB=1 \
//!   cargo test -p katakuri --features hub-net --test hub_net -- --ignored
//! ```
//!
//! Ignored by default; set `KATAKURI_TEST_HUB=1` to opt in.

#[cfg(feature = "hub-net")]
mod net {
    use katakuri::KatakuriError;
    use katakuri::hub::{Fetch, Hub, RepoId, Revision, UreqFetcher};
    use katakuri::model::BertSpec;
    use std::sync::Arc;

    /// Transport that panics if used — proves the second resolution is offline.
    #[derive(Debug)]
    struct PanickyFetcher;

    impl Fetch for PanickyFetcher {
        fn fetch(&self, _url: &str) -> Result<Vec<u8>, KatakuriError> {
            panic!("network must not be touched on a warm cache");
        }
    }

    #[test]
    #[ignore = "set KATAKURI_TEST_HUB=1 to run"]
    fn real_fetch_minilm_config() {
        if std::env::var("KATAKURI_TEST_HUB").as_deref() != Ok("1") {
            eprintln!("skipping real_fetch_minilm_config: KATAKURI_TEST_HUB != 1");
            return;
        }

        let root = std::env::temp_dir().join(format!(
            "katakuri-hub-net-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let repo = RepoId::parse("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        let rev = Revision::main();

        let hub = Hub::new(&root, Arc::new(UreqFetcher::new()));
        let path = hub.get(&repo, &rev, "config.json").unwrap();
        let config = std::fs::read_to_string(&path).unwrap();
        let spec = BertSpec::from_config_json(&config).unwrap();
        assert_eq!(spec.hidden_size(), 384);

        // The cache is warm: a hub with a panicking transport still resolves
        // the same file, and `cached` verifies it offline.
        let offline = Hub::new(&root, Arc::new(PanickyFetcher));
        assert_eq!(
            offline.cached(&repo, &rev, "config.json"),
            Some(path.clone())
        );
        assert_eq!(offline.get(&repo, &rev, "config.json").unwrap(), path);
    }
}
