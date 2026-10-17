# `hub`

Model fetch + content-addressed cache, offline-first (feature `hub`; the
network transport is feature `hub-net`).

Replaces `hf-hub` as the acquisition path. The default crate gains no
dependencies from this module: `Fetch` is a dependency-free trait, `sha2` is
the only addition under `hub`, and `ureq` only under `hub-net`.

## The types

| Type | Role |
|---|---|
| `RepoId` | `owner/model` identity; strict single/two-segment grammar |
| `Revision` | Branch, tag, or 40-hex commit sha — a validating newtype |
| `Hub` | The cache + fetcher; `new`, `with_default_root`, `get`, `get_with`, `cached`, `cache_path` |
| `Policy` | `CacheFirst` (default) or `AlwaysFetch` |
| `Fetch` | The transport seam: one method, no dependencies |
| `UreqFetcher` | Batteries-included blocking HTTPS transport (`hub-net`) |
| `sha256_hex` | Lowercase-hex sha256 helper (also the sidecar format) |

Constructors validate:

```rust
use katakuri::hub::{RepoId, Revision, validate_filename};

let repo = RepoId::parse("sentence-transformers/all-MiniLM-L6-v2")?;
assert_eq!(repo.as_url_segment(), "sentence-transformers/all-MiniLM-L6-v2");

let rev = Revision::main();                    // "main"
let pinned = Revision::new("8b3219a92973c328a8e22fadc826d4d1721c4ba7")?;
validate_filename("model.safetensors")?;
```

Grammar: one segment (two for `RepoId`), each character ASCII alphanumeric or
`-._`; no `..`, `/`, NUL, or empty segment. `Display` and `FromStr`
round-trip.

## Fetch + cache

```rust
use katakuri::hub::{Hub, Policy, RepoId, Revision, UreqFetcher};
use std::sync::Arc;

let repo = RepoId::parse("sentence-transformers/all-MiniLM-L6-v2")?;
let rev = Revision::main();
let hub = Hub::with_default_root(Arc::new(UreqFetcher::new()))?;

// CacheFirst: verified hit → path; miss/corrupt → fetch → verify → atomic write.
let path = hub.get(&repo, &rev, "model.safetensors")?;

// Explicit policy: re-fetch and overwrite even on a hit.
let path = hub.get_with(&repo, &rev, "config.json", Policy::AlwaysFetch)?;

// Offline only: digest-verified hit → Some(path); miss/corrupt → None.
if let Some(path) = hub.cached(&repo, &rev, "config.json") {
    let config = std::fs::read_to_string(path)?;
}
```

`get` returns a **local path**, never a URL. `cache_path` is a pure join:
useful for display, logging, and pre-flight checks.

Root resolution: `$KATAKURI_HUB_DIR`, else `$HOME/.cache/katakuri/hub`. A
custom root is just `Hub::new(path, fetch)`.

## The transport seam

```rust
use katakuri::KatakuriError;
use katakuri::hub::Fetch;

#[derive(Debug)]
struct Fake {
    body: Vec<u8>,
}

impl Fetch for Fake {
    fn fetch(&self, _url: &str) -> Result<Vec<u8>, KatakuriError> {
        Ok(self.body.clone())
    }
}
```

`Fetch: Send + Sync + Debug` with one method:

```rust
fn fetch(&self, url: &str) -> Result<Vec<u8>, KatakuriError>
```

Implementations must be HTTPS and must follow redirects per HTTP semantics.
Errors flow through `KatakuriError::HubFetch`. Tests run on fakes, so the hub
is hermetic; embedding a different HTTP client is one impl block.

## `UreqFetcher`

```rust
use katakuri::hub::UreqFetcher;

let anon = UreqFetcher::new();                       // public repos
let authed = UreqFetcher::with_token("hf_...");      // gated repos / rate limits
```

- Blocking; no async runtime.
- rustls; `https_only(true)` on the agent, so redirect downgrades refuse.
- Sends `Authorization: Bearer …` when constructed with a token; `Debug`
  renders `Some(<redacted>)`.
- Accepts only a final status of 200 and caps the body at
  `MAX_BODY_BYTES` (1 GiB), reading one byte past the ceiling to detect
  overflow rather than truncate.

## Errors

| Variant | Raised by |
|---|---|
| `UnsafeRepoPath` | `RepoId::parse`, `Revision::new`, `validate_filename` |
| `HubFetch` | transport failure, non-HTTPS URL, non-200 final status, body too large |
| `HubCache` | root resolution, create/write/rename, unreadable cache |

See [The Hub](../guide/hub.md) for layout, security posture, and the recorded
limitations (no locking, revision-string keying, Windows reserved names).
