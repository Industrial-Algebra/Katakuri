# The Hub: Fetch & Cache

`katakuri::hub` (feature `hub`) replaces `hf-hub` as the model-file
acquisition path: a revision-keyed, digest-verified local cache with a
pluggable transport seam. `hub-net` adds the blocking ureq/rustls transport.

## Resolving a file

```rust
use katakuri::hub::{Hub, RepoId, Revision, UreqFetcher};
use std::sync::Arc;

let repo = RepoId::parse("sentence-transformers/all-MiniLM-L6-v2")?;
let rev = Revision::main();
let hub = Hub::with_default_root(Arc::new(UreqFetcher::new()))?;

let config_path = hub.get(&repo, &rev, "config.json")?;
let config = std::fs::read_to_string(config_path)?;
```

`get` resolves a digest-verified cache hit, or fetches, verifies, writes
atomically, and returns the **local path** — never a URL. The URL used is the
Hub `resolve` form (`https://huggingface.co/{repo}/resolve/{rev}/{filename}`).

## Policies

```rust
use katakuri::hub::Policy;

// CacheFirst (default): verified hit short-circuits; miss or corrupt fetches.
let path = hub.get_with(&repo, &rev, "config.json", Policy::CacheFirst)?;

// AlwaysFetch: re-fetch even on a hit, verify, atomically overwrite.
let path = hub.get_with(&repo, &rev, "config.json", Policy::AlwaysFetch)?;
```

Offline lookups are a separate, infallible call:

```rust
if let Some(path) = hub.cached(&repo, &rev, "config.json") {
    // digest-verified hit — no network, no error path
}
```

`cached` returns `None` on a miss, a corrupt entry, or an unreadable one. It
never fetches and never errors, which is what makes the offline-first story
testable: warm the cache, then swap in a transport that panics if touched.

## Cache root

`$KATAKURI_HUB_DIR`, else `$HOME/.cache/katakuri/hub`. Override per-process:

```sh
KATAKURI_HUB_DIR=/var/cache/katakuri cargo run --example fetch-and-validate --features hub-net
```

Layout — content, digests, and temp files never share a namespace:

```text
{root}/{owner}~{model}/{revision}/content/{filename}
{root}/{owner}~{model}/{revision}/digests/{filename}
```

The sidecar holds the lowercase-hex sha256 plus a trailing newline and is
named exactly the filename. A fetched file therefore cannot shadow a sidecar,
whatever it is called.

## Tokens

`UreqFetcher::new()` is unauthenticated (public repos; the Hub's anonymous
rate limit is roughly 3000 requests / 5 minutes). For gated repos or higher
limits:

```rust
use katakuri::hub::UreqFetcher;

let fetcher = match std::env::var("HF_TOKEN") {
    Ok(token) if !token.is_empty() => UreqFetcher::with_token(token),
    _ => UreqFetcher::new(),
};
```

The token is sent as `Authorization: Bearer …` and is **never** rendered by
`Debug` (`UreqFetcher` prints `Some(<redacted>)`).

## Security notes

- **HTTPS only, end to end.** The initial URL is checked, and the ureq agent
  is built with `https_only(true)`, so an `https → http` redirect downgrade in
  the middle of the Hub's 307 chain refuses to connect rather than leaking the
  bearer or accepting plaintext bytes. (This was review finding 20: the
  initial-scheme check alone did not cover redirects.)
- **Digest on read.** A path is only returned after its bytes hashed to the
  sidecar digest, and every subsequent `cached` hit re-hashes. Tampering reads
  as a miss and triggers a refetch.
- **Strict path grammar.** Owner, model, revision, and filename are single
  segments of ASCII alphanumerics plus `-._`; `..`, `/`, NUL and everything
  else reject with `UnsafeRepoPath` before any path is built. Path traversal
  is unrepresentable, not filtered.
- **Case-safe encoding.** Uppercase becomes `^` + lowercase in every
  case-bearing component, so `V1`/`v1` and `Config.json`/`config.json` cannot
  alias on a case-insensitive filesystem.
- **Body ceiling.** 1 GiB (`MAX_BODY_BYTES`), read with one byte of headroom
  so an over-limit body is rejected rather than silently truncated.
- **Strict final status.** Only HTTP 200 after redirects is accepted; a 204 or
  304 is a failed fetch, never empty content to be cached.

## Known limitations

Recorded, not hidden:

- **No file locking.** Concurrent writers to the same cache entry are not
  serialized; the atomic rename keeps readers from seeing partial content, but
  two processes may both fetch. Fine for the single-process deployments this
  targets; revisit if a multi-process daemon shares a cache root.
- **Revision-string keying.** The cache key includes the revision exactly as
  given. A branch or tag (`main`) re-resolves and can name different bytes over
  time. Pin a commit sha for reproducibility. S4 performs no header-based
  (e.g. `x-repo-commit`) pinning; that is deferred.
- **Windows reserved device names are open.** The segment grammar admits `con`,
  `nul`, `aux` and friends; those become problematic only if such a character
  device is actually opened on Windows. Not yet rejected.
- **No metadata API.** No model listing, siblings, or uploads — file fetch
  only. No xet/LFS protocol implementation: the plain `resolve` URL with
  redirect-following covers the observed behavior.

## Replacing the transport

`Fetch` is the whole transport contract — one method, no dependencies:

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

That is how the hub's tests run hermetically, and how an embedder plugs in an
existing HTTP client instead of `ureq`.
