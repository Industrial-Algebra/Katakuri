# Katakuri S4 — Hub: fetch + content-addressed cache, offline-first

**Status:** contract for generation dispatch (ia-moment AUTHOR step).
**Branch:** `feature/s4-hub` (from develop `101588d`, post-PR-#4 merge).
**Predecessors:** S1 safetensors, S2 tokenizer, S3 model specs (all merged).

## 1. Goal

Replace `hf-hub` as Ijima's model-file acquisition path: revision-keyed,
content-addressed local cache with digest verification on every read,
offline-first resolution, and a pluggable transport seam. The default
crate gains **zero dependencies**; `hub` is an optional feature, and the
network transport a second optional feature on top.

## 2. Non-goals

- No async runtime, no tokio — this is a blocking, CLI/server-embeddable
  surface (`ijima-server` calls it from sync code today).
- No metadata API (model listing, siblings), no uploads — file fetch only.
- No xet/LFS protocol implementation — the plain HTTPS `resolve` URL
  with redirect-following covers it (observed: 307 → signed CDN URL).
- No header-based revision pinning (S4 subset): cache key includes the
  revision string as given; consumers pin immutability by using a commit
  revision (recorded limitation, §8).
- Katakuri still never computes; no new transitive compute deps.

## 3. Reference intel (observed, this machine, 2026-10-17)

- File URL: `https://huggingface.co/{owner}/{model}/resolve/{revision}/{filename}`
  — revision is branch | tag | 40-hex commit sha; all treated identically.
- First response: **HTTP 307** to `/api/resolve-cache/...`, then one more
  hop to a signed CDN URL (`us.aws.cdn.hf.co`, `Hash-Algorithm=SHA256`).
  A fetcher MUST follow redirects (≤ ~5 hops is plenty; ureq defaults on).
- `x-repo-commit: 1110a243...` on the first hop = the commit `main`
  resolved to (NOT used in S4 — recorded for a future pinning slice).
- Unauthenticated GET works for public repos (rate-limited ~3000/5min);
  optional `Authorization: Bearer {HF_TOKEN}` raises limits / gates models.
- Real fetch smoke-tested this session: `config.json` and
  `model.safetensors` both resolve; MiniLM weights ≈ 90 MB.
- Crate versions (crates.io, this date): `ureq` 3.4.2, `sha2` 0.11.1.

## 4. Cargo.toml (exact additions, ledger entries in the PR)

```toml
[features]
# (existing std/verify unchanged)
hub = ["dep:sha2"]
hub-net = ["hub", "dep:ureq"]

[dependencies]
# S4 hub: content addressing — every cache read is digest-verified.
sha2 = { version = "0.11", optional = true }
# S4 hub-net: blocking HTTPS with rustls; redirect-following; no async
# runtime. Optional — the transport seam (Fetch trait) is dep-free.
ureq = { version = "3.4", optional = true, default-features = false, features = ["rustls"] }
```

(Verify ureq 3.4's exact rustls feature name against the crate's
manifest at implementation; if `rustls` is wrong, use the documented
tls feature and record the delta. `cargo tree -e features -p ureq` must
show no native-tls / no async runtime.)

## 5. Public API (exact) — `src/hub.rs`, all `#[cfg(feature = "hub")]`

```rust
//! Model fetch + content-addressed cache, offline-first (S4).
//!
//! ... (module docs: replace the placeholder text; crate-root-scoped
//! links — the S1-r2 lesson — e.g. [`crate::error::KatakuriError`];
//! document the cache layout, the digest-on-read guarantee, the
//! revision-keying limitation, and the transport seam.)

use std::fmt;
use std::path::{Path, PathBuf};

/// `owner/model` identity — e.g. `sentence-transformers/all-MiniLM-L6-v2`.
///
/// Strict: both segments non-empty; each char must be ASCII alphanumeric
/// or one of `-._` (the Hub's own identifier grammar); no `/`, no `..`,
/// no NUL. Rejects path-traversal inputs at construction (see
/// [`RepoId::parse`]).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepoId { /* owner, model: String */ }

impl RepoId {
    /// Parses `owner/model`, rejecting empty segments, bad characters,
    /// and anything that could escape the cache directory.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] on any invalid form.
    pub fn parse(id: &str) -> Result<Self, KatakuriError>;

    /// The URL form `{owner}/{model}`.
    #[must_use] pub fn as_url_segment(&self) -> String;
}
impl fmt::Display for RepoId { /* owner/model */ }
impl std::str::FromStr for RepoId { type Err = KatakuriError; /* parse */ }

/// A revision: branch, tag, or 40-hex commit sha — the Hub treats all
/// three identically in `resolve` URLs, so this is a validating newtype.
///
/// Strict: non-empty; ASCII alnum plus `-._` only (a commit sha fits);
/// rejects `/`, `..`, NUL. `Revision::MAIN` is the conventional default.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Revision { /* inner: String */ }

impl Revision {
    /// The `main` revision.
    #[must_use] pub fn main() -> Self;
    /// Validates and wraps (same grammar as [`RepoId::parse`]).
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] on invalid input.
    pub fn new(rev: &str) -> Result<Self, KatakuriError>;
    /// The revision string as used in URLs and cache keys.
    #[must_use] pub fn as_str(&self) -> &str;
}

/// A repo filename — e.g. `config.json`, `model.safetensors`.
/// Same strict grammar as the other segments (single segment, no `/`).
///
/// # Errors
/// [`crate::error::KatakuriError::UnsafeRepoPath`] on invalid input.
pub fn validate_filename(name: &str) -> Result<(), KatakuriError>;

/// The transport seam. Implementations fetch `url` over HTTPS
/// (following redirects per HTTP semantics) and return the response
/// body. The hub core ships no implementation — [`UreqFetcher`]
/// (feature `hub-net`) is the batteries-included one, and tests run on
/// fakes: the trait keeps the crate dep-free and hermetic.
pub trait Fetch: Send + Sync + fmt::Debug {
    /// Fetch exactly this URL and return the full body.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubFetch`] on any transport
    /// failure (DNS, TLS, non-2xx after redirects, body read error,
    /// body over [`MAX_BODY_BYTES`]).
    fn fetch(&self, url: &str) -> Result<Vec<u8>, KatakuriError>;
}

/// Response-body ceiling: 1 GiB (largest IA-relevant model file ≈ 90 MB;
/// 440 MB BERT-base; generous headroom, bounded memory).
pub const MAX_BODY_BYTES: u64 = 1 << 30;

/// Cache-resolution policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Policy {
    /// Cache hit (digest-verified) short-circuits; miss or corrupt
    /// falls through to fetch. The default, and the offline-first path.
    #[default]
    CacheFirst,
    /// Re-fetch even on hit, verify, atomically overwrite the entry.
    AlwaysFetch,
}

/// Revision-keyed, content-addressed model-file cache + fetcher.
///
/// Layout (digest file carries lowercase hex sha256 + `\n`):
///
/// ```text
/// {root}/{owner}--{model}/{revision}/{filename}
/// {root}/{owner}--{model}/{revision}/{filename}.sha256
/// ```
///
/// Guarantee: a path returned by [`Hub::get`] held bytes whose sha256
/// matched the sidecar digest at return time, and every subsequent
/// [`Hub::cached`] hit re-verifies. A tampered cache entry reads as a
/// miss (corrupt → refetch under [`Policy::CacheFirst`]).
pub struct Hub { /* root: PathBuf, fetch: std::sync::Arc<dyn Fetch> */ }

impl Hub {
    /// Builds a hub rooted at `root` with transport `fetch`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, fetch: std::sync::Arc<dyn Fetch>) -> Self;

    /// Default root: `$KATAKURI_HUB_DIR` if set, else
    /// `$HOME/.cache/katakuri/hub` (std-only; no `dirs` dep). Errors if
    /// neither env var yields a usable non-empty path.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubCache`] when no root is derivable.
    pub fn with_default_root(fetch: std::sync::Arc<dyn Fetch>) -> Result<Self, KatakuriError>;

    /// Resolves a repo file under [`Policy::CacheFirst`]: digest-verified
    /// cache hit → path; miss → fetch → verify → atomic write → path.
    /// Returns the LOCAL path (never the URL).
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubFetch`] (transport),
    /// [`crate::error::KatakuriError::HubCache`] (filesystem),
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] (should be
    /// impossible with validated inputs — defensive).
    pub fn get(&self, repo: &RepoId, rev: &Revision, filename: &str)
        -> Result<PathBuf, KatakuriError>;

    /// As [`Hub::get`], with an explicit policy.
    pub fn get_with(&self, repo: &RepoId, rev: &Revision, filename: &str,
        policy: Policy) -> Result<PathBuf, KatakuriError>;

    /// Pure offline lookup: digest-verified cache hit → `Some(path)`;
    /// miss, corrupt, or unreadable → `None`. Never fetches, never errors.
    #[must_use]
    pub fn cached(&self, repo: &RepoId, rev: &Revision, filename: &str)
        -> Option<PathBuf>;

    /// The cache path a file would occupy (parent dirs created on demand
    /// only in the fetch path; this method is a pure join + validation).
    #[must_use]
    pub fn cache_path(&self, repo: &RepoId, rev: &Revision, filename: &str)
        -> PathBuf;
}

/// Digest helper (also used by tests): lowercase hex sha256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String;
```

`UreqFetcher` (in the same file, `#[cfg(feature = "hub-net")]`):

```rust
/// Blocking HTTPS transport over ureq/rustls (feature `hub-net`).
///
/// Follows redirects (Hub `resolve` URLs 307 to a signed CDN link).
/// Optional `Authorization: Bearer` token for gated repos / higher
/// rate limits.
#[derive(Debug)]
pub struct UreqFetcher { /* token: Option<String>, agent: ureq::Agent */ }

impl UreqFetcher {
    /// Unauthenticated (public repos; ~3000 requests / 5 min).
    #[must_use]
    pub fn new() -> Self;
    /// Authenticated (`HF_TOKEN`-style bearer).
    #[must_use]
    pub fn with_token(token: impl Into<String>) -> Self;
}
impl Default for UreqFetcher { /* new() */ }
impl Fetch for UreqFetcher { /* agent.get(url), optional bearer header,
    status check, body read capped at MAX_BODY_BYTES via
    `Read::take`, map errors to HubFetch */ }
```

Implementation notes (binding):
- Atomic write: write `{path}.tmp-{std::process::id()}` then
  `std::fs::rename` (same directory ⇒ atomic on POSIX and Windows);
  the digest sidecar written the same way AFTER the content file.
- Digest-of-download recorded at fetch time; every `cached()`/`get()`
  cache hit re-hashes the content file and compares to the sidecar —
  mismatch = miss (delete nothing; refetch overwrites).
- `get_with(AlwaysFetch)`: fetch, write, then return the path; on fetch
  failure the previous good entry stays intact (surface the error).
- No file locking in S4 (single-process use today); recorded limitation.

## 6. Errors (`src/error.rs`, additions — match existing style)

```rust
    /// A repo id, revision, or filename fails the strict path-safety
    /// grammar (empty segment, `/`, `..`, NUL, or a character outside
    /// ASCII alnum + `-._`).
    #[error("unsafe repo path: {detail}")]
    UnsafeRepoPath {
        /// What was rejected and why.
        detail: String,
    },

    /// Transport failure while fetching a model file (DNS, TLS, HTTP
    /// status, body read, size cap).
    #[error("hub fetch failed: {detail}")]
    HubFetch {
        /// Transport-level description.
        detail: String,
    },

    /// Local cache failure (unusable root, IO error, corrupt entry that
    /// cannot be repaired by refetch).
    #[error("hub cache error: {detail}")]
    HubCache {
        /// Cache-level description.
        detail: String,
    },
```

## 7. lib.rs

`#[cfg(feature = "hub")] pub mod hub;` replaces the unconditional
placeholder declaration (check how `hub` is currently declared; if the
module was unconditional, gate it and update the crate-docs bullet to
"optional `hub` feature: fetch + content-addressed cache, offline-first;
`hub-net` adds the ureq/rustls transport"). Note: `--all-features` CI
builds compile the gated module — both doc gates must run with
`--features hub-net` added to the existing matrix (see §10).

## 8. Port-plan addendum (Ijima swap boundary, extends S3 §8)

After S4: replace the three `hf-hub` downloads in
`embeddings_candle.rs::load` with
`Hub::with_default_root(Arc::new(UreqFetcher::with_token(HF_TOKEN)))`
and three `get()` calls (`config.json`, `tokenizer.json`,
`model.safetensors`), then feed the local paths into
`BertSpec::from_config_json` / `Tokenizer::from_tokenizer_json` /
`SafeTensors::parse` + `validate_weights` as S3 specified. Pin the
revision to the commit sha (via `IJIMA_EMBED_REVISION`) for reproducible
embeddings; `main` works but re-resolves (S4 cache keys the revision
string as given — no header-based sha pinning yet, §2). Offline-first:
with the cache warm, no network is touched (`Policy::CacheFirst`).

## 9. Test plan

`src/hub.rs` `#[cfg(all(test, feature = "hub"))] mod tests` — hermetic,
network-free, via a `CountingFetcher { calls: AtomicUsize, body: Vec<u8> }`
fake (also a `PanickyFetcher` whose `fetch` panics — proves offline paths
never call it):

1. `repo_id_grammar` — parse accepts
   `sentence-transformers/all-MiniLM-L6-v2`, `gpt2`; rejects `""`,
   `a/`, `/b`, `a/b/c`, `../evil`, `a/..`, embedded `/` in segments,
   non-ASCII, NUL. `FromStr` + `Display` round-trip.
2. `revision_grammar` — `main`, a 40-hex sha, `v1.0-tags` accept;
   `""`, `a/b`, `..`, NUL reject. `Revision::main().as_str() == "main"`.
3. `filename_grammar` — `config.json`, `model.safetensors` accept;
   `a/b`, `..`, `` reject (via `validate_filename`).
4. `sha256_hex_known_vector` — `sha256_hex(b"")` ==
   `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
   and one more known vector (`b"abc"` =
   `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`).
5. `get_fetches_then_caches` — fresh tempdir root (std::env::temp_dir +
   unique suffix via process id + counter): first `get()` → fetch count
   1, file exists, bytes match, sidecar digest == independent sha256;
   second `get()` → count still 1 (cache hit).
6. `tampered_entry_reads_as_miss_and_refetches` — corrupt the content
   file (write different bytes) → next `get()` refetches (count
   increments) and restores correct bytes + digest.
7. `always_fetch_overwrites` — change the fake's body, `AlwaysFetch` →
   bytes updated, digest updated, fetch count increments.
8. `cached_is_pure_offline` — after one `get()`, `PanckyFetcher` hub:
   `cached()` → `Some(path)`; unknown file → `None`; corrupt content →
   `None`. No panic (fetch never called).
9. `cache_path_layout` — `cache_path(repo, rev, "config.json")` ends
   with `owner--model/revision/config.json` (platform separator).
10. `fetch_failure_leaves_cache_intact` — fake that errors after a good
    entry exists: `AlwaysFetch` fails, previous entry still verifies via
    `cached()`.
11. `url_is_resolve_form` (needs a recording fake capturing the URL):
    `get()` requests exactly
    `https://huggingface.co/{owner}/{model}/resolve/{rev}/{file}`.

Env-gated `tests/hub_net.rs` (`#[cfg(feature = "hub-net")]`,
`#[ignore]`, gated on `KATAKURI_TEST_HUB=1`):

12. `real_fetch_minilm_config` — `UreqFetcher::new()`, tempdir root,
    `get(minilm repo, main, "config.json")`; assert bytes parse via
    `BertSpec::from_config_json` (S3 interop: hidden_size 384) — the
    full hub→spec loop. Second `get()` serves from cache with the
    network pulled (swap in `PanckyFetcher` hub on the same root).

Kani (`#[cfg(all(kani, feature = "hub"))]`, bottom of hub.rs):

13. `repo_id_parse_never_yields_traversal`: for arbitrary `&str` input,
    `RepoId::parse` either errs, or the rendered
    `as_url_segment()`/cache components contain no `..` segment and no
    `/` inside either component. Keep the harness tight (symbolic
    string length bounded, e.g. `kani::any()` over a fixed-size buffer).

Expected counts: 11 unit tests + 1 env-gated integration. Lib default
UNCHANGED (60 — hub is feature-gated); `--features hub` ≈ 71 (60 + 11);
`--features hub-net` same 71 + integration 1 ignored.

## 10. Gate matrix (outputs piped)

```
cargo test -p katakuri                                  # 60 lib unchanged
cargo test -p katakuri --features verify                # 61
cargo test -p katakuri --features hub                   # ~71
cargo test -p katakuri --features hub-net               # ~71 (+1 ignored)
cargo test -p katakuri --all-features                   # resolves, green
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps --features verify
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps --features hub-net
# env-gated (this session):
KATAKURI_TEST_HUB=1 cargo test -p katakuri --features hub-net --test hub_net -- --ignored
# dependency audit (no silent growth):
cargo tree -e features -p ureq | head -30    # rustls only, no native-tls, no async
cargo tree -p sha2 | head -10
```

## 11. Ambiguity protocol

Resolve toward strictness + hermeticity; document each resolution in the
completion report. If ureq 3.4's rustls feature name or agent API
differs from §4/§5, adapt to the real crate and record the delta. If
`--all-features` pulls `verify` + `hub-net` together and any lint
interaction appears, fix on the feature-combination matrix, never by
weakening a gate.
