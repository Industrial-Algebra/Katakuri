// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Model fetch + content-addressed cache, offline-first (S4).
//!
//! Replaces `hf-hub` as the model-file acquisition path: a revision-keyed,
//! digest-verified local cache with a pluggable transport seam. The default
//! crate gains zero dependencies; this module is behind the `hub` feature, and
//! the batteries-included network transport behind the second `hub-net`
//! feature.
//!
//! # Cache layout
//!
//! Files live under a hub root (`$KATAKURI_HUB_DIR`, else
//! `$HOME/.cache/katakuri/hub`). Content, digest sidecars, and temporary
//! files occupy separate namespaces, so a fetched file can never be mistaken
//! for a sidecar:
//!
//! ```text
//! {root}/{owner}~{model}/{revision}/content/{filename}
//! {root}/{owner}~{model}/{revision}/digests/{filename}
//! ```
//!
//! The sidecar under `digests/` holds the lowercase-hex sha256 of the content
//! plus a trailing newline and is named **exactly** the filename — a fetched
//! file, which may legally be named like anything (including `*.sha256`), can
//! only ever land under `content/`, never in the `digests/` namespace.
//!
//! Every case-bearing path component (owner, model, revision, filename) is
//! case-safe encoded: an ASCII uppercase `X` becomes `^` + `lowercase(X)`, and
//! `^` is outside the segment grammar, so encoded components contain no
//! uppercase letters. Case-insensitive filesystems (macOS/Windows defaults)
//! therefore compare encoded paths exactly as a case-sensitive one would:
//! revisions `V1`/`v1` or names `Config.json`/`config.json` cannot alias.
//!
//! # The digest-on-read guarantee
//!
//! A path returned by [`Hub::get`] held bytes whose sha256 matched the sidecar
//! digest at return time; every subsequent [`Hub::cached`] hit re-hashes the
//! content and compares it to the sidecar. A tampered or truncated cache entry
//! reads as a miss: under [`Policy::CacheFirst`] it is refetched and atomically
//! overwritten. This is the offline-first path's integrity contract — the cache
//! is trusted only as far as its digest.
//!
//! # Revision keying (recorded limitation)
//!
//! The cache key includes the revision string exactly as given. A branch or tag
//! revision such as `main` re-resolves on the network and can therefore name
//! different bytes over time; consumers who need immutability pin a commit sha
//! revision. S4 performs no header-based (for example `x-repo-commit`) pinning.
//!
//! # The transport seam
//!
//! [`Fetch`] is a dependency-free trait. The crate core ships no
//! implementation; `UreqFetcher` (feature `hub-net`) is the
//! batteries-included one, and tests run on fakes. This keeps the hub
//! testable hermetically and keeps the default crate dependency-light. Errors
//! flow through the crate's central [`crate::error::KatakuriError`].

use crate::error::KatakuriError;
use std::fmt;
use std::path::{Path, PathBuf};

/// `owner/model` identity — e.g. `sentence-transformers/all-MiniLM-L6-v2`.
///
/// Strict: one or two non-empty segments; each char must be ASCII alphanumeric
/// or one of `-._` (the Hub's own identifier grammar); no `/`, no `..`, no
/// NUL. Rejects path-traversal inputs at construction (see
/// [`RepoId::parse`]).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepoId {
    owner: String,
    model: String,
}

impl RepoId {
    /// Parses `owner/model` (or a single-segment id such as `gpt2`),
    /// rejecting empty segments, bad characters, and anything that could
    /// escape the cache directory.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] on any invalid form.
    pub fn parse(id: &str) -> Result<Self, KatakuriError> {
        let mut parts = id.split('/');
        let first = parts.next().unwrap_or("");
        let second = parts.next();
        if parts.next().is_some() {
            return Err(unsafe_path(id, "more than one '/' separator"));
        }
        match second {
            Some(model) => {
                validate_segment(first)?;
                validate_segment(model)?;
                Ok(Self {
                    owner: first.to_string(),
                    model: model.to_string(),
                })
            }
            None => {
                validate_segment(first)?;
                Ok(Self {
                    owner: String::new(),
                    model: first.to_string(),
                })
            }
        }
    }

    /// The URL form `{owner}/{model}` (single-segment ids render bare).
    #[must_use]
    pub fn as_url_segment(&self) -> String {
        if self.owner.is_empty() {
            self.model.clone()
        } else {
            format!("{}/{}", self.owner, self.model)
        }
    }

    /// The `{enc(owner)}~{enc(model)}` cache-directory component.
    ///
    /// `~` is outside the segment grammar (ASCII alnum + `-._`), so the
    /// join is injective: distinct [`RepoId`]s can never share a cache
    /// directory. A `--` separator would be forgeable — `m/--x` and
    /// `m--/-x` would both render `m----x` and silently alias two repos'
    /// files (digest-verified, so the confusion would go undetected).
    /// Caught in orchestrator verify against the S4 contract. Each half is
    /// additionally case-safe encoded (see `encode_case_safe`) so that
    /// `Owner/Model` and `owner/model` cannot alias on case-insensitive
    /// filesystems.
    fn cache_dir_component(&self) -> String {
        format!(
            "{}~{}",
            encode_case_safe(&self.owner),
            encode_case_safe(&self.model)
        )
    }
}

impl fmt::Display for RepoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_url_segment())
    }
}

impl std::str::FromStr for RepoId {
    type Err = KatakuriError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// A revision: branch, tag, or 40-hex commit sha — the Hub treats all three
/// identically in `resolve` URLs, so this is a validating newtype.
///
/// Strict: non-empty; ASCII alnum plus `-._` only (a commit sha fits); rejects
/// `/`, `..`, NUL. [`Revision::main`] is the conventional default.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Revision {
    inner: String,
}

impl Revision {
    /// The `main` revision.
    #[must_use]
    pub fn main() -> Self {
        Self {
            inner: "main".to_string(),
        }
    }

    /// Validates and wraps (same grammar as [`RepoId::parse`]).
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] on invalid input.
    pub fn new(rev: &str) -> Result<Self, KatakuriError> {
        validate_segment(rev)?;
        Ok(Self {
            inner: rev.to_string(),
        })
    }

    /// The revision string as used in URLs and cache keys.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

/// A repo filename — e.g. `config.json`, `model.safetensors`.
/// Same strict grammar as the other segments (single segment, no `/`).
///
/// # Errors
/// [`crate::error::KatakuriError::UnsafeRepoPath`] on invalid input.
pub fn validate_filename(name: &str) -> Result<(), KatakuriError> {
    validate_segment(name)
}

/// Validates a single path segment against the strict grammar: non-empty, no
/// `.`/`..`, and every char ASCII alphanumeric or one of `-._`.
fn validate_segment(segment: &str) -> Result<(), KatakuriError> {
    if segment.is_empty() {
        return Err(unsafe_path(segment, "empty path segment"));
    }
    if segment == "." || segment == ".." {
        return Err(unsafe_path(segment, "relative path segment"));
    }
    for (idx, ch) in segment.char_indices() {
        if !(ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')) {
            return Err(unsafe_path(
                segment,
                &format!("disallowed character {ch:?} at byte {idx}"),
            ));
        }
    }
    Ok(())
}

/// Builds the typed rejection for a path segment that fails the grammar.
fn unsafe_path(segment: &str, reason: &str) -> KatakuriError {
    KatakuriError::UnsafeRepoPath {
        detail: format!("{segment:?} rejected: {reason}"),
    }
}

/// Case-safe encoding for a path component: every ASCII uppercase letter `X`
/// becomes `^` + `lowercase(X)`; every other character passes through
/// unchanged.
///
/// `^` is outside the segment grammar (ASCII alnum + `-._`), so the escape is
/// unambiguous. The encoded string contains no uppercase ASCII, so two
/// components differing only by case map to distinct encoded strings that stay
/// distinct on a case-insensitive filesystem — `cache_path` is injective
/// across case aliases (owner, model, revision, filename).
fn encode_case_safe(seg: &str) -> String {
    let mut out = String::with_capacity(seg.len());
    for ch in seg.chars() {
        if ch.is_ascii_uppercase() {
            out.push('^');
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// The transport seam. Implementations fetch `url` over HTTPS
/// (following redirects per HTTP semantics) and return the response body. The
/// hub core ships no implementation — `UreqFetcher` (feature `hub-net`) is
/// the batteries-included one, and tests run on fakes: the trait keeps the
/// crate dep-free and hermetic.
pub trait Fetch: Send + Sync + fmt::Debug {
    /// Fetch exactly this URL and return the full body.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubFetch`] on any transport failure
    /// (DNS, TLS, non-2xx after redirects, body read error, body over
    /// [`MAX_BODY_BYTES`]).
    fn fetch(&self, url: &str) -> Result<Vec<u8>, KatakuriError>;
}

/// Response-body ceiling: 1 GiB (largest IA-relevant model file ≈ 90 MB;
/// 440 MB BERT-base; generous headroom, bounded memory).
pub const MAX_BODY_BYTES: u64 = 1 << 30;

/// Cache-resolution policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Policy {
    /// Cache hit (digest-verified) short-circuits; miss or corrupt falls
    /// through to fetch. The default, and the offline-first path.
    #[default]
    CacheFirst,
    /// Re-fetch even on hit, verify, atomically overwrite the entry.
    AlwaysFetch,
}

/// Revision-keyed, content-addressed model-file cache + fetcher.
///
/// Layout (digest sidecar carries lowercase hex sha256 + `\n`):
///
/// ```text
/// {root}/{owner}~{model}/{revision}/content/{filename}
/// {root}/{owner}~{model}/{revision}/digests/{filename}
/// ```
///
/// Owner, model, revision, and filename are case-safe encoded
/// (`encode_case_safe`): uppercase ASCII becomes `^` + lowercase, so encoded
/// components contain no uppercase and cannot alias on case-insensitive
/// filesystems.
///
/// Guarantee: a path returned by [`Hub::get`] held bytes whose sha256 matched
/// the sidecar digest at return time, and every subsequent [`Hub::cached`] hit
/// re-verifies. A tampered cache entry reads as a miss (corrupt → refetch under
/// [`Policy::CacheFirst`]).
pub struct Hub {
    root: PathBuf,
    fetch: std::sync::Arc<dyn Fetch>,
}

impl Hub {
    /// Builds a hub rooted at `root` with transport `fetch`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, fetch: std::sync::Arc<dyn Fetch>) -> Self {
        Self {
            root: root.into(),
            fetch,
        }
    }

    /// Default root: `$KATAKURI_HUB_DIR` if set, else
    /// `$HOME/.cache/katakuri/hub` (std-only; no `dirs` dep). Errors if
    /// neither env var yields a usable non-empty path.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubCache`] when no root is derivable.
    pub fn with_default_root(fetch: std::sync::Arc<dyn Fetch>) -> Result<Self, KatakuriError> {
        let root = std::env::var_os("KATAKURI_HUB_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|home| PathBuf::from(home).join(".cache/katakuri/hub"))
            })
            .ok_or_else(|| KatakuriError::HubCache {
                detail: "no hub cache root: set KATAKURI_HUB_DIR or HOME".to_string(),
            })?;
        Ok(Self::new(root, fetch))
    }

    /// Resolves a repo file under [`Policy::CacheFirst`]: digest-verified
    /// cache hit → path; miss → fetch → verify → atomic write → path.
    /// Returns the LOCAL path (never the URL).
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubFetch`] (transport),
    /// [`crate::error::KatakuriError::HubCache`] (filesystem),
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] (should be impossible
    /// with validated inputs — defensive).
    pub fn get(
        &self,
        repo: &RepoId,
        rev: &Revision,
        filename: &str,
    ) -> Result<PathBuf, KatakuriError> {
        self.get_with(repo, rev, filename, Policy::CacheFirst)
    }

    /// As [`Hub::get`], with an explicit policy.
    ///
    /// # Errors
    /// [`crate::error::KatakuriError::HubFetch`] (transport),
    /// [`crate::error::KatakuriError::HubCache`] (filesystem),
    /// [`crate::error::KatakuriError::UnsafeRepoPath`] (defensive).
    pub fn get_with(
        &self,
        repo: &RepoId,
        rev: &Revision,
        filename: &str,
        policy: Policy,
    ) -> Result<PathBuf, KatakuriError> {
        validate_filename(filename)?;
        let path = self.cache_path(repo, rev, filename);

        let cached_hit = if policy == Policy::CacheFirst {
            self.cached(repo, rev, filename)
        } else {
            None
        };
        if let Some(hit) = cached_hit {
            return Ok(hit);
        }

        let url = resolve_url(repo, rev, filename);
        let bytes = self.fetch.fetch(&url)?;
        let digest = sha256_hex(&bytes);

        write_atomic(&path, &bytes)?;
        let digest_line = format!("{digest}\n");
        write_atomic(&sidecar_path(&path), digest_line.as_bytes())?;

        Ok(path)
    }

    /// Pure offline lookup: digest-verified cache hit → `Some(path)`; miss,
    /// corrupt, or unreadable → `None`. Never fetches, never errors.
    #[must_use]
    pub fn cached(&self, repo: &RepoId, rev: &Revision, filename: &str) -> Option<PathBuf> {
        validate_filename(filename).ok()?;
        let path = self.cache_path(repo, rev, filename);
        let expected = std::fs::read_to_string(sidecar_path(&path)).ok()?;
        let expected = expected.trim();
        if expected.is_empty() {
            return None;
        }
        let bytes = std::fs::read(&path).ok()?;
        (sha256_hex(&bytes) == expected).then_some(path)
    }

    /// The cache path a file would occupy (parent dirs created on demand only
    /// in the fetch path; this method is a pure join + validation).
    ///
    /// The `filename` parameter is debug-asserted against
    /// [`validate_filename`]; callers that can receive untrusted names should
    /// validate first (the fallible methods do).
    #[must_use]
    pub fn cache_path(&self, repo: &RepoId, rev: &Revision, filename: &str) -> PathBuf {
        debug_assert!(
            validate_filename(filename).is_ok(),
            "cache_path called with an invalid filename: {filename:?}"
        );
        self.root
            .join(repo.cache_dir_component())
            .join(encode_case_safe(rev.as_str()))
            .join("content")
            .join(encode_case_safe(filename))
    }
}

/// The Hugging Face `resolve` URL for a repo file.
fn resolve_url(repo: &RepoId, rev: &Revision, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo.as_url_segment(),
        rev.as_str(),
        filename
    )
}

/// The digest-sidecar path for a content path: the sibling `digests/`
/// directory, holding a file named exactly the (encoded) filename — never a
/// suffixed name, so a fetched file cannot shadow a sidecar.
fn sidecar_path(content_path: &Path) -> PathBuf {
    let filename = content_path.file_name().unwrap_or_default();
    let digests = content_path.parent().and_then(Path::parent).map_or_else(
        || PathBuf::from("digests"),
        |rev_dir| rev_dir.join("digests"),
    );
    digests.join(filename)
}

/// The per-process temporary path for an atomic write:
/// `.tmp-{process id}-{filename}` in the **same directory** as the target, so
/// the rename stays atomic. Applied to both content and digest writes.
fn tmp_path(path: &Path) -> PathBuf {
    let mut name = std::ffi::OsString::from(format!(".tmp-{}-", std::process::id()));
    name.push(path.file_name().unwrap_or_default());
    path.parent().unwrap_or_else(|| Path::new(".")).join(name)
}

/// Writes `bytes` to `path` atomically: full write to a sibling temp file,
/// then `rename` onto the target (same directory ⇒ atomic on POSIX and
/// Windows). Parent directories are created on demand.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), KatakuriError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| KatakuriError::HubCache {
            detail: format!("create_dir_all {}: {e}", parent.display()),
        })?;
    }
    let tmp = tmp_path(path);
    std::fs::write(&tmp, bytes).map_err(|e| KatakuriError::HubCache {
        detail: format!("write {}: {e}", tmp.display()),
    })?;
    std::fs::rename(&tmp, path).map_err(|e| KatakuriError::HubCache {
        detail: format!("rename {} -> {}: {e}", tmp.display(), path.display()),
    })
}

/// Digest helper (also used by tests): lowercase hex sha256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest.iter() {
        // Writing to a String is infallible.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Refuses a URL that is not HTTPS. A bearer token (or any model bytes) must
/// never transit plaintext HTTP, so this is checked before the request is
/// built. Pure and unit-tested.
///
/// # Errors
/// [`crate::error::KatakuriError::HubFetch`] naming the offending scheme.
#[cfg_attr(not(feature = "hub-net"), allow(dead_code))]
fn require_https(url: &str) -> Result<(), KatakuriError> {
    if url.starts_with("https://") {
        return Ok(());
    }
    let scheme = url.split_once("://").map_or(url, |(scheme, _)| scheme);
    Err(KatakuriError::HubFetch {
        detail: format!("refusing non-HTTPS URL ({scheme:?}): {url}"),
    })
}

/// Requires the final (post-redirect) response status to be exactly 200.
/// ureq follows redirects itself; Hugging Face's `resolve` URLs answer the
/// last hop with 200, so any other status (a 204, a 304, …) is a failed fetch
/// rather than empty content to be cached. Pure and unit-tested.
///
/// # Errors
/// [`crate::error::KatakuriError::HubFetch`] naming the URL and status.
#[cfg_attr(not(feature = "hub-net"), allow(dead_code))]
fn require_final_status(url: &str, status: u16) -> Result<(), KatakuriError> {
    if status == 200 {
        return Ok(());
    }
    Err(KatakuriError::HubFetch {
        detail: format!("GET {url}: unexpected final status {status}, expected 200"),
    })
}

/// Blocking HTTPS transport over ureq/rustls (feature `hub-net`).
///
/// Follows redirects (Hub `resolve` URLs 307 to a signed CDN link). Optional
/// `Authorization: Bearer` token for gated repos / higher rate limits. The
/// token is never rendered by [`fmt::Debug`].
#[cfg(feature = "hub-net")]
pub struct UreqFetcher {
    token: Option<String>,
    agent: ureq::Agent,
}

#[cfg(feature = "hub-net")]
impl UreqFetcher {
    /// Unauthenticated (public repos; ~3000 requests / 5 min).
    #[must_use]
    pub fn new() -> Self {
        Self {
            token: None,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Authenticated (`HF_TOKEN`-style bearer).
    #[must_use]
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            token: Some(token.into()),
            agent: ureq::Agent::new_with_defaults(),
        }
    }
}

#[cfg(feature = "hub-net")]
impl fmt::Debug for UreqFetcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never render the secret: `Some(<redacted>)` or `None` only.
        let token = self.token.as_ref().map_or("None", |_| "Some(<redacted>)");
        f.debug_struct("UreqFetcher")
            .field("token", &format_args!("{token}"))
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "hub-net")]
impl Default for UreqFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hub-net")]
impl Fetch for UreqFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, KatakuriError> {
        use std::io::Read as _;

        require_https(url)?;

        let mut request = self.agent.get(url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let response = request.call().map_err(|e| KatakuriError::HubFetch {
            detail: format!("GET {url}: {e}"),
        })?;
        require_final_status(url, response.status().as_u16())?;

        // Read one byte past the ceiling so an over-limit body is detected
        // rather than silently truncated.
        let mut reader = response.into_body().into_reader().take(MAX_BODY_BYTES + 1);
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .map_err(|e| KatakuriError::HubFetch {
                detail: format!("read body from {url}: {e}"),
            })?;
        if buf.len() as u64 > MAX_BODY_BYTES {
            return Err(KatakuriError::HubFetch {
                detail: format!("body from {url} exceeds MAX_BODY_BYTES ({MAX_BODY_BYTES})"),
            });
        }
        Ok(buf)
    }
}

#[cfg(all(test, feature = "hub"))]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Fake transport that counts calls and records URLs, with a mutable body.
    #[derive(Debug)]
    struct CountingFetcher {
        calls: AtomicUsize,
        body: Mutex<Vec<u8>>,
        urls: Mutex<Vec<String>>,
    }

    impl CountingFetcher {
        fn new(body: Vec<u8>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                body: Mutex::new(body),
                urls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        fn set_body(&self, body: Vec<u8>) {
            *self.body.lock().unwrap() = body;
        }

        fn last_url(&self) -> Option<String> {
            self.urls.lock().unwrap().last().cloned()
        }
    }

    impl Fetch for CountingFetcher {
        fn fetch(&self, url: &str) -> Result<Vec<u8>, KatakuriError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.urls.lock().unwrap().push(url.to_string());
            Ok(self.body.lock().unwrap().clone())
        }
    }

    /// Fake transport that panics: proves offline paths never fetch.
    #[derive(Debug)]
    struct PanickyFetcher;

    impl Fetch for PanickyFetcher {
        fn fetch(&self, _url: &str) -> Result<Vec<u8>, KatakuriError> {
            panic!("fetch must not be called on the offline path");
        }
    }

    /// Fake transport that always errors.
    #[derive(Debug)]
    struct FailingFetcher;

    impl Fetch for FailingFetcher {
        fn fetch(&self, url: &str) -> Result<Vec<u8>, KatakuriError> {
            Err(KatakuriError::HubFetch {
                detail: format!("forced failure for {url}"),
            })
        }
    }

    /// A unique, empty temp root per test.
    fn temp_root(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "katakuri-hub-test-{}-{tag}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn repo_id_grammar() {
        let good = RepoId::parse("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        assert_eq!(
            good.as_url_segment(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        let single = RepoId::parse("gpt2").unwrap();
        assert_eq!(single.as_url_segment(), "gpt2");

        for bad in [
            "",
            "a/",
            "/b",
            "a/b/c",
            "../evil",
            "a/..",
            "a/.",
            "a/ b",
            "a/b\u{00e9}",
            "a/b\0",
            "a\\b",
        ] {
            assert!(RepoId::parse(bad).is_err(), "should reject {bad:?}");
        }

        let parsed: RepoId = "sentence-transformers/all-MiniLM-L6-v2".parse().unwrap();
        assert_eq!(parsed, good);
        assert_eq!(parsed.to_string(), "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(single.to_string(), "gpt2");
    }

    #[test]
    fn revision_grammar() {
        assert_eq!(Revision::main().as_str(), "main");
        assert!(Revision::new("main").is_ok());
        assert!(Revision::new("1110a2433f0e7f2b4c5d6e7f8a9b0c1d2e3f4a5b").is_ok());
        assert!(Revision::new("v1.0-tags").is_ok());

        for bad in ["", "a/b", "..", ".", "v1/2", "\0", "a b"] {
            assert!(Revision::new(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn filename_grammar() {
        for good in ["config.json", "model.safetensors", "vocab.txt"] {
            assert!(validate_filename(good).is_ok(), "should accept {good:?}");
        }
        for bad in ["a/b", "..", "", ".", "\0", "a b"] {
            assert!(validate_filename(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn sha256_hex_known_vector() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn get_fetches_then_caches() {
        let root = temp_root("get");
        let fetcher = std::sync::Arc::new(CountingFetcher::new(b"hello hub".to_vec()));
        let hub = Hub::new(&root, fetcher.clone());
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::main();

        let path = hub.get(&repo, &rev, "config.json").unwrap();
        assert_eq!(fetcher.calls(), 1);
        assert_eq!(std::fs::read(&path).unwrap(), b"hello hub");
        let sidecar = sidecar_path(&path);
        assert_eq!(
            std::fs::read_to_string(&sidecar).unwrap(),
            format!("{}\n", sha256_hex(b"hello hub"))
        );

        let again = hub.get(&repo, &rev, "config.json").unwrap();
        assert_eq!(again, path);
        assert_eq!(fetcher.calls(), 1, "second get must be a cache hit");
    }

    #[test]
    fn tampered_entry_reads_as_miss_and_refetches() {
        let root = temp_root("tamper");
        let fetcher = std::sync::Arc::new(CountingFetcher::new(b"good bytes".to_vec()));
        let hub = Hub::new(&root, fetcher.clone());
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::main();

        let path = hub.get(&repo, &rev, "config.json").unwrap();
        assert_eq!(fetcher.calls(), 1);

        std::fs::write(&path, b"tampered!").unwrap();
        assert_eq!(hub.cached(&repo, &rev, "config.json"), None);

        let again = hub.get(&repo, &rev, "config.json").unwrap();
        assert_eq!(again, path);
        assert_eq!(fetcher.calls(), 2, "corrupt entry must trigger refetch");
        assert_eq!(std::fs::read(&path).unwrap(), b"good bytes");
        assert_eq!(
            std::fs::read_to_string(sidecar_path(&path)).unwrap().trim(),
            sha256_hex(b"good bytes")
        );
    }

    #[test]
    fn always_fetch_overwrites() {
        let root = temp_root("always");
        let fetcher = std::sync::Arc::new(CountingFetcher::new(b"v1".to_vec()));
        let hub = Hub::new(&root, fetcher.clone());
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::main();

        let path = hub.get(&repo, &rev, "config.json").unwrap();
        assert_eq!(fetcher.calls(), 1);

        fetcher.set_body(b"v2".to_vec());
        let again = hub
            .get_with(&repo, &rev, "config.json", Policy::AlwaysFetch)
            .unwrap();
        assert_eq!(again, path);
        assert_eq!(fetcher.calls(), 2);
        assert_eq!(std::fs::read(&path).unwrap(), b"v2");
        assert_eq!(
            std::fs::read_to_string(sidecar_path(&path)).unwrap().trim(),
            sha256_hex(b"v2")
        );
    }

    #[test]
    fn cached_is_pure_offline() {
        let root = temp_root("offline");
        let good = std::sync::Arc::new(CountingFetcher::new(b"payload".to_vec()));
        let hub = Hub::new(&root, good);
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::main();
        let path = hub.get(&repo, &rev, "config.json").unwrap();

        // Same root, a transport that panics if ever used.
        let offline = Hub::new(&root, std::sync::Arc::new(PanickyFetcher));
        assert_eq!(
            offline.cached(&repo, &rev, "config.json"),
            Some(path.clone())
        );
        assert_eq!(offline.cached(&repo, &rev, "missing.bin"), None);

        // Offline `get` on a warm entry must also never call the transport.
        assert_eq!(
            offline.get(&repo, &rev, "config.json").unwrap(),
            path.clone()
        );

        std::fs::write(&path, b"corrupt").unwrap();
        assert_eq!(offline.cached(&repo, &rev, "config.json"), None);
    }

    #[test]
    fn cache_path_layout() {
        let hub = Hub::new(
            PathBuf::from("/tmp/katakuri-hub-layout"),
            std::sync::Arc::new(PanickyFetcher),
        );
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::new("v1.0-tags").unwrap();
        let path = hub.cache_path(&repo, &rev, "config.json");
        let sep = std::path::MAIN_SEPARATOR;
        assert!(
            path.to_string_lossy().ends_with(&format!(
                "owner~model{sep}v1.0-tags{sep}content{sep}config.json"
            )),
            "unexpected cache path: {}",
            path.display()
        );

        // The digest sidecar lives in a sibling namespace, same filename.
        let digest = sidecar_path(&path);
        assert!(
            digest.to_string_lossy().ends_with(&format!(
                "owner~model{sep}v1.0-tags{sep}digests{sep}config.json"
            )),
            "unexpected sidecar path: {}",
            digest.display()
        );

        // Case is encoded away in every component.
        let upper = hub.cache_path(
            &RepoId::parse("Owner/Model").unwrap(),
            &Revision::new("V1.0-Tags").unwrap(),
            "Config.json",
        );
        assert!(
            upper.to_string_lossy().ends_with(&format!(
                "^owner~^model{sep}^v1.0-^tags{sep}content{sep}^config.json"
            )),
            "unexpected encoded cache path: {}",
            upper.display()
        );
    }

    #[test]
    fn cache_dirs_are_injective_across_repos() {
        // The `~` separator is outside the segment grammar, so distinct
        // RepoIds can never share a cache directory. A `--` separator
        // was forgeable: `m/--x` and `m--/-x` would both render `m----x`,
        // silently aliasing two repos' digest-verified files
        // (orchestrator verify against the S4 contract).
        let hub = Hub::new(
            PathBuf::from("/tmp/katakuri-hub-injective"),
            std::sync::Arc::new(PanickyFetcher),
        );
        let a = RepoId::parse("m/--x").unwrap();
        let b = RepoId::parse("m--/-x").unwrap();
        let rev = Revision::main();
        assert_ne!(
            hub.cache_path(&a, &rev, "config.json"),
            hub.cache_path(&b, &rev, "config.json")
        );

        // Single-segment ids never collide with two-segment ones either.
        let single = RepoId::parse("gpt2").unwrap();
        let double = RepoId::parse("~/gpt2").map(|r| r.to_string());
        assert!(double.is_err(), "~ is not a valid segment character");
        assert!(
            hub.cache_path(&single, &rev, "config.json")
                .to_string_lossy()
                .contains("~gpt2")
        );
    }

    #[test]
    fn sidecar_namespaces_separated() {
        let root = temp_root("sidecar-ns");
        let fetcher = std::sync::Arc::new(CountingFetcher::new(b"B1".to_vec()));
        let hub = Hub::new(&root, fetcher.clone());
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::main();

        let x = hub.get(&repo, &rev, "x").unwrap();
        assert_eq!(std::fs::read(&x).unwrap(), b"B1");
        assert_eq!(x.parent().unwrap().file_name().unwrap(), "content");
        let x_digest = sidecar_path(&x);
        assert_eq!(x_digest.parent().unwrap().file_name().unwrap(), "digests");
        assert_eq!(x_digest.file_name().unwrap(), "x");
        assert_eq!(
            std::fs::read_to_string(&x_digest).unwrap().trim(),
            sha256_hex(b"B1")
        );

        // A fetched file NAMED like a sidecar still lands under `content/`,
        // never in the `digests/` namespace that sidecars occupy.
        fetcher.set_body(b"B2".to_vec());
        let fake_sidecar_named = hub.get(&repo, &rev, "x.sha256").unwrap();
        assert_eq!(std::fs::read(&fake_sidecar_named).unwrap(), b"B2");
        assert_eq!(
            fake_sidecar_named.parent().unwrap().file_name().unwrap(),
            "content"
        );
        assert_eq!(fake_sidecar_named.file_name().unwrap(), "x.sha256");
        assert_ne!(fake_sidecar_named, x_digest);

        // The original entry's digest is untouched: `get("x")` still verifies
        // and serves the original bytes (no fetch, it is a digest hit).
        assert_eq!(
            std::fs::read_to_string(&x_digest).unwrap().trim(),
            sha256_hex(b"B1")
        );
        let calls_before = fetcher.calls();
        assert_eq!(hub.get(&repo, &rev, "x").unwrap(), x);
        assert_eq!(std::fs::read(&x).unwrap(), b"B1");
        assert_eq!(fetcher.calls(), calls_before, "get(\"x\") must be a hit");

        // `digests` is itself a legal segment: fetching it targets
        // `content/digests`, never the namespace directory.
        fetcher.set_body(b"B3".to_vec());
        let d = hub.get(&repo, &rev, "digests").unwrap();
        assert_eq!(d.parent().unwrap().file_name().unwrap(), "content");
        assert_eq!(d.file_name().unwrap(), "digests");
        assert!(d.is_file());
        assert!(!d.is_dir());
        assert_eq!(
            d,
            x.parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("content")
                .join("digests")
        );
    }

    #[test]
    fn case_safe_paths_injective() {
        let hub = Hub::new(
            PathBuf::from("/tmp/katakuri-hub-case"),
            std::sync::Arc::new(PanickyFetcher),
        );
        let rev = Revision::main();

        // Owner/model case pair.
        let upper_repo = RepoId::parse("Owner/Model").unwrap();
        let lower_repo = RepoId::parse("owner/model").unwrap();
        assert_ne!(
            hub.cache_path(&upper_repo, &rev, "config.json"),
            hub.cache_path(&lower_repo, &rev, "config.json")
        );

        // Revision + filename case pair.
        let upper_path = hub.cache_path(&upper_repo, &Revision::new("V1").unwrap(), "Config.json");
        let lower_path = hub.cache_path(&upper_repo, &Revision::new("v1").unwrap(), "config.json");
        assert_ne!(upper_path, lower_path);

        // No uppercase survives encoding, and `^` marks every uppercase that was.
        let rendered = upper_path.to_string_lossy();
        assert!(
            !rendered.chars().any(|c| c.is_ascii_uppercase()),
            "encoded path retains uppercase: {rendered}"
        );
        assert!(rendered.contains("^owner"), "owner not encoded: {rendered}");
        assert!(rendered.contains("^model"), "model not encoded: {rendered}");
        assert!(rendered.contains("^v1"), "revision not encoded: {rendered}");
        assert!(
            rendered.contains("^config.json"),
            "filename not encoded: {rendered}"
        );
    }

    #[cfg(feature = "hub-net")]
    #[test]
    fn bearer_token_redacted_in_debug() {
        let rendered = format!("{:?}", UreqFetcher::with_token("hf-secret-xyz"));
        assert!(
            rendered.contains("redacted"),
            "Debug did not redact the token: {rendered}"
        );
        assert!(
            !rendered.contains("hf-secret-xyz"),
            "Debug leaked the bearer token: {rendered}"
        );
    }

    #[test]
    fn require_final_status_matrix() {
        assert!(require_final_status("https://x/y", 200).is_ok());
        for status in [204u16, 301, 304, 404, 500] {
            let err = require_final_status("https://x/y", status).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains(&status.to_string()),
                "error for {status} did not name the status: {msg:?}"
            );
        }
    }

    #[test]
    fn require_https_rejects_plain_http() {
        assert!(require_https("http://x").is_err());
        assert!(require_https("https://x").is_ok());
        // Scheme prefix is case-sensitive: `HTTP://` is not `https://`.
        assert!(require_https("HTTP://x").is_err());
        assert!(require_https("ftp://x").is_err());
    }

    #[test]
    fn fetch_failure_leaves_cache_intact() {
        let root = temp_root("intact");
        let good = std::sync::Arc::new(CountingFetcher::new(b"keep me".to_vec()));
        let hub = Hub::new(&root, good);
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::main();
        let path = hub.get(&repo, &rev, "config.json").unwrap();

        let failing = Hub::new(&root, std::sync::Arc::new(FailingFetcher));
        let err = failing
            .get_with(&repo, &rev, "config.json", Policy::AlwaysFetch)
            .unwrap_err();
        assert!(matches!(err, KatakuriError::HubFetch { .. }));

        assert_eq!(
            failing.cached(&repo, &rev, "config.json"),
            Some(path.clone())
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"keep me");
    }

    #[test]
    fn url_is_resolve_form() {
        let root = temp_root("url");
        let fetcher = std::sync::Arc::new(CountingFetcher::new(b"x".to_vec()));
        let hub = Hub::new(&root, fetcher.clone());
        let repo = RepoId::parse("owner/model").unwrap();
        let rev = Revision::new("1110a2433f0e7f2b4c5d6e7f8a9b0c1d2e3f4a5b").unwrap();

        hub.get(&repo, &rev, "config.json").unwrap();
        assert_eq!(
            fetcher.last_url().as_deref(),
            Some(
                "https://huggingface.co/owner/model/resolve/\
                 1110a2433f0e7f2b4c5d6e7f8a9b0c1d2e3f4a5b/config.json"
            )
        );
    }
}

/// Kani proof harness: strict repo-id parsing never yields a traversal path.
#[cfg(all(kani, feature = "hub"))]
mod kani_harnesses {
    use super::RepoId;

    /// For an arbitrary bounded byte string that is valid UTF-8, `RepoId::parse`
    /// either errors, or every `/`-split component of the rendered
    /// `as_url_segment()` is neither `.` nor `..` and contains no `/`.
    #[kani::proof]
    fn repo_id_parse_never_yields_traversal() {
        let bytes: [u8; 4] = kani::any();
        let Ok(input) = std::str::from_utf8(&bytes) else {
            return;
        };
        if let Ok(id) = RepoId::parse(input) {
            for component in id.as_url_segment().split('/') {
                assert_ne!(component, "..");
                assert_ne!(component, ".");
                assert!(!component.contains('/'));
            }
        }
    }
}
