# Katakuri S2 — WordPiece Tokenizer (BERT-class)

**Date:** 2026-10-16
**Unit:** S2 (tokenizer) — second implementation slice
**Depends on:** S1 (PR #2, merged; develop @ c789ce8); fixtures already committed on this branch
**Followed by:** S3 (model specs + swap boundary), S4 (hub)
**Plan contract for generation dispatch** (ia-moment AUTHOR stage)

---

## 0. Context for the implementer

Katakuri is IA's model-ecosystem layer over Zunesha. It **never computes**.
S2 implements WordPiece (BERT-class) tokenization as a **bespoke, auditable
engine** replacing the `tokenizers` crate dependency for Ijima's
`embeddings_candle.rs` surface. The acceptance bar is **byte-identical token
id sequences** against the HF Rust `tokenizers` crate (0.21.4 — the exact
crate and version Ijima pins), proven by committed golden vectors generated
from the real `sentence-transformers/all-MiniLM-L6-v2` tokenizer.

### The pipeline (transcribed from the reference implementation)

`tokenizers-0.21.4` sources were read directly; the contract below is a
faithful transcription, not a re-derivation:

1. **Added-token split** (raw input, before normalization): registered added
   tokens (all MiniLM ones are `normalized: false`) are matched against the
   ORIGINAL text — leftmost position first, longest content at that position
   wins. Matched segments emit their id directly; the gaps flow on.
2. **BertNormalizer**, in this exact order, each stage behind its flag:
   - `clean_text`: drop chars where `cp == 0 || cp == 0xFFFD || is_other()`
     (Cc/Cf/Cn/Co), then map whitespace → `' '`;
   - `handle_chinese_chars`: space-pad CJK-block chars (range table below);
   - `strip_accents` (resolved as `strip_accents.unwrap_or(lowercase)`):
     NFD decompose, then drop `is_mark_nonspacing()` (Mn) chars;
   - `lowercase`: Unicode lowercase.
3. **BertPreTokenizer**: split on whitespace (delimiter removed), then
   isolate punctuation — a char is punctuation iff `is_ascii_punctuation()
   || is_punctuation()` (P* categories); each punctuation char is its own
   token; runs of other chars form tokens.
4. **WordPiece** greedy longest-match-first per word: if
   `chars().count() > max_input_chars_per_word` (default 100) → whole word
   becomes `unk`. Else walk `start` forward; for each start, walk `end` down
   from word end (stepping back one UTF-8 char at a time); candidate is the
   substring, prefixed with the continuing prefix (`"##"`) when `start > 0`;
   first (longest) vocab hit wins. No hit at any start → whole word `unk`.
5. **Truncation** (LongestFirst, single sequence): applied BEFORE specials —
   content truncates to `max_length - 2` when specials will be added, then
   `[CLS] … [SEP]` wraps → final length is exactly `max_length`.
6. **Padding** (BatchLongest): right-pad with the pad id, attention mask 0.
   Single-sequence encode NEVER pads (a batch of one is its own longest).

CJK ranges for `handle_chinese_chars` (from the reference):
`0x4E00..=0x9FFF, 0x3400..=0x4DBF, 0x20000..=0x2A6DF, 0x2A700..=0x2B73F,
0x2B740..=0x2B81F, 0x2B920..=0x2CEAF, 0xF900..=0xFAFF, 0x2F800..=0x2FA1F`.

### Dependencies (parity requires the reference's own Unicode tables)

- `unicode-normalization 0.1` — NFD (the reference's own NFD source).
- `unicode_categories 0.1` — `is_other` / `is_mark_nonspacing` /
  `is_punctuation` (the exact predicates the reference uses).

### Fixtures (ALREADY COMMITTED on this branch — do not modify)

- `tests/fixtures/minilm-vocab.txt` — the real MiniLM vocab, 30522 lines,
  line *i* = token id *i* (Apache-2.0; provenance in
  `tests/fixtures/README.md`, which is also already committed).
- `tests/fixtures/golden_minilm.json` — goldens generated with
  `tokenizers 0.21.4` using Ijima's exact call shape
  (`from_file` + `with_padding(BatchLongest, ..default)` + `encode(text, true)`):
  `singles[]` (26: {text, ids, attention_mask}) and `batch[]` (3, BatchLongest
  padding observable). PaddingParams::default facts: pad_id 0, pad_token
  "[PAD]", direction Right.

## 1. Unit

Implement `katakuri::tokenizer`: a `Tokenizer` constructed from
`tokenizer.json` (strict BERT-class subset) or `vocab.txt` (BERT defaults),
with `encode` (single, specials, truncation) and `encode_batch`
(BatchLongest padding), byte-identical to the goldens.

## 2. Files

Create or replace EXACTLY these files (no others; Cargo.lock excepted):

1. Replace `Cargo.toml` — full content in §3.1
2. Replace `src/lib.rs` — full content in §3.2
3. Replace `src/error.rs` — full content in §3.3
4. Replace `src/tokenizer.rs` — full content in §3.4
5. Create `tests/tokenizer_golden.rs` — full content in §3.5
6. Replace `CHANGELOG.md` — full content in §3.6

PRESENT, DO NOT MODIFY (already committed): `tests/fixtures/minilm-vocab.txt`,
`tests/fixtures/golden_minilm.json`, `tests/fixtures/README.md`.

Do NOT touch: `src/safetensors.rs`, `src/verify.rs`, `src/kani_harnesses.rs`,
`src/model.rs`, `src/hub.rs`, `tests/real_file.rs`, `LICENSE`,
`CONTRIBUTING.md`, `README.md`, `rust-toolchain.toml`, `docs/plans/*`.

## 3. Contracts

### 3.1 `Cargo.toml`

```toml
[package]
name = "katakuri"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
description = "Model ecosystem layer over Zunesha — safetensors weight loading, WordPiece tokenization, architecture specs, and hub access. Never computes."
license = "Apache-2.0"
repository = "https://github.com/Industrial-Algebra/Katakuri"
keywords = ["safetensors", "tokenization", "wordpiece", "model-weights", "bert"]
categories = ["science", "parser-implementations"]
readme = "README.md"

[features]
default = ["std"]
std = []
# Borsalino-pattern verification obligations (karpal stack).
verify = ["dep:karpal-verify", "dep:karpal-proof"]

[dependencies]
thiserror = "2"
# Header parsing: the safetensors header IS JSON; serde_json is the
# ecosystem-standard, dependency-light parser (serde/itoa/ryu/memchr only).
serde_json = "1"
# The device substrate: Buffer upload + the NoDeviceStub host path.
zunesha = "0.1.2"
# Verification stack, feature-gated per the Borsalino pattern.
karpal-verify = { version = "0.6.1", optional = true }
karpal-proof = { version = "0.6.1", optional = true }
# Unicode tables the tokenizers reference implementation itself uses —
# parity requires identical category tables and NFD.
unicode-normalization = "0.1"
unicode_categories = "0.1"

# NOTE: Katakuri never computes — no forward passes, no autodiff, no optimizers,
# no kernels, no shader compilation. That boundary is the architectural contract
# (docs/plans/2026-10-16-katakuri-architecture-handover.md §4.1).
#
# Dependency ledger (no silent growth — every entry justified in the PR):
# - thiserror 2: central error enum (IA standard, same as Zunesha)
# - serde_json 1: safetensors header parse (the format is JSON)
# - zunesha 0.1.2: device upload + NoDevice host path (the substrate)
# - karpal-verify/proof 0.6.1: verify-feature obligations (Borsalino pattern,
#   pinned to Borsalino's own pin)
# - unicode-normalization 0.1: NFD for accent stripping (S2 tokenizer)
# - unicode_categories 0.1: is_other/is_mark_nonspacing/is_punctuation
#   predicates (S2 tokenizer; the reference implementation's own crate)

# Declare the `kani` cfg so `#[cfg(kani)]` (lib.rs module gate) does not trip
# rustc's `unexpected_cfgs` lint under `-D warnings`. Inert outside Kani.
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(kani)'] }
```

### 3.2 `src/error.rs`

The S1 variant set, verbatim, plus eight S2 variants inserted after
`TensorNotFound` and before `Device`.

```rust
    /// A `tokenizer.json` carries a normalizer kind Katakuri does not
    /// implement (only `BertNormalizer` is supported).
    #[error("unsupported normalizer kind '{kind}' (only BertNormalizer)")]
    UnsupportedNormalizer {
        /// The rejected kind string.
        kind: String,
    },

    /// A `tokenizer.json` carries a pre-tokenizer kind Katakuri does not
    /// implement (only `BertPreTokenizer` is supported).
    #[error("unsupported pre-tokenizer kind '{kind}' (only BertPreTokenizer)")]
    UnsupportedPreTokenizer {
        /// The rejected kind string.
        kind: String,
    },

    /// A `tokenizer.json` carries a post-processor Katakuri does not
    /// implement, or a template shape other than `[CLS] A [SEP]`.
    #[error("unsupported post-processor: {kind}")]
    UnsupportedPostProcessor {
        /// The rejected kind or shape description.
        kind: String,
    },

    /// A `tokenizer.json` carries a model kind other than WordPiece.
    #[error("unsupported model kind '{kind}' (only WordPiece)")]
    UnsupportedModel {
        /// The rejected kind string.
        kind: String,
    },

    /// A `tokenizer.json` enables truncation with a strategy other than
    /// `LongestFirst` (the single-sequence BERT default).
    #[error("unsupported truncation strategy '{strategy}' (only LongestFirst)")]
    UnsupportedTruncationStrategy {
        /// The rejected strategy string.
        strategy: String,
    },

    /// A `tokenizer.json` is malformed: a required field is missing or has
    /// the wrong shape (model vocab, added-token entries, truncation
    /// parameters, added-token id/content mismatch against the vocab).
    #[error("tokenizer.json invalid: {detail}")]
    InvalidTokenizerJson {
        /// What was wrong.
        detail: String,
    },

    /// A vocab source contained no tokens at all.
    #[error("vocab is empty")]
    EmptyVocab,

    /// A BERT-class special token (`[CLS]`, `[SEP]`, `[UNK]`, `[PAD]`) is
    /// absent from the vocab.
    #[error("vocab is missing the special token '{token}'")]
    MissingSpecialToken {
        /// The missing special token text.
        token: String,
    },
```

### 3.3 `src/lib.rs`

Identical to the S1-merged `src/lib.rs` (HEAD c789ce8) except the
`tokenizer` bullet of the module map, which becomes:

```rust
//! - [`tokenizer`] — WordPiece (BERT-class) tokenization matching Ijima's
//!   exact tokenization contract, with golden-vector parity against HF
//!   tokenizers 0.21.x (S2). Replaces the `tokenizers` crate dependency.
```

Transcribe the current `src/lib.rs` from the repository verbatim and apply
ONLY that one bullet replacement (plus no other changes). This is the one
place a delta instruction is used: the base file is the reviewed, merged S1
content — do not retype it; copy it from the working tree and edit the one
bullet.

### 3.4 `src/tokenizer.rs`

```rust
// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! WordPiece (BERT-class) tokenization — a bespoke, auditable engine
//! replacing the `tokenizers` crate dependency (slice S2).
//!
//! Byte-identical to HF Rust `tokenizers 0.21.x` on the committed golden
//! vectors (`tests/fixtures/golden_minilm.json`, generated from the real
//! MiniLM tokenizer with Ijima's exact call shape). The pipeline is a
//! transcription of the reference implementation (see the plan contract
//! `docs/plans/2026-10-16-katakuri-s2-tokenizer.md` §0):
//!
//! 1. added-token split on the raw input (leftmost, longest-first);
//! 2. `BertNormalizer` — clean_text, CJK spacing, accent strip
//!    (`strip_accents.unwrap_or(lowercase)`), lowercase, in that order;
//! 3. `BertPreTokenizer` — whitespace split (removed) + punctuation
//!    isolation;
//! 4. WordPiece greedy longest-match-first with the `##` continuation
//!    prefix, whole-word `[UNK]` on failure or over-length words;
//! 5. truncation to `max_length - 2` content tokens (LongestFirst,
//!    single-sequence), then `[CLS] … [SEP]`;
//! 6. `encode_batch` right-pads to the batch longest (BatchLongest).
//!
//! Single-sequence `encode` never pads. Constructed from `tokenizer.json`
//! (strict BERT-class subset — unsupported kinds reject with typed errors
//! rather than silently diverging) or from a line-per-token BERT vocab.

use crate::error::KatakuriError;
use serde_json::Value;
use std::collections::BTreeMap;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

// ── Public data ───────────────────────────────────────────────────

/// One token registered for raw-text matching before normalization
/// (BERT-class special tokens).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedTokenDef {
    /// The literal text matched against the raw input.
    pub content: String,
    /// The id emitted when matched.
    pub id: u32,
    /// Whether this is a special token.
    pub special: bool,
}

/// BertNormalizer flags, in the reference's application order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizerConfig {
    /// Drop control/invalid chars, then map whitespace to `' '`.
    pub clean_text: bool,
    /// Space-pad CJK-block chars so they tokenize individually.
    pub handle_chinese_chars: bool,
    /// NFD + drop nonspacing marks; `None` follows `lowercase`.
    pub strip_accents: Option<bool>,
    /// Unicode lowercase.
    pub lowercase: bool,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    }
}

/// The token ids and attention mask for one encoded sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    /// Token ids, `[CLS] … [SEP]` when specials are added.
    pub ids: Vec<u32>,
    /// 1 for real tokens, 0 for batch padding.
    pub attention_mask: Vec<u32>,
}

// ── Tokenizer ─────────────────────────────────────────────────────

/// A BERT-class WordPiece tokenizer.
///
/// Construct with [`Tokenizer::from_tokenizer_json`] (strict subset of the
/// HF `tokenizer.json` format) or [`Tokenizer::from_bert_vocab_txt`]
/// (line-per-token vocab with BERT defaults). `encode` is infallible:
/// unknown pieces become `[UNK]` by design.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: BTreeMap<String, u32>,
    /// Sorted longest-content-first; matching scans leftmost, longest-first.
    added_tokens: Vec<AddedTokenDef>,
    norm: NormalizerConfig,
    continuing_subword_prefix: String,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    max_input_chars_per_word: usize,
    truncation_max_length: Option<usize>,
}

impl Tokenizer {
    /// Parse a BERT-class `tokenizer.json` (strict subset).
    ///
    /// Accepted: `BertNormalizer`, `BertPreTokenizer`,
    /// `TemplateProcessing` with the single-sequence template
    /// `[CLS] A [SEP]`, `WordPiece` model, added-tokens list,
    /// `LongestFirst` truncation (optional). The `decoder` and `padding`
    /// fields are accepted and ignored (single-sequence encoding;
    /// `encode_batch` always uses BatchLongest). Anything else rejects
    /// with a typed error — Katakuri never silently diverges from a
    /// tokenizer configuration it does not implement.
    ///
    /// # Errors
    ///
    /// [`KatakuriError::UnsupportedNormalizer`],
    /// [`KatakuriError::UnsupportedPreTokenizer`],
    /// [`KatakuriError::UnsupportedPostProcessor`],
    /// [`KatakuriError::UnsupportedModel`],
    /// [`KatakuriError::UnsupportedTruncationStrategy`],
    /// [`KatakuriError::InvalidTokenizerJson`],
    /// [`KatakuriError::EmptyVocab`],
    /// [`KatakuriError::MissingSpecialToken`] — see variant docs.
    pub fn from_tokenizer_json(bytes: &[u8]) -> Result<Self, KatakuriError> {
        let root: Value =
            serde_json::from_slice(bytes).map_err(|e| KatakuriError::InvalidTokenizerJson {
                detail: format!("not valid JSON: {e}"),
            })?;
        let obj = root.as_object().ok_or(KatakuriError::InvalidTokenizerJson {
            detail: "root must be an object".to_string(),
        })?;

        // normalizer
        let norm_obj = obj
            .get("normalizer")
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "missing normalizer".to_string(),
            })?
            .as_object()
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "normalizer must be an object".to_string(),
            })?;
        let n_kind = norm_obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "normalizer missing type".to_string(),
            })?;
        if n_kind != "BertNormalizer" {
            return Err(KatakuriError::UnsupportedNormalizer {
                kind: n_kind.to_string(),
            });
        }
        let flag = |name: &str| -> Option<bool> {
            norm_obj.get(name).and_then(Value::as_bool)
        };
        let norm = NormalizerConfig {
            clean_text: flag("clean_text").unwrap_or(true),
            handle_chinese_chars: flag("handle_chinese_chars").unwrap_or(true),
            strip_accents: flag("strip_accents"),
            lowercase: flag("lowercase").unwrap_or(true),
        };

        // pre_tokenizer
        let pt_kind = obj
            .get("pre_tokenizer")
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "pre_tokenizer missing type".to_string(),
            })?;
        if pt_kind != "BertPreTokenizer" {
            return Err(KatakuriError::UnsupportedPreTokenizer {
                kind: pt_kind.to_string(),
            });
        }

        // post_processor: single template must be [SpecialToken X, Sequence A, SpecialToken Y]
        let pp = obj
            .get("post_processor")
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "missing post_processor".to_string(),
            })?;
        let pp_kind = pp
            .get("type")
            .and_then(Value::as_str)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "post_processor missing type".to_string(),
            })?;
        if pp_kind != "TemplateProcessing" {
            return Err(KatakuriError::UnsupportedPostProcessor {
                kind: pp_kind.to_string(),
            });
        }
        let single = pp
            .get("single")
            .and_then(Value::as_array)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "post_processor missing single template".to_string(),
            })?;
        let template_token = |entry: &Value, wrapper: &str| -> Option<String> {
            entry
                .get(wrapper)?
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        if single.len() != 3 {
            return Err(KatakuriError::UnsupportedPostProcessor {
                kind: "single template is not [SpecialToken, Sequence, SpecialToken]"
                    .to_string(),
            });
        }
        let cls_token = template_token(&single[0], "SpecialToken").ok_or(
            KatakuriError::UnsupportedPostProcessor {
                kind: "single template[0] is not a SpecialToken".to_string(),
            },
        )?;
        let seq_ok = single[1].get("Sequence").and_then(|s| s.get("id")).and_then(Value::as_str)
            == Some("A");
        if !seq_ok {
            return Err(KatakuriError::UnsupportedPostProcessor {
                kind: "single template[1] is not Sequence A".to_string(),
            });
        }
        let sep_token = template_token(&single[2], "SpecialToken").ok_or(
            KatakuriError::UnsupportedPostProcessor {
                kind: "single template[2] is not a SpecialToken".to_string(),
            },
        )?;

        // model
        let model = obj.get("model").ok_or(KatakuriError::InvalidTokenizerJson {
            detail: "missing model".to_string(),
        })?;
        let m_kind = model
            .get("type")
            .and_then(Value::as_str)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "model missing type".to_string(),
            })?;
        if m_kind != "WordPiece" {
            return Err(KatakuriError::UnsupportedModel {
                kind: m_kind.to_string(),
            });
        }
        let vocab_obj = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "model missing vocab object".to_string(),
            })?;
        if vocab_obj.is_empty() {
            return Err(KatakuriError::EmptyVocab);
        }
        let mut vocab = BTreeMap::new();
        for (tok, id) in vocab_obj {
            let id = id.as_u64().and_then(|i| u32::try_from(i).ok()).ok_or_else(|| {
                KatakuriError::InvalidTokenizerJson {
                    detail: format!("vocab id for '{tok}' is not a u32"),
                }
            })?;
            vocab.insert(tok.clone(), id);
        }
        let unk_token = model
            .get("unk_token")
            .and_then(Value::as_str)
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "model missing unk_token".to_string(),
            })?
            .to_string();
        let continuing_subword_prefix = model
            .get("continuing_subword_prefix")
            .and_then(Value::as_str)
            .unwrap_or("##")
            .to_string();
        let max_input_chars_per_word = model
            .get("max_input_chars_per_word")
            .and_then(Value::as_u64)
            .unwrap_or(100) as usize;

        // added tokens (matched on the raw input; the normalized flag is
        // accepted but not distinguished — BERT-class specials are
        // normalized: false)
        let mut added_tokens = Vec::new();
        if let Some(arr) = obj.get("added_tokens").and_then(Value::as_array) {
            for entry in arr {
                let content = entry
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| KatakuriError::InvalidTokenizerJson {
                        detail: "added_tokens entry missing content".to_string(),
                    })?
                    .to_string();
                let id = entry.get("id").and_then(Value::as_u64).and_then(|i| u32::try_from(i).ok()).ok_or_else(
                    || KatakuriError::InvalidTokenizerJson {
                        detail: format!("added token '{content}' missing u32 id"),
                    },
                )?;
                let special = entry
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if vocab.get(&content) != Some(&id) {
                    return Err(KatakuriError::InvalidTokenizerJson {
                        detail: format!(
                            "added token '{content}' id {id} does not match vocab"
                        ),
                    });
                }
                added_tokens.push(AddedTokenDef {
                    content,
                    id,
                    special,
                });
            }
        }
        added_tokens.sort_by_key(|t| core::cmp::Reverse(t.content.len()));

        // truncation (optional; LongestFirst only)
        let truncation_max_length = match obj.get("truncation") {
            None | Some(Value::Null) => None,
            Some(t) => {
                let strategy = t.get("strategy").and_then(Value::as_str).ok_or_else(|| {
                    KatakuriError::InvalidTokenizerJson {
                        detail: "truncation missing strategy".to_string(),
                    }
                })?;
                if strategy != "LongestFirst" {
                    return Err(KatakuriError::UnsupportedTruncationStrategy {
                        strategy: strategy.to_string(),
                    });
                }
                let max_length = t.get("max_length").and_then(Value::as_u64).ok_or_else(|| {
                    KatakuriError::InvalidTokenizerJson {
                        detail: "truncation missing u64 max_length".to_string(),
                    }
                })?;
                Some(usize::try_from(max_length).unwrap_or(usize::MAX))
            }
        };

        // resolve special ids from the vocab
        let special_id = |token: &str| -> Result<u32, KatakuriError> {
            vocab.get(token).copied().ok_or_else(|| {
                KatakuriError::MissingSpecialToken {
                    token: token.to_string(),
                }
            })
        };
        let unk_id = special_id(&unk_token)?;
        let cls_id = special_id(&cls_token)?;
        let sep_id = special_id(&sep_token)?;
        let pad_id = special_id("[PAD]")?;

        Ok(Self {
            vocab,
            added_tokens,
            norm,
            continuing_subword_prefix,
            unk_id,
            cls_id,
            sep_id,
            pad_id,
            max_input_chars_per_word,
            truncation_max_length,
        })
    }

    /// Build from a line-per-token BERT vocab (line *i* = token id *i*)
    /// with BERT-class defaults: BertNormalizer defaults, `"##"`
    /// continuation prefix, `[UNK]`/`[CLS]`/`[SEP]`/`[PAD]` specials
    /// resolved from the vocab, `max_input_chars_per_word` 100, no
    /// truncation.
    ///
    /// # Errors
    ///
    /// [`KatakuriError::EmptyVocab`] for an empty vocab;
    /// [`KatakuriError::MissingSpecialToken`] when a special token is
    /// absent.
    pub fn from_bert_vocab_txt(bytes: &[u8]) -> Result<Self, KatakuriError> {
        let text = std::str::from_utf8(bytes).map_err(|e| KatakuriError::InvalidTokenizerJson {
            detail: format!("vocab.txt is not UTF-8: {e}"),
        })?;
        let mut vocab = BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            let id = u32::try_from(i).unwrap_or(u32::MAX);
            vocab.insert(line.to_string(), id);
        }
        if vocab.is_empty() {
            return Err(KatakuriError::EmptyVocab);
        }
        let special_id = |token: &str| -> Result<u32, KatakuriError> {
            vocab.get(token).copied().ok_or_else(|| {
                KatakuriError::MissingSpecialToken {
                    token: token.to_string(),
                }
            })
        };
        Ok(Self {
            added_tokens: Vec::new(),
            unk_id: special_id("[UNK]")?,
            cls_id: special_id("[CLS]")?,
            sep_id: special_id("[SEP]")?,
            pad_id: special_id("[PAD]")?,
            norm: NormalizerConfig::default(),
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
            truncation_max_length: None,
            vocab,
        })
    }

    /// Number of tokens in the vocab.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode one sequence. Never pads; unknown pieces are `[UNK]`.
    #[must_use]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let mut ids = Vec::new();
        for segment in self.split_added(text) {
            match segment {
                AddedSegment::Raw(raw) => {
                    let normalized = self.normalize(raw);
                    for word in self.pre_tokenize(&normalized) {
                        ids.extend(self.wordpiece(word));
                    }
                }
                AddedSegment::Added(id) => ids.push(id),
            }
        }

        if let Some(max_length) = self.truncation_max_length {
            let content_max = if add_special_tokens {
                max_length.saturating_sub(2)
            } else {
                max_length
            };
            ids.truncate(content_max);
        }

        let mut final_ids = Vec::with_capacity(ids.len() + 2);
        if add_special_tokens {
            final_ids.push(self.cls_id);
        }
        final_ids.extend(ids);
        if add_special_tokens {
            final_ids.push(self.sep_id);
        }
        Encoding {
            attention_mask: vec![1; final_ids.len()],
            ids: final_ids,
        }
    }

    /// Encode a batch, right-padding every sequence to the batch longest
    /// with the pad id (attention mask 0 for padding).
    #[must_use]
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Vec<Encoding> {
        let mut encodings: Vec<Encoding> = texts
            .iter()
            .map(|t| self.encode(t, add_special_tokens))
            .collect();
        let longest = encodings.iter().map(|e| e.ids.len()).max().unwrap_or(0);
        for e in &mut encodings {
            let pad = longest - e.ids.len();
            e.ids.extend(std::iter::repeat_n(self.pad_id, pad));
            e.attention_mask.extend(std::iter::repeat_n(0, pad));
        }
        encodings
    }

    // ── pipeline stages ───────────────────────────────────────────

    /// Split the raw input on added tokens (leftmost, longest-first).
    fn split_added<'t>(&self, text: &'t str) -> Vec<AddedSegment<'t>> {
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            let mut hit: Option<(usize, u32)> = None;
            for t in &self.added_tokens {
                if text[i..].starts_with(t.content.as_str()) {
                    hit = Some((t.content.len(), t.id));
                    break;
                }
            }
            if let Some((len, id)) = hit {
                if i > start {
                    out.push(AddedSegment::Raw(&text[start..i]));
                }
                out.push(AddedSegment::Added(id));
                i += len;
                start = i;
            } else {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if start < text.len() {
            out.push(AddedSegment::Raw(&text[start..]));
        }
        out
    }

    /// The BertNormalizer pipeline, in the reference's order.
    fn normalize(&self, s: &str) -> String {
        // 1. clean_text
        let cleaned: String = if self.norm.clean_text {
            s.chars()
                .filter(|&c| {
                    let cp = c as usize;
                    cp != 0 && cp != 0xFFFD && !c.is_other()
                })
                .map(|c| if c.is_whitespace() { ' ' } else { c })
                .collect()
        } else {
            s.to_string()
        };
        // 2. CJK spacing
        let spaced: String = if self.norm.handle_chinese_chars {
            let mut out = String::with_capacity(cleaned.len());
            for c in cleaned.chars() {
                if is_chinese_char(c) {
                    out.push(' ');
                    out.push(c);
                    out.push(' ');
                } else {
                    out.push(c);
                }
            }
            out
        } else {
            cleaned
        };
        // 3. strip accents (NFD + drop Mn)
        let stripped: String = if self.norm.strip_accents.unwrap_or(self.norm.lowercase) {
            spaced.chars().nfd().filter(|c| !c.is_mark_nonspacing()).collect()
        } else {
            spaced
        };
        // 4. lowercase
        if self.norm.lowercase {
            stripped.to_lowercase()
        } else {
            stripped
        }
    }

    /// BertPreTokenizer: whitespace split (removed), punctuation isolated.
    fn pre_tokenize<'t>(&self, s: &'t str) -> Vec<&'t str> {
        let mut words = Vec::new();
        let mut run: Option<(usize, usize)> = None; // (start, end) byte range
        for (idx, c) in s.char_indices() {
            if c.is_whitespace() {
                if let Some((st, en)) = run.take() {
                    words.push(&s[st..en]);
                }
            } else if is_bert_punc(c) {
                if let Some((st, en)) = run.take() {
                    words.push(&s[st..en]);
                }
                words.push(&s[idx..idx + c.len_utf8()]);
            } else {
                run = match run {
                    Some((st, _)) => Some((st, idx + c.len_utf8())),
                    None => Some((idx, idx + c.len_utf8())),
                };
            }
        }
        if let Some((st, en)) = run {
            words.push(&s[st..en]);
        }
        words
    }

    /// WordPiece greedy longest-match-first; whole word → `[UNK]` on any
    /// failure or over-length.
    fn wordpiece(&self, word: &str) -> Vec<u32> {
        if word.chars().count() > self.max_input_chars_per_word {
            return vec![self.unk_id];
        }
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < word.len() {
            let mut end = word.len();
            let mut matched: Option<u32> = None;
            while start < end {
                let sub = &word[start..end];
                let id = if start > 0 {
                    let key = format!("{}{}", self.continuing_subword_prefix, sub);
                    self.vocab.get(key.as_str()).copied()
                } else {
                    self.vocab.get(sub).copied()
                };
                if let Some(id) = id {
                    matched = Some(id);
                    break;
                }
                let step = sub.chars().next_back().map_or(1, char::len_utf8);
                end -= step;
            }
            match matched {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => return vec![self.unk_id],
            }
        }
        out
    }
}

enum AddedSegment<'t> {
    Raw(&'t str),
    Added(u32),
}

/// CJK Unified Ideographs blocks (reference range table).
fn is_chinese_char(c: char) -> bool {
    matches!(
        c as usize,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// ASCII punctuation or a Unicode P* category char.
fn is_bert_punc(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}
```

function exactly once, `split_added` in the corrected form.)

#### 3.4 (continued) — append this block verbatim to the END of `src/tokenizer.rs`

It is the in-file test module (same single-file convention as S1; it
belongs after `is_bert_punc`). Do not modify the implementation above to
satisfy a test — the implementation is the contract.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Direct struct construction with a tiny vocab (ids by list index).
    fn tiny(vocab: &[&str]) -> Tokenizer {
        let mut map = BTreeMap::new();
        for (i, t) in vocab.iter().enumerate() {
            map.insert((*t).to_string(), i as u32);
        }
        let id = |tok: &str| -> u32 {
            *map.get(tok).unwrap_or_else(|| panic!("{tok} in tiny vocab"))
        };
        Tokenizer {
            added_tokens: Vec::new(),
            unk_id: id("[UNK]"),
            cls_id: id("[CLS]"),
            sep_id: id("[SEP]"),
            pad_id: id("[PAD]"),
            norm: NormalizerConfig::default(),
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 100,
            truncation_max_length: None,
            vocab: map,
        }
    }

    /// A valid BERT-class tokenizer.json built from a tiny vocab, for the
    /// from_tokenizer_json tests. Mutate the returned Value per test.
    fn tiny_json() -> Value {
        let vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "un", "##aff", "##able", "aff"];
        let mut vocab_obj = serde_json::Map::new();
        for (i, t) in vocab.iter().enumerate() {
            vocab_obj.insert((*t).to_string(), Value::from(i as u32));
        }
        let special = |content: &str, id: u32| {
            json!({"id": id, "content": content, "single_word": false,
                   "lstrip": false, "rstrip": false, "normalized": false,
                   "special": true})
        };
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                special("[PAD]", 0), special("[UNK]", 1),
                special("[CLS]", 2), special("[SEP]", 3),
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
        })
    }

    fn from_json(v: &Value) -> Result<Tokenizer, KatakuriError> {
        let bytes = serde_json::to_vec(v).expect("serialize");
        Tokenizer::from_tokenizer_json(&bytes)
    }

    // ── Normalizer ───────────────────────────────────────────────

    #[test]
    fn normalizer_full_pipeline() {
        let tk = tiny(&[]);
        // lowercase + NFD accent strip (strip follows lowercase when None)
        assert_eq!(tk.normalize("Café ÉMIGRÉ"), "cafe emigre");
        // control chars and replacement char dropped; whitespace → space
        assert_eq!(tk.normalize("a\u{0}b\u{fffd}c\u{1}d"), "abcd");
        assert_eq!(tk.normalize("x\ty"), "x y");
    }

    #[test]
    fn normalizer_respects_flag_gates() {
        let mut tk = tiny(&[]);
        // strip explicitly false, lowercase false → untouched
        tk.norm = NormalizerConfig {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: Some(false),
            lowercase: false,
        };
        assert_eq!(tk.normalize("Café ÉMIGRÉ"), "Café ÉMIGRÉ");
        // strip explicitly true, lowercase false → accents gone, case kept
        tk.norm = NormalizerConfig {
            strip_accents: Some(true),
            lowercase: false,
            ..tk.norm
        };
        assert_eq!(tk.normalize("Café ÉMIGRÉ"), "Cafe EMIGRE");
        // clean_text false → control chars survive
        tk.norm = NormalizerConfig {
            clean_text: false,
            strip_accents: Some(false),
            lowercase: false,
            handle_chinese_chars: true,
        };
        assert_eq!(tk.normalize("a\u{1}b"), "a\u{1}b");
    }

    #[test]
    fn normalizer_cjk_spacing() {
        let tk = tiny(&[]);
        assert_eq!(tk.normalize("日本x"), " 日 本 x");
        // non-CJK scripts untouched
        assert_eq!(tk.normalize("café"), "cafe");
    }

    // ── Pre-tokenizer ────────────────────────────────────────────

    #[test]
    fn pretokenizer_whitespace_and_punctuation() {
        let tk = tiny(&[]);
        // Vector from the reference implementation's own test suite.
        let words = tk.pre_tokenize("Hey friend!     How are you?!?");
        let expect = ["Hey", "friend", "!", "How", "are", "you", "?", "!", "?"];
        assert_eq!(words, expect);
        let words = tk.pre_tokenize("don't stop—now");
        assert_eq!(words, ["don", "'", "t", "stop", "—", "now"]);
    }

    // ── WordPiece ────────────────────────────────────────────────

    #[test]
    fn wordpiece_greedy_match() {
        let tk = tiny(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "aff", "hello",
        ]);
        assert_eq!(tk.wordpiece("hello"), [8]);
        assert_eq!(tk.wordpiece("unaffable"), [4, 5, 6]);
        assert_eq!(tk.wordpiece("affable"), [7, 6]);
        // no first-piece match → whole word UNK, even though pieces exist
        assert_eq!(tk.wordpiece("unxyz"), [1]);
        assert_eq!(tk.wordpiece("xyz"), [1]);
    }

    #[test]
    fn wordpiece_long_word_becomes_unk() {
        let tk = tiny(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "##a"]);
        let hundred: String = "a".repeat(100);
        let hundred_one: String = "a".repeat(101);
        assert_eq!(tk.wordpiece(&hundred), [4]);
        assert_eq!(tk.wordpiece(&hundred_one), [1]);
    }

    // ── encode / encode_batch ────────────────────────────────────

    #[test]
    fn encode_wraps_specials_and_never_pads() {
        let tk = tiny(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##ing",
        ]);
        let enc = tk.encode("helloworld", false);
        assert_eq!(enc.ids, [4, 5]);
        assert_eq!(enc.attention_mask, [1, 1]);
        let enc = tk.encode("hello", true);
        assert_eq!(enc.ids, [2, 4, 3]);
        assert_eq!(enc.attention_mask, [1, 1, 1]);
    }

    #[test]
    fn truncation_reserves_special_slots() {
        let mut v = tiny_json();
        v["truncation"] = json!({"max_length": 10, "strategy": "LongestFirst", "stride": 0});
        v["model"]["vocab"]["quick"] = json!(100);
        let tk = from_json(&v).expect("parse");
        let enc = tk.encode("quick quick quick quick quick quick quick quick", true);
        assert_eq!(enc.ids.len(), 10, "exactly max_length");
        assert_eq!(enc.ids[0], 2, "starts with CLS");
        assert_eq!(enc.ids[9], 3, "ends with SEP");
        assert_eq!(enc.attention_mask, [1; 10]);
    }

    #[test]
    fn batch_pads_to_longest_right() {
        let tk = tiny(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h",
        ]);
        let encs = tk.encode_batch(&["a", "a b c d e f g h", "b c"], true);
        assert_eq!(encs.len(), 3);
        for e in &encs {
            assert_eq!(e.ids.len(), 10, "all padded to longest");
        }
        assert_eq!(encs[0].ids, [2, 4, 3, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(encs[0].attention_mask, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(encs[1].ids, [2, 4, 5, 6, 7, 8, 9, 10, 11, 3]);
        assert_eq!(encs[1].attention_mask, [1; 10]);
        assert_eq!(encs[2].ids, [2, 5, 6, 3, 0, 0, 0, 0, 0, 0]);
        assert!(tk.encode_batch(&[], true).is_empty());
    }

    #[test]
    fn added_tokens_matched_on_raw_text() {
        let mut v = tiny_json();
        // tiny_json already registers [CLS]/[SEP]/[UNK]/[PAD] as added
        let tk = from_json(&v).expect("parse");
        let enc = tk.encode("[SEP] hello", false);
        // [SEP] matched raw → id 3, then hello → 8
        assert_eq!(enc.ids, [3, 8]);
        // case-mismatched literal is NOT matched; flows through the
        // normal pipeline (punctuation runs → UNK where vocab lacks them)
        let enc = tk.encode("[sep] hello", false);
        assert_eq!(enc.ids, [1, 1, 1, 8]);
    }

    #[test]
    fn encode_empty_and_whitespace() {
        let tk = tiny(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"]);
        assert_eq!(tk.encode("", true).ids, [2, 3]);
        assert_eq!(tk.encode(" ", true).ids, [2, 3]);
        assert_eq!(tk.encode("", false).ids, []);
    }

    // ── Construction / validation ────────────────────────────────

    #[test]
    fn from_tokenizer_json_rejects_unsupported_kinds() {
        let mut v = tiny_json();
        v["normalizer"]["type"] = json!("Metaspace");
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedNormalizer { kind }) if kind == "Metaspace"
        ));

        let mut v = tiny_json();
        v["pre_tokenizer"]["type"] = json!("ByteLevel");
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedPreTokenizer { kind }) if kind == "ByteLevel"
        ));

        let mut v = tiny_json();
        v["model"]["type"] = json!("BPE");
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedModel { kind }) if kind == "BPE"
        ));

        let mut v = tiny_json();
        v["post_processor"]["type"] = json!("BertProcessing");
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedPostProcessor { .. })
        ));

        let mut v = tiny_json();
        v["post_processor"]["single"] = json!([
            {"Sequence": {"id": "A", "type_id": 0}},
        ]);
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedPostProcessor { .. })
        ));

        let mut v = tiny_json();
        v["truncation"] = json!({"max_length": 8, "strategy": "OnlyFirst", "stride": 0});
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedTruncationStrategy { strategy })
                if strategy == "OnlyFirst"
        ));
    }

    #[test]
    fn from_tokenizer_json_validates_structure() {
        let mut v = tiny_json();
        v.as_object_mut().unwrap().remove("model");
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));

        let mut v = tiny_json();
        v["model"]["vocab"] = json!({});
        assert!(matches!(from_json(&v), Err(KatakuriError::EmptyVocab)));

        let mut v = tiny_json();
        v["added_tokens"][0]["id"] = json!(99);
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));

        // vocab without [PAD] → MissingSpecialToken
        let mut v = tiny_json();
        v["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("[PAD]");
        v["added_tokens"].as_array_mut().unwrap().remove(0);
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::MissingSpecialToken { token }) if token == "[PAD]"
        ));
    }

    #[test]
    fn from_bert_vocab_txt_resolves_specials_no_truncation() {
        let txt = "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n##ing\ngo\n";
        let tk = Tokenizer::from_bert_vocab_txt(txt.as_bytes()).expect("parse");
        assert_eq!(tk.vocab_size(), 7);
        let enc = tk.encode("go", true);
        assert_eq!(enc.ids, [2, 6, 3]);
        // no truncation configured → long input stays long
        let long = "go ".repeat(50);
        let enc = tk.encode(&long, true);
        assert_eq!(enc.ids.len(), 52);
        // empty vocab rejects
        assert!(matches!(
            Tokenizer::from_bert_vocab_txt(b""),
            Err(KatakuriError::EmptyVocab)
        ));
    }
}
```

### 3.5 `tests/tokenizer_golden.rs`

```rust
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
        .map(|x| x.as_u64().expect("u32 id") as u32)
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
    for s in golden["singles"].as_array().expect("singles").iter().take(5) {
        let text = s["text"].as_str().expect("text");
        let enc = tk.encode(text, true);
        assert_eq!(enc.ids, ids_of(&s["ids"]), "real-file ids diverge on {text:?}");
    }
}
```

### 3.6 `CHANGELOG.md`

The current CHANGELOG (HEAD c789ce8) with a new section inserted directly
after the `## [Unreleased]` line:

```markdown
### Added — S2: WordPiece tokenizer (2026-10-16)

- `katakuri::tokenizer`: bespoke BERT-class WordPiece engine, replacing
  the `tokenizers` crate dependency for Ijima's embedding surface.
  Pipeline transcribed from tokenizers-0.21.4: raw added-token split
  (leftmost, longest-first), BertNormalizer (clean_text → CJK spacing →
  NFD accent strip following `lowercase` → lowercase), BertPreTokenizer
  (whitespace split + punctuation isolation), greedy longest-match
  WordPiece with `##` continuations, LongestFirst truncation reserving
  the two special slots, `[CLS] … [SEP]` wrapping, BatchLongest
  right-padding.
- Constructors: `from_tokenizer_json` (strict BERT-class subset —
  unsupported kinds reject with typed errors, never silent divergence)
  and `from_bert_vocab_txt` (BERT defaults).
- Golden-vector parity: 26 singles + 3 batch goldens generated from the
  real MiniLM tokenizer with Ijima's exact call shape
  (`tests/fixtures/`, tokenizers 0.21.4); vocab fixture committed so CI
  replays parity without network.
- Dependencies: `unicode-normalization 0.1` (NFD) and
  `unicode_categories 0.1` (is_other/is_mark_nonspacing/is_punctuation)
  — the reference implementation's own Unicode tables.
```

## 4. Tests (the enumeration)

| # | Test (name) | Rule pinned |
|---|---|---|
| 1 | `normalizer_full_pipeline` | lowercase+NFD strip; control/replacement chars dropped; whitespace → space |
| 2 | `normalizer_respects_flag_gates` | strip Some(false)/lowercase false; strip Some(true); clean_text false |
| 3 | `normalizer_cjk_spacing` | CJK space-padded; non-CJK untouched |
| 4 | `pretokenizer_whitespace_and_punctuation` | reference test vector verbatim; em-dash isolation |
| 5 | `wordpiece_greedy_match` | full word; subword split; first-piece miss → whole-word UNK |
| 6 | `wordpiece_long_word_becomes_unk` | 100 chars OK, 101 → UNK |
| 7 | `encode_wraps_specials_and_never_pads` | CLS/SEP wrap; mask all 1; no padding |
| 8 | `truncation_reserves_special_slots` | exactly max_length, starts CLS, ends SEP |
| 9 | `batch_pads_to_longest_right` | right-pad pad_id, mask 0; empty batch |
| 10 | `added_tokens_matched_on_raw_text` | raw [SEP] matched; case-mismatch flows through pipeline |
| 11 | `encode_empty_and_whitespace` | `""`/`" "` → `[CLS] [SEP]`; no-specials → `[]` |
| 12 | `from_tokenizer_json_rejects_unsupported_kinds` | 6 rejection shapes |
| 13 | `from_tokenizer_json_validates_structure` | missing model, empty vocab, added-id mismatch, missing [PAD] |
| 14 | `from_bert_vocab_txt_resolves_specials_no_truncation` | line-number ids; specials resolved; no truncation; empty rejects |
| 15 | `golden_singles_match_hf_tokenizers` (§3.5) | all 26 singles: ids AND masks byte-identical |
| 16 | `golden_batch_matches_hf_tokenizers` (§3.5) | 3 batch: BatchLongest padding identical |
| 17 | `real_tokenizer_json_file_matches_goldens` (§3.5, `#[ignore]`, env-gated) | real tokenizer.json path |

Acceptance mapping (handover §5 S2): byte-identical vs HF on fixed corpus
= #15/#16/#17; Ijima's tokenization contract = specials (#7), truncation
(#8), padding (#9), raw added-token matching (#10).

## 5. Constraints

- Dependencies are PINNED to §3.1 — no new crates beyond the listed set,
  no version edits.
- Every `.rs` file starts with the two-line Apache-2.0 SPDX header.
- No `unwrap()`/`expect()`/`panic!()` in non-test library code paths.
- Feature gates additive; the no-verify configuration must compile clean.
- `tests/fixtures/*` are INPUTS — copy exactly, never regenerate or edit.
- Files outside §2's list are out of bounds.
- No git commands — leave the working tree dirty.

## 6. Completion

Run, filter, and fix until ALL are green (pipe long outputs; cargo
timeouts in seconds, ≤ 600). Note the STRICT doc gates — both feature
configurations:

```bash
cargo test -p katakuri 2>&1 | grep -E 'test result|FAILED|error' | tail -10
cargo test -p katakuri --features verify 2>&1 | grep -E 'test result|FAILED|error' | tail -10
cargo clippy --all-targets -- -D warnings 2>&1 | tail -5
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -5
cargo fmt --check && echo FMT-OK
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps 2>&1 | tail -3
RUSTDOCFLAGS="-D warnings" cargo doc -p katakuri --no-deps --features verify 2>&1 | tail -3
```

Expected: 40 unit tests in the lib (26 safetensors + 14 tokenizer) plus
the doctest, all green in default config; integration: 2 golden tests
pass, 2 ignored (real_file, real tokenizer.json — env-gated); +1 verify
test with `--features verify`; both clippy configs zero warnings; fmt
clean; both doc gates zero warnings. The golden tests MUST pass — if any
golden diverges, the implementation is wrong, not the golden.

## 7. Out of scope (fences)

- Pair-sequence encoding (`encode` with two inputs; the pair template is
  accepted but unused).
- The `decoder` and `padding` fields of tokenizer.json (accepted,
  ignored; `encode_batch` always BatchLongest).
- `normalized: true` added tokens (matched on raw text like the rest;
  documented approximation).
- Decoding ids back to text; offset tracking; type ids beyond 0.
- tokenizer.json kinds beyond the BERT-class subset (Metaspace, ByteLevel
  BPE, SentencePiece, Unigram, …) — those reject by design.
- Ijima integration PR (separate, in Ijima's repo).
- Performance optimization (the wordpiece per-candidate `format!` is
  correct-first; optimization is a later slice if measurements demand).

## 8. Author-side checklist (already applied)

- No RPITIT traits; no trait objects over generic methods.
- `verify`-gated surface unchanged; no new ungated karpal references.
- Every demanded assertion is observable: unit tests live INSIDE the
  module (private-stage access — `normalize`, `pre_tokenize`,
  `wordpiece`, and the struct literal for `tiny`); golden tests use only
  the public API.
- Cumulative files are full final contents (Cargo.toml, error.rs
  delta-block inserted at the named position, tokenizer.rs in two
  verbatim blocks concatenated, lib.rs copied from the working tree with
  ONE bullet replaced).
- Both strict doc gates in Completion (the round-2 lesson from S1).
- The pipeline order, flag semantics, range tables, and test vectors are
  transcriptions of tokenizers-0.21.4 sources read at
  `~/.cargo/registry/src/*/tokenizers-0.21.4/src/` — not from memory.
