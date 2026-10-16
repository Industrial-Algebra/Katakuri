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
        let obj = root
            .as_object()
            .ok_or(KatakuriError::InvalidTokenizerJson {
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
        let n_kind = norm_obj.get("type").and_then(Value::as_str).ok_or(
            KatakuriError::InvalidTokenizerJson {
                detail: "normalizer missing type".to_string(),
            },
        )?;
        if n_kind != "BertNormalizer" {
            return Err(KatakuriError::UnsupportedNormalizer {
                kind: n_kind.to_string(),
            });
        }
        let flag = |name: &str| -> Option<bool> { norm_obj.get(name).and_then(Value::as_bool) };
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
        let pp_kind =
            pp.get("type")
                .and_then(Value::as_str)
                .ok_or(KatakuriError::InvalidTokenizerJson {
                    detail: "post_processor missing type".to_string(),
                })?;
        if pp_kind != "TemplateProcessing" {
            return Err(KatakuriError::UnsupportedPostProcessor {
                kind: pp_kind.to_string(),
            });
        }
        let single = pp.get("single").and_then(Value::as_array).ok_or(
            KatakuriError::InvalidTokenizerJson {
                detail: "post_processor missing single template".to_string(),
            },
        )?;
        let template_token = |entry: &Value, wrapper: &str| -> Option<String> {
            entry
                .get(wrapper)?
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        if single.len() != 3 {
            return Err(KatakuriError::UnsupportedPostProcessor {
                kind: "single template is not [SpecialToken, Sequence, SpecialToken]".to_string(),
            });
        }
        let cls_token = template_token(&single[0], "SpecialToken").ok_or(
            KatakuriError::UnsupportedPostProcessor {
                kind: "single template[0] is not a SpecialToken".to_string(),
            },
        )?;
        let seq_ok = single[1]
            .get("Sequence")
            .and_then(|s| s.get("id"))
            .and_then(Value::as_str)
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
        let model = obj
            .get("model")
            .ok_or(KatakuriError::InvalidTokenizerJson {
                detail: "missing model".to_string(),
            })?;
        let m_kind = model.get("type").and_then(Value::as_str).ok_or(
            KatakuriError::InvalidTokenizerJson {
                detail: "model missing type".to_string(),
            },
        )?;
        if m_kind != "WordPiece" {
            return Err(KatakuriError::UnsupportedModel {
                kind: m_kind.to_string(),
            });
        }
        let vocab_obj = model.get("vocab").and_then(Value::as_object).ok_or(
            KatakuriError::InvalidTokenizerJson {
                detail: "model missing vocab object".to_string(),
            },
        )?;
        if vocab_obj.is_empty() {
            return Err(KatakuriError::EmptyVocab);
        }
        let mut vocab = BTreeMap::new();
        for (tok, id) in vocab_obj {
            let id = id
                .as_u64()
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| KatakuriError::InvalidTokenizerJson {
                    detail: format!("vocab id for '{tok}' is not a u32"),
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
                if content.is_empty() {
                    return Err(KatakuriError::InvalidTokenizerJson {
                        detail: "added token content must be non-empty".to_string(),
                    });
                }
                for flag_name in ["single_word", "lstrip", "rstrip"] {
                    if entry.get(flag_name).and_then(Value::as_bool) == Some(true) {
                        return Err(KatakuriError::InvalidTokenizerJson {
                            detail: format!(
                                "added token '{content}' sets unsupported flag '{flag_name}'"
                            ),
                        });
                    }
                }
                let id = entry
                    .get("id")
                    .and_then(Value::as_u64)
                    .and_then(|i| u32::try_from(i).ok())
                    .ok_or_else(|| KatakuriError::InvalidTokenizerJson {
                        detail: format!("added token '{content}' missing u32 id"),
                    })?;
                let special = entry
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if vocab.get(&content) != Some(&id) {
                    return Err(KatakuriError::InvalidTokenizerJson {
                        detail: format!("added token '{content}' id {id} does not match vocab"),
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
                let direction = t
                    .get("direction")
                    .and_then(Value::as_str)
                    .unwrap_or("Right");
                if direction != "Right" {
                    return Err(KatakuriError::UnsupportedTruncationDirection {
                        direction: direction.to_string(),
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

        let cls_label = cls_token.clone();
        let sep_label = sep_token.clone();

        // post_processor.special_tokens, when PRESENT, must be an object —
        // HF rejects malformed maps at deserialization; absence alone is
        // the documented fallback (review rounds 1–2).
        match pp.get("special_tokens") {
            None => {}
            Some(Value::Object(map)) => {
                // Every entry must be structurally valid: object with
                // non-empty, equal-length ids (u64) / tokens (strings).
                for (key, entry) in map {
                    let (ids, tokens) = match (
                        entry.get("ids").and_then(Value::as_array),
                        entry.get("tokens").and_then(Value::as_array),
                    ) {
                        (Some(ids), Some(tokens))
                            if !ids.is_empty()
                                && ids.len() == tokens.len()
                                && ids.iter().all(|i| {
                                    i.as_u64().is_some_and(|x| u32::try_from(x).is_ok())
                                })
                                && tokens.iter().all(|t| t.as_str().is_some()) =>
                        {
                            (ids, tokens)
                        }
                        _ => {
                            return Err(KatakuriError::InvalidTokenizerJson {
                                detail: format!("special_tokens entry '{key}' is malformed"),
                            });
                        }
                    };
                    // Template labels additionally: singleton expansions
                    // agreeing with the vocab (review round 1).
                    if key == &cls_label || key == &sep_label {
                        if ids.len() != 1 {
                            return Err(KatakuriError::InvalidTokenizerJson {
                                detail: format!(
                                    "special_tokens mapping for '{key}' must be a singleton expansion"
                                ),
                            });
                        }
                        let id = ids[0].as_u64().unwrap_or(u64::MAX);
                        let tok = tokens[0].as_str().unwrap_or("");
                        let vocab_id =
                            u64::from(vocab.get(key.as_str()).copied().unwrap_or(u32::MAX));
                        if id != vocab_id || tok != key.as_str() {
                            return Err(KatakuriError::InvalidTokenizerJson {
                                detail: format!(
                                    "special_tokens mapping for '{key}' diverges from the vocab"
                                ),
                            });
                        }
                    }
                }
                // Both template labels must be mapped when the map is present.
                for label in [&cls_label, &sep_label] {
                    if !map.contains_key(label.as_str()) {
                        return Err(KatakuriError::InvalidTokenizerJson {
                            detail: format!("special_tokens missing mapping for '{label}'"),
                        });
                    }
                }
            }
            Some(other) => {
                return Err(KatakuriError::InvalidTokenizerJson {
                    detail: format!("post_processor.special_tokens must be an object, got {other}"),
                });
            }
        }

        // resolve special ids from the vocab
        let special_id = |token: &str| -> Result<u32, KatakuriError> {
            vocab
                .get(token)
                .copied()
                .ok_or_else(|| KatakuriError::MissingSpecialToken {
                    token: token.to_string(),
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
            vocab
                .get(token)
                .copied()
                .ok_or_else(|| KatakuriError::MissingSpecialToken {
                    token: token.to_string(),
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
                if t.content.is_empty() {
                    continue;
                }
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
                    cp != 0 && cp != 0xFFFD && !is_control(c)
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
            spaced
                .chars()
                .nfd()
                .filter(|c| !c.is_mark_nonspacing())
                .collect()
        } else {
            spaced
        };
        // 4. lowercase — per character: the reference maps each char
        // independently; str::to_lowercase would apply contextual final
        // sigma and diverge from HF on e.g. "ΟΣ" (review round 1).
        if self.norm.lowercase {
            stripped.chars().flat_map(char::to_lowercase).collect()
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

/// Control character per the reference's definition: `\t`, `\n`, and
/// `\r` are excluded because the reference counts them as whitespace.
fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => c.is_other(),
    }
}

/// ASCII punctuation or a Unicode P* category char.
fn is_bert_punc(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Direct struct construction with a tiny vocab (ids by list index).
    fn tiny(vocab: &[&str]) -> Tokenizer {
        let mut map = BTreeMap::new();
        for (i, t) in vocab.iter().enumerate() {
            map.insert((*t).to_string(), i as u32);
        }
        let id = |tok: &str| -> u32 { map.get(tok).copied().unwrap_or(0) };
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
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "aff", "hello",
        ];
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
        assert_eq!(tk.normalize("日本x"), " 日  本 x");
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
        let hundred: String = "a".repeat(100);
        let hundred_one: String = "a".repeat(101);
        let tk = tiny(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", &hundred, "##a"]);
        assert_eq!(tk.wordpiece(&hundred), [4]);
        assert_eq!(tk.wordpiece(&hundred_one), [1]);
    }

    // ── encode / encode_batch ────────────────────────────────────

    #[test]
    fn encode_wraps_specials_and_never_pads() {
        let tk = tiny(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##world", "##ing",
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
        let v = tiny_json();
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
        assert_eq!(tk.encode("", false).ids, [] as [u32; 0]);
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
        v["model"]["vocab"].as_object_mut().unwrap().remove("[PAD]");
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

    #[test]
    fn lowercase_per_char_no_final_sigma() {
        let tk = tiny(&[]);
        // Per-char: Σ → σ always. str::to_lowercase would give "ος" here.
        assert_eq!(tk.normalize("ΟΣ"), "οσ");
        assert_eq!(tk.normalize("ΟΣΟ"), "οσο");
    }

    #[test]
    fn empty_and_flagged_added_tokens_reject() {
        let mut v = tiny_json();
        v["model"]["vocab"][""] = json!(9);
        v["added_tokens"].as_array_mut().unwrap().push(
            json!({"id": 9, "content": "", "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false,
                "special": false}),
        );
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));

        let mut v = tiny_json();
        v["added_tokens"][3]["single_word"] = json!(true);
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));
    }

    #[test]
    fn truncation_direction_left_rejects() {
        let mut v = tiny_json();
        v["truncation"] = json!({
            "direction": "Left", "max_length": 8,
            "strategy": "LongestFirst", "stride": 0
        });
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::UnsupportedTruncationDirection { direction })
                if direction == "Left"
        ));
    }

    #[test]
    fn template_special_tokens_mapping_validated() {
        // Divergent id (CLS mapped to [SEP]'s id) → reject.
        let mut v = tiny_json();
        v["post_processor"]["special_tokens"] = json!({
            "[CLS]": {"ids": [3], "tokens": ["[CLS]"]},
            "[SEP]": {"ids": [3], "tokens": ["[SEP]"]}
        });
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));
        // Non-singleton expansion → reject.
        let mut v = tiny_json();
        v["post_processor"]["special_tokens"] = json!({
            "[CLS]": {"ids": [2, 3], "tokens": ["[CLS]"]},
            "[SEP]": {"ids": [3], "tokens": ["[SEP]"]}
        });
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));
        // Agreeing mapping → accepted (both specials resolve as before).
        let mut v = tiny_json();
        v["post_processor"]["special_tokens"] = json!({
            "[CLS]": {"ids": [2], "tokens": ["[CLS]"]},
            "[SEP]": {"ids": [3], "tokens": ["[SEP]"]}
        });
        assert!(from_json(&v).is_ok());
    }

    #[test]
    fn malformed_special_tokens_maps_reject() {
        // Present-but-not-an-object: HF rejects these bytes at
        // deserialization; absence alone is the fallback (review r2).
        for bad in [json!([]), json!(null), json!("bad")] {
            let mut v = tiny_json();
            v["post_processor"]["special_tokens"] = bad.clone();
            assert!(
                matches!(
                    from_json(&v),
                    Err(KatakuriError::InvalidTokenizerJson { .. })
                ),
                "special_tokens {bad} should reject"
            );
        }
        // Structurally-garbage extra entry rejects too.
        let mut v = tiny_json();
        v["post_processor"]["special_tokens"] = json!({
            "[CLS]": {"ids": [2], "tokens": ["[CLS]"]},
            "[SEP]": {"ids": [3], "tokens": ["[SEP]"]},
            "[MASK]": 3
        });
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));
        // IDs above u32::MAX reject (HF deserializes ids as Vec<u32>).
        let mut v = tiny_json();
        v["post_processor"]["special_tokens"] = json!({
            "[CLS]": {"ids": [2], "tokens": ["[CLS]"]},
            "[SEP]": {"ids": [3], "tokens": ["[SEP]"]},
            "[MASK]": {"ids": [4294967296u64], "tokens": ["[MASK]"]}
        });
        assert!(matches!(
            from_json(&v),
            Err(KatakuriError::InvalidTokenizerJson { .. })
        ));
        // Structurally-valid extra entry accepts (non-template entries
        // are not vocab-checked).
        let mut v = tiny_json();
        v["post_processor"]["special_tokens"] = json!({
            "[CLS]": {"ids": [2], "tokens": ["[CLS]"]},
            "[SEP]": {"ids": [3], "tokens": ["[SEP]"]},
            "[MASK]": {"ids": [9], "tokens": ["[MASK]"]}
        });
        assert!(from_json(&v).is_ok());
        // Genuinely absent map still falls back (committed-fixture shape).
        let mut v = tiny_json();
        v["post_processor"]
            .as_object_mut()
            .unwrap()
            .remove("special_tokens");
        assert!(from_json(&v).is_ok());
    }
}
