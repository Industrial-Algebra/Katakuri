## S3 model fixtures (`tiny-config.json`, `tiny-model.safetensors`)

Deterministic synthetic BERT: `num_hidden_layers=2, hidden_size=8, vocab_size=32,
max_position_embeddings=16, type_vocab_size=2, intermediate_size=16,
num_attention_heads=4`. Weight set: 5 embeddings + 16/layer × 2 layers + 2
pooler = **39 F32 weights**, plus the `embeddings.position_ids` I64 `[1,16]`
non-weight buffer (40 manifest entries total). All-zero payloads — these
fixtures exercise **layout validation only** (names, dtypes, shapes, set
completeness), never numeric behavior.

Mirror of the real `sentence-transformers/all-MiniLM-L6-v2` shape family
(H=384, L=6, V=30522, I=1536, 104 entries), used by the env-gated
`real_file` tests. Values: regenerated only by editing the generator in the
S3 plan doc; treat as immutable inputs.
