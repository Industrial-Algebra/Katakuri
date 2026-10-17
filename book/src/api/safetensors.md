# `safetensors`

Weight parse + validate: dtype, shape, layout, and manifest checks over the
borrowed input bytes, plus a device-upload seam over `zunesha::Device`. No
math.

## Format

A safetensors file is an 8-byte little-endian `u64` header length `N`, then
`N` bytes of JSON header, then the data buffer. Header keys are tensor names
(or `__metadata__`); each tensor entry is
`{"dtype", "shape", "data_offsets": [start, end]}` with offsets relative to
the start of the data region.

## Parsing

```rust
use katakuri::safetensors::SafeTensors;

let bytes = std::fs::read("model.safetensors")?;
let st = SafeTensors::parse(&bytes)?;

println!("{} tensors, {} data bytes", st.tensor_count(), st.data_len());
for name in st.names() {
    let t = st.tensor(name)?;
    println!("{name}: {:?} {:?} ({} bytes)", t.dtype(), t.shape(), t.bytes().len());
}
```

The parse is zero-copy: `SafeTensors` borrows the slice and tensor views are
`&[u8]` windows into it. Nothing here needs a device, which is why GPU-less
hardware is the first-class path.

A minimal valid file (the doc-test in `lib.rs`) is eight bytes of header
length `2` followed by `{}`:

```rust
let bytes: &[u8] = &[2, 0, 0, 0, 0, 0, 0, 0, b'{', b'}'];
let st = SafeTensors::parse(bytes)?;
assert_eq!(st.tensor_count(), 0);
```

## Validation contract

All validation happens at parse time. Every tensor reachable afterwards
satisfies:

| Rule | Violation |
|---|---|
| header length ≤ 100 MB (DoS guard, matches the reference) | `HeaderTooLarge` |
| header length fits the file | `HeaderLengthExceedsFile` |
| header is a JSON object | `HeaderParse` / `HeaderNotAnObject` |
| `__metadata__`, when present, is a string→string map | `InvalidMetadata` |
| entries carry exactly `dtype`, `shape`, `data_offsets` | `InvalidTensorInfo` |
| dtype is a safetensors dtype | `InvalidDtype` |
| `start ≤ end ≤ data_len` | `OffsetOutOfBounds` |
| `end − start == elements × dtype_size` (checked u64) | `SizeMismatch` / `SizeOverflow` |
| no two spans overlap (gaps allowed) | `OverlappingTensors` |

Gaps are legal because real files have them; overlaps never are, because they
would make two named tensors aliases of the same bytes. Zero-length spans are
filtered out of the overlap check — an empty tensor claims no bytes and cannot
conflict (review finding 1).

## Types

`Dtype` is the safetensors set: `Bool`, `U8`, `I8`, `F8E5M2`, `F8E4M3`,
`I16`, `U16`, `F16`, `Bf16`, `I32`, `U32`, `F32`, `F64`, `I64`, `U64`, with
`byte_size()` and `parse()` (canonical string, case-sensitive).

`Tensor` exposes `dtype()`, `shape()`, `data_offsets()`, `bytes()`, and
`elem_count()`. `SafeTensors` exposes `names()`, `tensor_count()`,
`metadata()`, `data_len()`, and `tensor(name)`.

## Uploading to a device

```rust
let st = SafeTensors::parse(&bytes)?;

match st.upload_to_device("embeddings.word_embeddings.weight", &device) {
    Ok(device_tensor) => {
        // device_tensor.buffer: zunesha::Buffer — GPU-resident weights.
        // device_tensor.meta:   the manifest entry the bytes were validated against.
    }
    Err(katakuri::KatakuriError::Device(err)) => {
        // Includes NoDevice: the typed signal that no GPU backend exists —
        // the host views above stay valid and usable.
    }
    Err(other) => return Err(other),
}
```

The host view remains valid after an upload, so the same weights can feed a
precision CPU path or a Borsalino GPU path. Alignment and lifecycle
obligations are Zunesha's; Katakuri passes buffers through and never restates
them.
