// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Inspect a safetensors file: tensor count, names, dtypes, shapes.
//!
//! ```sh
//! cargo run --example inspect-safetensors -- tests/fixtures/tiny-model.safetensors
//! ```
//!
//! With no argument the example prints usage and exits non-zero. Parse
//! failures surface the crate's typed [`KatakuriError`], not a panic — the
//! validation contract rejects corrupt files with a described reason.

use katakuri::safetensors::SafeTensors;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: cargo run --example inspect-safetensors -- <file.safetensors>");
        std::process::exit(2);
    };

    if let Err(err) = inspect(&path) {
        // `KatakuriError`'s Display names the offending tensor and rule.
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn inspect(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::parse(&bytes)?;

    println!("file:        {path} ({} bytes)", bytes.len());
    println!("data region: {} bytes", st.data_len());
    println!("tensors:     {}", st.tensor_count());
    if st.metadata().is_empty() {
        println!("metadata:    (none)");
    } else {
        for (key, value) in st.metadata() {
            println!("metadata:    {key} = {value}");
        }
    }
    println!();

    // Widths sized for any BERT-family manifest; longer names keep full width.
    let name_width = st.names().map(str::len).max().unwrap_or(4).max(4);
    println!(
        "{:>4}  {:<name_width$}  {:<5}  {:<16}  {:>10}",
        "#", "name", "dtype", "shape", "bytes"
    );
    for (i, name) in st.names().enumerate() {
        let tensor = st.tensor(name)?;
        let shape = format!("{:?}", tensor.shape());
        println!(
            "{:>4}  {:<name_width$}  {:<5}  {:<16}  {:>10}",
            i + 1,
            name,
            format!("{:?}", tensor.dtype()),
            shape,
            tensor.bytes().len()
        );
    }

    println!();
    println!("(every span above was checked against shape × dtype at parse time)");
    Ok(())
}
