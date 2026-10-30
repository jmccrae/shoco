# shoco-dyn

A Rust implementation of [shoco](https://github.com/Ed-von-Schleck/shoco), a
compressor for short strings, that can **train compression models on your own
text at runtime**.

shoco compresses each string on its own, using a model of which characters are
common and which tend to follow each other. Most shoco implementations compile
in a single model trained on English. `shoco-dyn` includes that model, and
also ports shoco's model generator (originally a Python script), so you can
train a model on your own data, store it alongside the compressed text, and
load it again later.

## Why train your own model?

shoco's default model is trained on English, and on other languages it often
makes text *larger*. In the evaluation of the
[Cuaċ corpus format](https://aclanthology.org/2025.ldk-1.17/) (McCrae et al.,
2025), which uses this crate, text from the Universal Dependencies corpora was
compressed with each method. Sizes are relative to the uncompressed text;
lower is better:

| Language | smaz | shoco, default model | shoco, generated model |
|---|---|---|---|
| English | 84.96% | 87.97% | 84.88% |
| German | 87.02% | 88.66% | 82.02% |
| Spanish | 81.61% | 87.07% | 74.74% |
| Turkish | 100.11% | 99.57% | 84.24% |
| Russian | 108.61% | 153.17% | 78.86% |
| Arabic | 119.35% | 158.54% | 71.97% |
| Hebrew | 117.53% | 180.62% | 73.88% |
| **Average (24 languages)** | 98.69% | 112.43% | **88.05%** |

A generated model needs enough text to pay for storing it: on a very small
corpus (Old Irish, 27 KB) it increased the size (126.24%). For languages with
very many characters, such as Chinese and Japanese, no short-string method
reduced the size.

## Usage

```toml
[dependencies]
shoco-dyn = "0.2"
```

Compress with the default model:

```rust
use shoco_dyn::{compress, decompress, ShocoModel};

let model = ShocoModel::default();
let compressed = compress("This is a very simple test", &model);
assert_eq!(decompress(&compressed, &model).unwrap(), "This is a very simple test");
```

Train a model on your own text, from files or from any iterator of byte
strings:

```rust
use shoco_dyn::{compress, decompress, GenShocoModel, Split};

let model = GenShocoModel::from_files(vec!["corpus.txt"])
    .split(Split::Newline)
    .generate()?;

let compressed = compress("Ní neart go cur le chéile", &model);
assert_eq!(decompress(&compressed, &model).unwrap(), "Ní neart go cur le chéile");
```

All of `ShocoModel`'s fields are public, so a model can be stored with the
data it compressed and rebuilt to decompress it.

`decompress` returns `DecompressError::Malformed` for data that isn't valid
for the model (for example, truncated or corrupt input), and
`DecompressError::Utf8` if the result isn't valid UTF-8. It never panics.

## Compatibility

`ShocoModel::default()` is the model from the original C implementation, so
for strings without NUL bytes the output is byte-identical to C shoco and to
the [`shoco`](https://crates.io/crates/shoco) crate. The C implementation stops
at the first NUL byte; `shoco-dyn` instead escapes it, in a form the C
decompressor reads correctly.

Versions of this crate before 0.2 used a slightly different default model.
Use `ShocoModel::legacy_default()` to read data compressed with it.

## Performance

Implementations with a single compiled-in model, such as the `shoco` crate,
can use fixed lookup tables and are around 15–25% faster than `shoco-dyn`,
which reads its tables from the model at runtime. That's the cost of
supporting trained models, which can make the output much smaller, as above.

## Licence

MIT. Based on [shoco](https://github.com/Ed-von-Schleck/shoco) by Christian
Schramm, including a port of its model generator.

## Citation

If you use this in research, please cite:

> John P. McCrae, Bernardo Stearns, Alamgir Munir Qazi, Shubhanker Banerjee
> and Atul Kr. Ojha. 2025. Cuaċ: Fast and Small Universal Representations of
> Corpora. In *Proceedings of the 5th Conference on Language, Data and
> Knowledge (LDK 2025)*, pages 153–161.
