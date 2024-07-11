//! # Shoco
//!
//! This is a Rust implementation of the Shoco compression algorithm.
//! Shoco is a simple and fast compressor for short strings.
//! 
//! This version is based on the original C code by [Ed-von-Schleck](https://github.com/Ed-von-Schleck/shoco).
//!
//! It is intended to be used in embedded systems rather than as a stand-alone command.
//! As such you can train your own model using the `gen_model` function.
//!
//! ## Example
//!
//! ```rust
//! use shoco::{shoco_compress, shoco_decompress, ShocoModel};
//! let model = ShocoModel::default();
//! let original = "This is a very simple test";
//! let compressed = shoco_compress(original, &model);
//! let decompressed = shoco_decompress(&compressed, &model).unwrap();
//! assert_eq!(original, decompressed);
//! ```

mod model;
mod gen_model;

pub use model::{ShocoModel, Pack};
pub use gen_model::{GenShocoModel, Split, Strip};
use std::path::Path;


fn decode_header(val: u8) -> i32 {
    let mut i = -1;
    let mut val = val as i8;
    while val < 0 {
        val <<= 1;
        i += 1;
    }
    i
}

struct Code(u32);

impl Code {
    fn bytes(&self) -> [u8; 4] {
        self.0.to_be_bytes()
    }
}

fn check_indices(indices: &[i8], pack_n: usize, model : &ShocoModel) -> bool {
    for i in 0..model.packs[pack_n].bytes_unpacked {
        if indices[i as usize] as i16 > model.packs[pack_n].masks[i as usize] {
            return false;
        }
    }
    true
}

fn find_best_encoding(indices: &[i8], n_consecutive: usize, model : &ShocoModel) -> i32 {
    for p in (0..model.packs.len()).rev() {
        if n_consecutive >= model.packs[p].bytes_unpacked as usize && check_indices(indices, p as usize, model) {
            return p as i32;
        }
    }
    -1
}

/// Compress a string using a Shoco model
///
/// # Arguments
/// * `original` - The string to compress
/// * `model` - The model to use for compression
///
/// # Returns
/// A vector of bytes representing the compressed string
pub fn shoco_compress(original: &str, model : &ShocoModel) -> Vec<u8> {
    let mut out = Vec::new();
    let bytes = original.as_bytes();
    let mut in_idx = 0usize; // Now an index

    while in_idx < original.len() {

        let mut indices = vec![0; model.max_successor_n + 1];
        indices[0] = model.chr_ids_by_chr[bytes[in_idx] as usize];
        let mut last_chr_index = indices[0];
        if last_chr_index < 0 {
            last_resort(bytes, &mut in_idx, &mut out);
            continue;
        }

        let rest = original.len() - in_idx;
        let mut n_consecutive = 1;
        while n_consecutive <= model.max_successor_n {
            if n_consecutive == rest {
                break;
            }

            let current_index = model.chr_ids_by_chr[bytes[(in_idx + n_consecutive) as usize] as usize];
            if current_index < 0 {
                break;
            }

            let successor_index = model.successor_ids_by_chr_id_and_chr_id[last_chr_index as usize][current_index as usize];
            if successor_index < 0 {
                break;
            }

            indices[n_consecutive as usize] = successor_index;
            last_chr_index = current_index;

            n_consecutive += 1;
        }
        if n_consecutive < 2 {
            last_resort(bytes, &mut in_idx, &mut out);  
            continue;
        }

        let pack_n = find_best_encoding(&indices, n_consecutive, model);
        if pack_n >= 0 {
            //if o_idx + model.packs[pack_n as usize].bytes_packed > out.len() {
            //    return out;
            //}

            let mut code = Code(model.packs[pack_n as usize].word);
            for i in 0..model.packs[pack_n as usize].bytes_unpacked {
                code.0 |= (indices[i as usize] as u32) << model.packs[pack_n as usize].offsets[i as usize];
            }

            let code_bytes = code.bytes();
            for i in 0..model.packs[pack_n as usize].bytes_packed {
                out.push(code_bytes[i as usize]);
            }

            in_idx += model.packs[pack_n as usize].bytes_unpacked;
        } else {
            last_resort(bytes, &mut in_idx, &mut out);
            continue;
        }
    }
    return out;
}

fn last_resort(o : &[u8], in_idx : &mut usize, out : &mut Vec<u8>) {
    if (o[*in_idx] & 0x80) != 0 {
        // Non-ASCII insert sentinel value
        out.push(0x00);
    } 
    out.push(o[*in_idx]);
    *in_idx += 1;
}

/// Decompress a vector of bytes using a Shoco model
///
/// # Arguments
/// * `original` - The vector of bytes to decompress
/// * `model` - The model to use for decompression
///
/// # Returns
/// A Result containing the decompressed string or an error if the decompression fails
pub fn shoco_decompress(original : &[u8], model : &ShocoModel) -> Result<String, std::string::FromUtf8Error> {
    let mut out : Vec<u8> = Vec::new();
    let mut in_idx = 0usize;

    while in_idx < original.len() {
        let mark = decode_header(original[in_idx]);
        if mark < 0 {
            if original[in_idx] == 0x00 {
                in_idx += 1;
            }
            out.push(original[in_idx]);
            in_idx += 1;
        } else {
            let mark = mark as usize;
            if in_idx + model.packs[mark].bytes_packed > original.len() {
                return String::from_utf8(out);
            }

            let mut code_bytes = [0u8; 4];
            for i in 0..model.packs[mark].bytes_packed {
                code_bytes[i as usize] = original[in_idx + i as usize];
            }
            
            let code = u32::from_be_bytes(code_bytes);

            let mut offset = model.packs[mark].offsets[0];
            let mut mask = model.packs[mark].masks[0];
            let mut last_chr = model.chrs_by_chr_id[((code >> offset) & (mask as u32)) as usize];
            out.push(last_chr);

            for i in 1..model.packs[mark as usize].bytes_unpacked {
                offset = model.packs[mark as usize].offsets[i as usize];
                mask = model.packs[mark as usize].masks[i as usize];
                last_chr = model.chrs_by_chr_and_successor_id[(last_chr - model.min_chr) as usize][(code >> offset) as usize & mask as usize];
                out.push(last_chr);
            }

            in_idx += model.packs[mark as usize].bytes_packed;
        }
    }
    return String::from_utf8(out);
}

/// Generate a new model from a list of files
///
/// # Arguments
/// * `files` - A vector of paths to files to use for training the model
///
/// # Returns
/// A builder for generating a new model
///
/// # Example
/// ```rust
/// use shoco::gen_model;
/// let model = gen_model(vec!["training_data/pride_and_prejudice.txt"]).generate();
/// ```
pub fn gen_model<P: AsRef<Path>>(files : Vec<P>) -> GenShocoModel {
    GenShocoModel::from_files(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shoco_compress() {
        let original = "This is a very simple test";
        let model = ShocoModel::default();
        let compressed = shoco_compress(original, &model);
        assert_eq!(compressed, 
            vec![0x54, 0x96, 0x73, 0x20, 0x89, 0x20, 0x61, 0x20, 0x76, 0x80, 0x79, 0x20, 0xd0, 0xdd, 0xa4, 0x20, 0xc8, 0x99 ]);
    }

    #[test]
    fn test_shoco_decompress() {
        let compressed = vec![0x54, 0x96, 0x73, 0x20, 0x89, 0x20, 0x61, 0x20, 0x76, 0x80, 0x79, 0x20, 0xd0, 0xdd, 0xa4, 0x20, 0xc8, 0x99 ];
        let model = ShocoModel::default();
        let decompressed = shoco_decompress(&compressed, &model).unwrap();
        assert_eq!(decompressed, "This is a very simple test");
    }
}
