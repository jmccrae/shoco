use std::path::Path;
use crate::model::{ShocoModel, Pack};
use std::collections::HashMap;
use crate::model::MAX_CONSECUTIVES;
use ordermap::OrderMap as OrderedDict;

const WHITESPACE : &[u8] = b" \t\n\r\x0b\x0c\xc2\xad";
const PUNCTUATION : &[u8] = b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

fn accumulate(seq : &[u32], start : u32) -> Vec<u32> {
    let mut total = start;
    let mut result = Vec::new();
    for elem in seq {
        total += elem;
        result.push(total);
    }
    result
}

macro_rules! structure {
    ($name:ident) => {
        impl $name {
            fn header(&self) -> u32 {
                self.0[0]
            }

            //fn lead(&self) -> u32 {
            //    self.0[1]
            //}

            //fn successors(&self) -> &[u32] {
            //    &self.0[2..]
            //}

            fn consecutive(&self) -> &[u32] {
                &self.0[1..]
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Bits(Vec<u32>);

impl Bits {
    fn new(bitlist : &[u32]) -> Bits {
        Bits(bitlist.to_vec())
    }

    fn lead(&self) -> u32 {
        self.0[1]
    }
}

structure!(Bits);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Masks(Vec<u32>);

impl Masks {
    fn new(bitlist : &[u32]) -> Masks {
        Masks(bitlist.iter().map(|bits| (1 << bits) - 1).collect())
    }
}

structure!(Masks);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Offsets(Vec<u32>);

impl Offsets {
    fn new(bitlist : &[u32]) -> Offsets {
        let inverse = accumulate(bitlist, 0);
        let offsets = inverse.iter().map(|offset| 32 - offset).collect();
        Offsets(offsets)
    }
}

structure!(Offsets);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Encoding {
    bits: Bits,
    masks: Masks,
    offsets: Offsets,
    packed: usize,
    size: usize,
    unpacked: usize
}

impl Encoding {
    fn new(bitlist : &[u32]) -> Encoding {
        let bits = Bits::new(bitlist);
        let masks = Masks::new(bitlist);
        let offsets = Offsets::new(bitlist);
        let packed = (bitlist.iter().sum::<u32>() / 8) as usize;
        let size = bitlist.iter().filter(|bits| **bits > 0).count();
        let unpacked = size - 1;
        Encoding {
            bits,
            masks,
            offsets,
            packed,
            size,
            unpacked
        }
    }

    fn header_code(&self) -> u32 {
        ((1 << self.bits.header()) - 2) << (8 - self.bits.header())
    }

    fn header_mask(&self) -> u32 {
        self.masks.header() << (8 - self.bits.header())
    }

    fn word(&self) -> u32 {
        ((1 << self.bits.header()) - 2) << self.offsets.header()
    }

    fn can_encode(&self, part : &[u8], successors : &OrderedDict<u8, Vec<u8>>, chrs_indices : &OrderedDict<u8, usize>) -> bool {
        let lead_index = chrs_indices.get(&part[0]).map(|x| *x);
        if let Some(lead_index) = lead_index {
            if lead_index > (1 << self.bits.header()) {
                return false;
            }
            let mut last_char = part[0];
            for (bits, char) in self.bits.consecutive().iter().zip(part[1..].iter()) {
                if !successors[&last_char].contains(char) {
                    return false;
                }
                let successor_index = successors[&last_char].iter().position(|c| c == char).unwrap();
                if successor_index > (1 << bits) {
                    return false;
                }
                last_char = *char;
            }
            true
        } else {
            false
        }
    }
}

fn pack_structures() -> Vec<(usize, Vec<Vec<u32>>)> {
    vec![
        (1, vec![
         vec![2, 4, 2],
         vec![2, 3, 3],
         vec![2, 4, 1, 1],
         vec![2, 3, 2, 1],
         vec![2, 2, 2, 2],
         vec![2, 3, 1, 1, 1],
         vec![2, 2, 2, 1, 1],
         vec![2, 2, 1, 1, 1, 1],
         vec![2, 1, 1, 1, 1, 1, 1],
        ]),
        (2, vec![
         vec![3, 5, 4, 2, 2],
         vec![3, 5, 3, 3, 2],
         vec![3, 4, 4, 3, 2],
         vec![3, 4, 3, 3, 3],
         vec![3, 5, 3, 2, 2, 1],
         vec![3, 5, 2, 2, 2, 2],
         vec![3, 4, 4, 2, 2, 1],
         vec![3, 4, 3, 2, 2, 2],
         vec![3, 4, 3, 3, 2, 1],
         vec![3, 4, 2, 2, 2, 2],
         vec![3, 3, 3, 3, 2, 2],
         vec![3, 4, 3, 2, 2, 1, 1],
         vec![3, 4, 2, 2, 2, 2, 1],
         vec![3, 3, 3, 2, 2, 2, 1],
         vec![3, 3, 2, 2, 2, 2, 2],
         vec![3, 2, 2, 2, 2, 2, 2],
         vec![3, 3, 3, 2, 2, 1, 1, 1],
         vec![3, 3, 2, 2, 2, 2, 1, 1],
         vec![3, 2, 2, 2, 2, 2, 2, 1],
        ]),
        (4, vec![
         vec![4, 5, 4, 4, 4, 3, 3, 3, 2],
         vec![4, 5, 5, 4, 4, 3, 3, 2, 2],
         vec![4, 4, 4, 4, 4, 4, 3, 3, 2],
         vec![4, 4, 4, 4, 4, 3, 3, 3, 3],
        ])
            ]
}

fn encodings() -> Vec<(usize, Vec<Encoding>)> {
    pack_structures().iter().map(|(packed, bitlists)| {
        let s : Vec<Encoding> = bitlists.iter().map(|bitlist| Encoding::new(bitlist.as_slice())).collect();
        (*packed, s)
    }).collect()
}


struct BigramIterator<'a> {
    iter : std::slice::Iter<'a, u8>,
    last : Option<u8>
}

impl <'a> BigramIterator<'a> {
    fn new(iter : std::slice::Iter<'a, u8>) -> BigramIterator<'a> {
        BigramIterator {
            iter,
            last : None
        }
    }
}

impl <'a> Iterator for BigramIterator<'a> {
    type Item = (u8, u8);

    fn next(&mut self) -> Option<(u8, u8)> {
        let last = self.last;
        let next = self.iter.next();
        if let Some(next) = next {
            self.last = Some(*next);
            if let Some(last) = last {
                Some((last, *next))
            } else {
                self.next()
            }
        } else {
            None
        }
    }
}

fn bigrams<'a>(sequence : &'a [u8]) -> BigramIterator<'a> {
    BigramIterator::new(sequence.iter())
}
//    let mut iter = sequence.iter();
//    let mut last = *iter.next().unwrap();
//    let mut result = Vec::new();
//    for item in iter {
//        result.push((last, *item));
//        last = *item;
//    }
//    result
//}

//fn format_int_line<U : std::fmt::Display>(items : &[U]) -> String {
//    items.iter().map(|k| format!("{}", k)).collect::<Vec<String>>().join(", ")
//}

fn u32_line(items: &[u32]) ->  [u32; MAX_CONSECUTIVES] {
    let mut result = [0; 8];
    for i in 0..items.len() {
        result[i] = items[i];
    }
    result
}

fn i16_line(items: &[u32]) -> [i16; MAX_CONSECUTIVES] {
    let mut result = [0; 8];
    for i in 0..items.len() {
        result[i] = items[i] as i16;
    }
    result
}

fn conv_successors_reversed(successors_reversed : OrderedDict<u8, Vec<Option<usize>>>) -> Vec<Vec<i8>> {
    let mut result = Vec::new();
    for successors in successors_reversed.values() {
        result.push(successors.iter().map(|s| s.map(|x| x as i8).unwrap_or(-1)).collect());
    }
    result
}

fn chunkinator<P : AsRef<Path>>(files : &[P], split : Split, strip : Strip) -> Vec<Vec<u8>> {
    let all_in : Vec<Vec<u8>> = files.iter().map(|filename| std::fs::read(filename).unwrap()).collect();

    let chunks = match split {
        Split::None => all_in,
        Split::Newline => {
            let mut result = Vec::new();
            for data in all_in {
                let iter = data.split(|c| *c == b'\n');
                for chunk in iter {
                    let mut v = chunk.to_vec();
                    if v.len() > 0 && v[v.len() - 1] == b'\r' {
                        v.pop();
                    }
                    result.push(v);
                }
            }
            result
        },
        Split::Whitespace => {
            let mut result = Vec::new();
            for data in all_in {
                let iter = data.split(|c| WHITESPACE.contains(c));
                for chunk in iter {
                    result.push(chunk.to_vec());
                }
            }
            result
        }
    };

    let mut result = Vec::new();
    for chunk in chunks {
        let chunk = match strip {
            Strip::Whitespace => chunk.into_iter().skip_while(|c| WHITESPACE.contains(c)).collect(),
            Strip::Punctuation => chunk.into_iter().skip_while(|c| PUNCTUATION.contains(c)).collect(),
            _ => chunk
        };
        if chunk.len() > 0 {
            result.push(chunk);
        }
    }
    result
}

pub enum Strip {
    Whitespace,
    Punctuation,
    None,
}

pub enum Split {
    Newline,
    Whitespace,
    None,
}

pub struct GenShocoModel<P : AsRef<Path>> {
    files : Vec<P>,
    split : Split,
    strip : Strip,
    max_leading_char_bits : u32,
    max_successor_bits : u32,
    encoding_types : usize,
    optimize_encoding : bool,
    compatibility : bool
}

struct Counter<T> {
    counts : OrderedDict<T, u32>
}

impl<T: std::hash::Hash + std::cmp::Eq> Counter<T> {
    fn new() -> Counter<T> {
        Counter {
            counts : OrderedDict::new()
        }
    }

    fn increment(&mut self, item : T) {
        let count = self.counts.entry(item).or_insert(0);
        *count += 1;
    }

    fn most_common(&self, n : usize) -> Vec<(T, u32)>  where T: Clone {
        let mut items : Vec<(T, u32)> = self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.into_iter().take(n).collect()
    }
}

fn most_common(freqs : &[usize], n : usize) -> Vec<(u8, usize)> {
    let mut results = Vec::new();
    for (i, freq) in freqs.iter().enumerate() {
        if results.len() < n {
            results.push((i as u8, *freq));
        } else {
            // Could be quicker
            results.push((i as u8, *freq));
            results.sort_by_key(|x| -(x.1 as i32));
            results.pop();
        }
    }
    results
}

impl<P: AsRef<Path>> GenShocoModel<P> {
    pub fn new(files : Vec<P>) -> GenShocoModel<P> {
        GenShocoModel {
            files,
            split: Split::Newline,
            strip: Strip::Whitespace,
            max_leading_char_bits: 5,
            max_successor_bits: 4,
            encoding_types: 3,
            optimize_encoding: false,
            compatibility: false
        }
    }

    pub fn split(self, split : Split) -> GenShocoModel<P> {
        GenShocoModel {
            split,
            ..self
        }
    }

    pub fn strip(self, strip : Strip) -> GenShocoModel<P> {
        GenShocoModel {
            strip,
            ..self
        }
    }

    pub fn max_leading_char_bits(self, max_leading_char_bits : u32) -> GenShocoModel<P> {
        GenShocoModel {
            max_leading_char_bits,
            ..self
        }
    }

    pub fn max_successor_bits(self, max_successor_bits : u32) -> GenShocoModel<P> {
        GenShocoModel {
            max_successor_bits,
            ..self
        }
    }


    pub fn encoding_types(self, encoding_types : usize) -> GenShocoModel<P> {
        GenShocoModel {
            encoding_types,
            ..self
        }
    }

    pub fn optimize_encoding(self, optimize_encoding : bool) -> GenShocoModel<P> {
        GenShocoModel {
            optimize_encoding,
            ..self
        }
    }

    pub fn compatibility(self, compatibility : bool) -> GenShocoModel<P> {
        GenShocoModel {
            compatibility,
            ..self
        }
    }

    pub fn generate(self) -> Result<ShocoModel, std::io::Error> {
        let chars_count = 1 << self.max_leading_char_bits;
        //let successors_count = 1 << self.max_successor_bits;

        let (successors, chunks) = if self.compatibility {
            let mut bigram_counters : OrderedDict<u8, Counter<u8>> = OrderedDict::new();
            let mut first_char_counter = Counter::new();
            let chunks = chunkinator(self.files.as_slice(), self.split, self.strip);
            for chunk in chunks.iter() {
                let bgs = bigrams(chunk.as_slice());
                for bg in bgs {
                    let (a, b) = bg;
                    first_char_counter.increment(a);
                    let counter = bigram_counters.entry(a).or_insert_with(|| Counter::new());
                    counter.increment(b);
                }
            }

            let mut successors : OrderedDict<u8, Vec<u8>> = OrderedDict::new();
            for (char, _) in &first_char_counter.most_common(1 << self.max_leading_char_bits) {
                let mut successor_list = bigram_counters.get(char).unwrap().most_common(
                        1 << self.max_successor_bits).into_iter().
                    map(|(c, _)| { 
                        c
                    }).collect::<Vec<u8>>();
                successor_list.extend(std::iter::repeat(b'\0').take((1 << self.max_successor_bits) - successor_list.len()));
                successors.insert(*char, successor_list);
            }
            (successors, chunks)
        } else {
            //let mut bigram_counters : OrderedDict<u8, Counter<u8>> = OrderedDict::new();
            //let mut first_char_counter = Counter::new();
            let mut bigram_counters : OrderedDict<u8, [usize; 255]> = OrderedDict::new();
            let mut first_char_counter = [0; 256];
            let chunks = chunkinator(self.files.as_slice(), self.split, self.strip);
            for chunk in chunks.iter() {
                let bgs = bigrams(chunk.as_slice());
                for bg in bgs {
                    let (a, b) = bg;
                    first_char_counter[a as usize] += 1;
                    let counter = bigram_counters.entry(a).or_insert_with(|| [0; 255]);
                    counter[b as usize] += 1;
                }
            }

            let mut successors : OrderedDict<u8, Vec<u8>> = OrderedDict::new();
            for (char, _) in most_common(&first_char_counter, 1 << self.max_leading_char_bits) {
                let mut successor_list = most_common(bigram_counters.get(&char).unwrap(),
                        1 << self.max_successor_bits).into_iter().
                    map(|(c, _)| { 
                        c
                    }).collect::<Vec<u8>>();
                successor_list.extend(std::iter::repeat(b'\0').take((1 << self.max_successor_bits) - successor_list.len()));
                successors.insert(char, successor_list);
            }
            (successors, chunks)
        };
        let max_chr = *successors.keys().max().unwrap() + 1;
        let min_chr = *successors.keys().min().unwrap();

        let chrs_indices = successors.keys().enumerate().map(|(i, k)| (*k, i)).collect::<OrderedDict<u8, usize>>();
        let chrs_reversed = {
            let mut c = [0;256];
            for i in 0u8..=255u8 {
                c[i as usize] = chrs_indices.get(&i).map(|x| *x as i8).unwrap_or(-1);
            }
            c
        };

        let mut successors_reversed : OrderedDict<u8, Vec<Option<usize>>> = OrderedDict::new();
        for (char, successor_list) in successors.iter() {
            let mut reversed = Vec::new();
            for _ in 0..chars_count {
                reversed.push(None);
            }
            let s_indices = successor_list.iter().enumerate().map(|(i, s)| (*s, i)).collect::<OrderedDict<u8, usize>>();
            for (i, s) in successors.keys().enumerate() {
                reversed[i] = s_indices.get(s).map(|x| *x);
            }
            successors_reversed.insert(*char, reversed);
        }

        let zeros_line = vec![0; 1 << self.max_successor_bits];
        let chrs_by_chr_and_successor_id = (min_chr..max_chr).map(|i| successors.get(&i).unwrap_or(&zeros_line).to_vec()).collect::<Vec<Vec<u8>>>();

        let (max_encoding_len, best_encodings) = if self.optimize_encoding {
            let mut counters : HashMap<usize, Counter<Encoding>> = HashMap::new();

            for (packed, _) in encodings().into_iter().take(self.encoding_types) {
                counters.insert(packed, Counter::new());
            }

            for chunk in chunks.iter() {
                for i in 0..chunk.len() {
                    for (packed, encodings) in encodings().into_iter().take(self.encoding_types) {
                        for encoding in encodings {
                            if (encoding.bits.lead() > self.max_leading_char_bits) || 
                                (*encoding.bits.consecutive().iter().max().unwrap() > self.max_successor_bits) {
                                    continue;
                            }
                            if encoding.can_encode(&chunk[i..], &successors, &chrs_indices) {
                                let counter = counters.get_mut(&packed).unwrap();
                                counter.increment(encoding);
                            }
                        }
                    }
                }
            }

            let best_encodings_raw = counters.iter().map(|(packed, counter)| (*packed, counter.most_common(1)[0].0.clone())).collect::<Vec<(usize, Encoding)>>();
            let max_encoding_len = best_encodings_raw.iter().map(|(_, encoding)| encoding.size).max().unwrap();
            let best_encodings = best_encodings_raw.iter().map(|(_, encoding)|  {
                let mut e = encoding.bits.0.clone();
                e.extend(std::iter::repeat(0).take(MAX_CONSECUTIVES - encoding.size));
                Encoding::new(e.as_slice())
            }).collect::<Vec<Encoding>>();
            (max_encoding_len, best_encodings)
        } else {
            (8, vec![Encoding::new(&[2, 4, 2, 0, 0, 0, 0, 0, 0]),
            Encoding::new(&[3, 4, 3, 3, 3, 0, 0, 0, 0]),
            Encoding::new(&[4, 5, 4, 4, 4, 3, 3, 3, 2])].into_iter().take(self.encoding_types).collect())
        };


        let packs = (0..self.encoding_types).map(|i| {
            Pack {
                word: best_encodings[i].word(),
                bytes_packed: best_encodings[i].packed,
                bytes_unpacked: best_encodings[i].unpacked,
                offsets: u32_line(best_encodings[i].offsets.consecutive()),
                masks: i16_line(best_encodings[i].masks.consecutive()),
                header_mask: best_encodings[i].header_mask() as u8,
                header: best_encodings[i].header_code() as u8
            }
        }).collect();

        let chrs = successors.keys().map(|x| *x).collect();

        let model = ShocoModel {
            min_chr,
            max_chr,
            chrs_by_chr_id: chrs,
            chr_ids_by_chr: chrs_reversed,
            successor_ids_by_chr_id_and_chr_id : conv_successors_reversed(successors_reversed),
            chrs_by_chr_and_successor_id,
            packs,
            max_successor_n: max_encoding_len - 1
        };
        Ok(model)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gen_model() {
        let model = GenShocoModel::new(vec![
            "training_data/dorian_gray.txt",
            "training_data/metamorphosis.txt",
            "training_data/pride_and_prejudice.txt"])
            .compatibility(true)
            .generate().unwrap();

        assert_eq!(model.min_chr, 32);
        assert_eq!(model.max_chr, 122);

        assert_eq!(model.chrs_by_chr_id,
            vec![32, 101, 116, 97, 111, 110, 105, 104, 115, 114, 100, 108, 117, 109, 99, 119, 121, 102, 103, 44, 112, 98, 46, 118, 107, 73, 34, 45, 72, 77, 84, 39]);

        assert_eq!(model.chr_ids_by_chr,
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 26, -1, -1, -1, -1, 31, -1, -1, -1, -1, 19, 27, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 28, 25, -1, -1, -1, 29, -1, -1, -1, -1, -1, -1, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 21, 14, 10, 1, 17, 18, 7, 6, -1, 24, 11, 13, 5, 4, 20, -1, 9, 8, 2, 12, 23, 15, -1, 16, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);
        assert_eq!(model.successor_ids_by_chr_id_and_chr_id,
            vec![
            vec![12, -1, 0, 1, 5, 13, 6, 2, 4, -1, 11, 14, -1, 7, 10, 3, -1, 9, -1, -1, 15, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![0, 8, 7, 5, -1, 2, 15, -1, 4, 1, 3, 6, -1, 10, 11, -1, 14, -1, -1, 9, -1, -1, 13, 12, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![1, 3, 6, 5, 2, -1, 4, 0, 13, 7, -1, 12, 9, -1, 15, 14, 11, -1, -1, 8, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![6, -1, 1, -1, -1, 0, 7, -1, 2, 3, 5, 4, -1, 10, 12, -1, 8, -1, 13, -1, 14, 11, -1, 9, 15, -1, -1, -1, -1, -1, -1, -1],
            vec![2, -1, 6, -1, 8, 1, 15, -1, 9, 3, 12, 10, 0, 5, -1, 7, -1, 4, -1, -1, 13, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1],
            vec![0, 3, 4, 10, 5, 11, 8, -1, 7, -1, 1, 14, -1, -1, 6, -1, 12, -1, 2, 9, -1, -1, 13, -1, 15, -1, -1, -1, -1, -1, -1, -1],
            vec![-1, 9, 2, 11, 4, 0, -1, -1, 1, 8, 7, 5, -1, 3, 6, -1, -1, 12, 10, -1, -1, 15, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![3, 0, 5, 1, 4, -1, 2, -1, 13, 7, -1, 14, 8, 12, -1, -1, 10, -1, -1, 6, -1, 15, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![0, 1, 2, 7, 5, -1, 4, 3, 6, -1, -1, 13, 8, 15, 12, 14, -1, -1, -1, 9, 11, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![1, 0, 7, 4, 3, 12, 2, -1, 5, 11, 9, 15, -1, 14, 13, -1, 6, -1, -1, 10, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![0, 1, -1, 6, 3, -1, 2, -1, 7, 9, 11, 12, 10, -1, -1, -1, 8, -1, 15, 4, -1, -1, 5, -1, -1, -1, -1, 14, -1, -1, -1, -1],
            vec![3, 0, 11, 7, 6, -1, 2, -1, 12, -1, 5, 1, 9, -1, -1, 15, 4, 8, -1, 10, -1, -1, 14, -1, 13, -1, -1, -1, -1, -1, -1, -1],
            vec![5, 9, 1, 11, -1, 4, 10, -1, 3, 0, 12, 2, -1, 13, 7, -1, -1, -1, 6, 15, 8, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![1, 0, -1, 2, 3, -1, 4, -1, 8, -1, -1, -1, 5, 12, -1, -1, 7, 14, -1, 10, 6, 11, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![12, 1, 4, 3, 0, -1, 6, 2, 15, 7, -1, 8, 9, -1, 11, -1, 10, -1, -1, -1, -1, -1, 14, -1, 5, -1, -1, -1, -1, -1, -1, -1],
            vec![5, 3, -1, 0, 4, 6, 1, 2, 9, 7, 12, 10, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, 11, -1, 15, -1, -1, 14, -1, -1, -1, -1],
            vec![0, 4, 6, -1, 1, -1, 7, -1, 5, -1, 8, 12, -1, -1, -1, -1, -1, -1, -1, 2, -1, 13, 3, -1, -1, -1, -1, 15, -1, -1, -1, 10],
            vec![0, 2, 8, 4, 1, -1, 5, -1, -1, 3, -1, 9, 7, -1, -1, -1, 13, 6, -1, 10, -1, -1, 11, -1, -1, -1, -1, 12, -1, -1, -1, -1],
            vec![0, 2, -1, 4, 3, 12, 6, 1, 9, 5, -1, 7, 11, -1, -1, -1, -1, -1, 13, 8, -1, 15, 10, -1, -1, -1, -1, 14, -1, -1, -1, -1],
            vec![0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, -1, -1, -1, 2],
            vec![7, 0, 8, 2, 3, -1, 6, 11, 10, 1, -1, 4, 9, -1, -1, -1, 12, -1, -1, 13, 5, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![12, 0, 10, 5, 2, -1, 7, -1, 8, 6, -1, 1, 3, 15, -1, -1, 4, -1, -1, -1, -1, 11, 14, 13, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, 2, -1, -1, -1, 1, 4, -1, -1, -1, 3],
            vec![-1, 0, -1, 2, 3, -1, 1, -1, -1, 6, -1, -1, 5, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![1, 0, -1, 13, -1, 2, 3, 5, 4, -1, -1, 8, -1, -1, -1, 14, 10, 9, -1, 6, -1, -1, 7, -1, -1, -1, -1, 11, -1, -1, -1, -1],
            vec![0, -1, 1, -1, -1, 2, -1, -1, 4, 15, -1, -1, -1, 6, -1, -1, -1, 3, -1, 7, -1, -1, 14, -1, -1, 8, -1, -1, -1, -1, -1, 5],
            vec![0, -1, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, -1, 1, -1, -1, 7, 6, 4, -1],
            vec![8, -1, 1, 2, -1, 9, 15, 5, 4, 7, 14, 13, -1, 12, 6, 10, -1, -1, -1, -1, 11, 3, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            vec![-1, 0, -1, 1, 2, -1, 3, -1, -1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![8, 4, -1, 3, 5, -1, 1, -1, -1, 0, -1, -1, 6, -1, -1, -1, 2, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            vec![10, 2, -1, 8, 1, -1, 5, 0, -1, 6, -1, -1, 4, -1, -1, 7, 11, -1, -1, 14, -1, -1, 15, -1, -1, -1, -1, -1, 9, -1, -1, -1],
            vec![2, 15, 1, 6, -1, -1, -1, -1, 0, 9, 7, 4, -1, 5, 3, -1, -1, -1, -1, 13, -1, -1, -1, 8, -1, -1, 14, -1, -1, -1, 10, -1] ]);

        assert_eq!(model.chrs_by_chr_and_successor_id, vec![
            vec![116, 97, 104, 119, 115, 111, 105, 109, 98, 102, 99, 100, 32, 110, 108, 112],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![32, 73, 89, 87, 84, 65, 77, 72, 79, 66, 78, 68, 116, 83, 97, 44],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![115, 116, 32, 99, 108, 109, 97, 100, 118, 114, 84, 65, 76, 44, 34, 101],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![32, 34, 39, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![45, 116, 97, 98, 115, 104, 99, 114, 32, 110, 119, 112, 109, 108, 100, 105],
            vec![32, 34, 46, 39, 45, 44, 63, 59, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![101, 97, 111, 105, 117, 65, 69, 121, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![32, 116, 110, 102, 115, 39, 109, 44, 73, 78, 95, 76, 69, 65, 46, 114],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![114, 105, 121, 97, 101, 111, 117, 89, 32, 46, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![104, 111, 101, 69, 117, 105, 114, 119, 97, 72, 32, 121, 82, 90, 44, 46],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![110, 116, 115, 114, 108, 100, 32, 105, 121, 118, 109, 98, 99, 103, 112, 107],
            vec![101, 108, 111, 117, 121, 97, 114, 105, 115, 106, 116, 98, 32, 118, 46, 109],
            vec![111, 101, 104, 97, 116, 107, 105, 114, 108, 117, 121, 99, 32, 113, 46, 115],
            vec![32, 101, 105, 111, 44, 46, 97, 115, 121, 114, 117, 100, 108, 59, 45, 103],
            vec![32, 114, 110, 100, 115, 97, 108, 116, 101, 44, 109, 99, 118, 46, 121, 105],
            vec![32, 111, 101, 114, 97, 105, 102, 117, 116, 108, 44, 46, 45, 121, 59, 63],
            vec![32, 104, 101, 111, 97, 114, 105, 108, 44, 115, 46, 117, 110, 103, 45, 98],
            vec![101, 97, 105, 32, 111, 116, 44, 114, 117, 46, 121, 33, 109, 115, 108, 98],
            vec![110, 115, 116, 109, 111, 108, 99, 100, 114, 101, 103, 97, 102, 118, 122, 98],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![101, 32, 110, 105, 115, 104, 44, 46, 108, 102, 121, 45, 59, 97, 119, 33],
            vec![101, 108, 105, 32, 121, 100, 111, 97, 102, 117, 44, 116, 115, 107, 46, 119],
            vec![101, 32, 97, 111, 105, 117, 112, 121, 115, 46, 44, 98, 109, 59, 102, 63],
            vec![32, 100, 103, 101, 116, 111, 99, 115, 105, 44, 97, 110, 121, 46, 108, 107],
            vec![117, 110, 32, 114, 102, 109, 116, 119, 111, 115, 108, 118, 100, 112, 107, 105],
            vec![101, 114, 97, 111, 108, 112, 105, 32, 116, 117, 115, 104, 121, 44, 46, 98],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![101, 32, 105, 111, 97, 115, 121, 116, 46, 100, 44, 114, 110, 99, 109, 108],
            vec![32, 101, 116, 104, 105, 111, 115, 97, 117, 44, 46, 112, 99, 108, 119, 109],
            vec![104, 32, 111, 101, 105, 97, 116, 114, 44, 117, 46, 121, 108, 115, 119, 99],
            vec![114, 116, 108, 115, 110, 32, 103, 99, 112, 101, 105, 97, 100, 109, 98, 44],
            vec![101, 105, 97, 111, 121, 117, 114, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![97, 105, 104, 101, 111, 32, 110, 114, 44, 115, 108, 46, 100, 59, 45, 107],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![32, 111, 44, 46, 101, 115, 116, 105, 100, 59, 39, 63, 108, 98, 33, 45] ]);

        assert_eq!(model.packs, vec![
            Pack { word: 0x80000000, bytes_packed: 1, bytes_unpacked: 2, offsets: [26, 24, 24, 24, 24, 24, 24, 24], masks: [15, 3, 0, 0, 0, 0, 0, 0], header_mask: 0xc0, header: 0x80 },
            Pack { word: 0xc0000000, bytes_packed: 2, bytes_unpacked: 4, offsets: [25, 22, 19, 16, 16, 16, 16, 16], masks: [15, 7, 7, 7, 0, 0, 0, 0], header_mask: 0xe0, header: 0xc0 },
            Pack { word: 0xe0000000, bytes_packed: 4, bytes_unpacked: 8, offsets: [23, 19, 15, 11, 8, 5, 2, 0], masks: [31, 15, 15, 15, 7, 7, 7, 3], header_mask: 0xf0, header: 0xe0 }
        ]);
    }
}
