use shoco::{compress, decompress, DecompressError, GenShocoModel, ShocoModel};

/// Small deterministic pseudo-random generator, so failures are reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }

    fn below(&mut self, n : usize) -> usize {
        (self.next_u8() as usize * 256 + self.next_u8() as usize) % n
    }

    fn bytes(&mut self, max_len : usize) -> Vec<u8> {
        let len = self.below(max_len + 1);
        (0..len).map(|_| self.next_u8()).collect()
    }

    fn text(&mut self, max_len : usize) -> String {
        let alphabet : Vec<char> = "etaoinshrdlucmfwypvbgkjqxz ETAOINSHRDLU.,;:!?'\"-\n\t0123456789éü日\0"
            .chars().collect();
        let len = self.below(max_len + 1);
        (0..len).map(|_| alphabet[self.below(alphabet.len())]).collect()
    }
}

/// The default models and one generated from the training data
fn models() -> Vec<ShocoModel> {
    vec![
        ShocoModel::default(),
        ShocoModel::legacy_default(),
        GenShocoModel::from_files(vec!["training_data/metamorphosis.txt"]).generate().unwrap(),
    ]
}

#[test]
fn decompress_never_panics_on_short_input() {
    for model in models() {
        decompress(&[], &model).unwrap();
        for a in 0..=255u8 {
            let _ = decompress(&[a], &model);
            for b in 0..=255u8 {
                let _ = decompress(&[a, b], &model);
            }
        }
    }
}

#[test]
fn decompress_never_panics_on_random_input() {
    let mut rng = Lcg(42);
    for model in models() {
        for _ in 0..100_000 {
            let _ = decompress(&rng.bytes(20), &model);
        }
    }
}

#[test]
fn malformed_input_is_an_error() {
    let model = ShocoModel::default();
    // 0x00 escapes the following byte, so it can't be last
    assert!(matches!(decompress(&[0x00], &model), Err(DecompressError::Malformed)));
    assert!(matches!(decompress(b"ab\x00", &model), Err(DecompressError::Malformed)));
    // A 2-byte and a 4-byte pack, cut short
    assert!(matches!(decompress(&[0xc0], &model), Err(DecompressError::Malformed)));
    assert!(matches!(decompress(&[0xe0, 0x00, 0x00], &model), Err(DecompressError::Malformed)));
    // The default model has 3 packs, so there is no pack for this header
    assert!(matches!(decompress(&[0xf0, 0x00, 0x00, 0x00], &model), Err(DecompressError::Malformed)));
}

#[test]
fn nul_bytes_round_trip() {
    for model in models() {
        for s in ["\0", "\0\0", "a\0b", "hello\0world", "the\0the"] {
            assert_eq!(decompress(&compress(s, &model), &model).unwrap(), s);
        }
    }
}

#[test]
fn round_trip_random_text() {
    let mut rng = Lcg(7);
    for model in models() {
        for _ in 0..20_000 {
            let s = rng.text(200);
            assert_eq!(decompress(&compress(&s, &model), &model).unwrap(), s);
        }
    }
}

#[test]
fn round_trip_training_text() {
    let text = std::fs::read_to_string("training_data/metamorphosis.txt").unwrap();
    for model in models() {
        for line in text.lines() {
            assert_eq!(decompress(&compress(line, &model), &model).unwrap(), line);
        }
    }
}

/// With the default model, the output must be identical to the shoco crate on
/// crates.io, which matches the original C implementation (and escapes NUL
/// bytes the same way)
#[test]
fn default_model_matches_reference() {
    let model = ShocoModel::default();
    let text = std::fs::read_to_string("training_data/metamorphosis.txt").unwrap();
    let mut rng = Lcg(3);
    let inputs = text.lines().map(String::from).chain((0..20_000).map(|_| rng.text(100)));
    for s in inputs {
        let ours = compress(&s, &model);
        assert_eq!(ours, shoco_reference::compress(s.as_bytes()), "{:?}", s);
        assert_eq!(shoco_reference::decompress(&ours).unwrap(), s.as_bytes());
    }
}

/// The default model is the C implementation's; the legacy model is the one
/// this crate used before 0.2, which must still decode data written with it
#[test]
fn default_and_legacy_models() {
    let s = "of time to think undisturbed about how he would have to re-arrange";
    let common = [143, 32, 200, 216, 32, 145, 32, 200, 16, 107, 32, 229, 32, 9, 60, 188, 100,
        32, 97, 222, 129, 32, 151, 119, 32, 148, 32, 217, 2, 100, 32, 149, 118, 101, 32, 145,
        32, 156, 235];
    let c_default = [&common[..], &[17, 195, 5]].concat();
    let legacy = [&common[..], &[145, 195, 5]].concat();
    assert_eq!(compress(s, &ShocoModel::default()), c_default);
    assert_eq!(compress(s, &ShocoModel::legacy_default()), legacy);
    assert_eq!(decompress(&legacy, &ShocoModel::legacy_default()).unwrap(), s);
}
