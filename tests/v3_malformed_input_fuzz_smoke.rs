#![cfg(feature = "fuzzing")]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn malformed_bytes_do_not_panic_decode_paths() {
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..2_000 {
        let len = rng.gen_range(0..4096usize);
        let mut data = vec![0u8; len];
        rng.fill(data.as_mut_slice());

        let _ = vibrato_db::prod::engine::fuzz_read_v2_header_bytes(&data);
        let _ = vibrato_db::prod::engine::fuzz_parse_archive_segment_bytes(&data);
    }
}
