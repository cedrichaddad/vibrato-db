#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = vibrato_db::prod::engine::fuzz_read_v2_header_bytes(data);
});
