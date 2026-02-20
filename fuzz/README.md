# Vibrato DB Fuzz Targets

These targets are built for `cargo-fuzz` and focus on malformed segment bytes:

- `read_v2_header`: exercises `read_v2_header` via `fuzz_read_v2_header_bytes`.
- `load_archive_segment_handle`: exercises archive PQ decoding via `fuzz_parse_archive_segment_bytes`.

Run:

```bash
cargo install cargo-fuzz
cd fuzz
cargo fuzz run read_v2_header
cargo fuzz run load_archive_segment_handle
```
