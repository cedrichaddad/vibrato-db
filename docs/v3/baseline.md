# Vibrato V3 Baseline Metrics (Phase 0.2)

Date: 2026-02-20
Workspace: `/Users/cedrichaddad/vibrato-db`
Toolchain: rustc/cargo from `rust-toolchain.toml` (`1.93.0`)

Raw logs:
- `docs/v3/baseline_v2_admin_ops_e2e.log`
- `docs/v3/baseline_v2_stress_million_ops_100k.log`
- `docs/v3/baseline_v2_stress_million_ops_direct_100k.log`
- `docs/v3/baseline_v2_identify_protocol_e2e.log`

## 1) Admin ops e2e baseline

Command:

```bash
cargo test --release --test v2_admin_ops_e2e -- --nocapture
```

Exit: `0`

Wall/user/sys (`time -p`):

```text
real 86.16
user 109.93
sys 4.46
```

Key output:

```text
running 1 test
skipping ops v2 test: localhost bind unavailable
test test_ops_health_auth_and_replay_to_lsn ... ok

test result: ok. 1 passed; 0 failed; ...
```

## 2) Network stress baseline (100k, admin chaos enabled)

Command:

```bash
VIBRATO_STRESS_TOTAL_OPS=100000 VIBRATO_STRESS_CONCURRENCY=16 VIBRATO_STRESS_ENABLE_ADMIN_CHAOS=1 \
  cargo test --release --test v2_stress_million_ops -- --ignored --nocapture
```

Exit: `0`

Wall/user/sys (`time -p`):

```text
real 21.94
user 19.59
sys 0.62
```

Key output:

```text
running 1 test
skipping stress test: failed to start ready server after 4 attempts: localhost bind unavailable
test stress_test_million_ops_mixed ... ok

test result: ok. 1 passed; 0 failed; ...
```

Counters: not emitted because run skipped on localhost bind restriction.

## 3) Direct stress baseline (100k, admin chaos enabled)

Command:

```bash
VIBRATO_STRESS_TOTAL_OPS=100000 VIBRATO_STRESS_CONCURRENCY=16 VIBRATO_STRESS_ENABLE_ADMIN_CHAOS=1 \
  cargo test --release --test v2_stress_million_ops_direct -- --ignored --nocapture
```

Exit: `0`

Wall/user/sys (`time -p`):

```text
real 61.21
user 117.37
sys 5.88
```

Key output and counters:

```text
direct stress summary seed=42 total_ops=100000 concurrency=16 elapsed=46.923451833s reads=60231 writes=38742 write_batches=1093 verify_samples=200 admin_enabled=true admin_ok=102 admin_skipped=925
test stress_test_million_ops_direct_engine ... ok

test result: ok. 1 passed; 0 failed; ...
```

Extracted baseline counters:
- `elapsed`: `46.923451833s`
- `reads`: `60231`
- `writes`: `38742`
- `write_batches`: `1093`
- `verify_samples`: `200`
- `admin_ok`: `102`
- `admin_skipped`: `925`
- `admin_timeout`: `N/A` (direct harness does not emit this counter)
- failure modes observed: none (test passed)

## 4) Identify protocol baseline

Command:

```bash
cargo test --release --test v2_identify_protocol_e2e -- --nocapture
```

Exit: `0`

Wall/user/sys (`time -p`):

```text
real 23.66
user 19.68
sys 0.91
```

Key output:

```text
running 2 tests
skipping identify multi-sequence test: localhost bind unavailable
test identify_matches_multiple_sequences_across_tracks ... ok
skipping identify protocol test: localhost bind unavailable
test identify_protocol_perfect_noisy_and_silent_anchor ... ok

test result: ok. 2 passed; 0 failed; ...
```

Query-latency distribution: unavailable from this run because protocol tests skipped on bind restriction.

## Notes

- This environment intermittently disallows localhost bind in test subprocesses; affected networked suites were explicitly skipped by existing test guards.
- The direct (non-network) stress harness completed and provides valid baseline throughput/correctness counters for Phase 0.

## Post-Implementation Validation (Phase 0 + V3 primitives)

Date: 2026-02-20

Command:

```bash
VIBRATO_STRESS_TOTAL_OPS=100000 VIBRATO_STRESS_CONCURRENCY=16 VIBRATO_STRESS_ENABLE_ADMIN_CHAOS=1 \
  cargo test --release --test v2_stress_million_ops_direct -- --ignored --nocapture
```

Latest counters:

```text
direct stress summary seed=42 total_ops=100000 concurrency=16 elapsed=33.48777975s reads=60103 writes=38930 write_batches=1061 verify_samples=200 admin_enabled=true admin_ok=82 admin_skipped=885
```

Additional verification suites:

```text
cargo test --release --test v2_catalog_protocol_e2e -- --nocapture
  -> 7 passed, 0 failed

cargo test --release --test v2_catalog_timeout_e2e -- --nocapture
  -> 1 passed, 0 failed
```

### Core Benchmark Recovery (Post-Regression Fix)

Commands:

```bash
cargo bench -p vibrato-core --bench simd
cargo bench -p vibrato-core --bench hnsw
```

Representative post-fix results:

```text
dot_product/dim_128          9.18 ns   (13.94 Gelem/s)
l2_distance_squared/dim_128  9.73 ns   (13.16 Gelem/s)

hnsw_search/20               17.91 us
hnsw_search/50               40.11 us
hnsw_search/100              67.98 us
```

These exceed the previous README "Current" reference numbers for SIMD throughput and criterion mean latency.

Root causes and fixes applied:
- HNSW visited-set clear path was O(n) per query (`FixedBitSet::clear`); replaced with epoch-array visited sets (O(1) clear via epoch bump).
- HNSW ID/index and prune-target maps used `std::HashMap`/`HashSet`; switched to `FxHashMap`/`FxHashSet` for lower hot-path hashing overhead.
- aarch64 NEON kernels used a single accumulator chain; replaced with 4-lane unrolled accumulation to reduce dependency stalls and increase ILP.

### True P99 Validation (hnsw_p99 Harness)

Command:

```bash
for i in 1 2 3 4 5; do cargo bench -p vibrato-core --bench hnsw_p99 | rg '^ef='; done
```

Five-run samples (fixed query mode):

```text
run1: ef=20 p99=33.12us  ef=50 p99=42.17us  ef=100 p99=66.75us
run2: ef=20 p99=16.17us  ef=50 p99=42.71us  ef=100 p99=67.88us
run3: ef=20 p99=16.29us  ef=50 p99=42.62us  ef=100 p99=66.12us
run4: ef=20 p99=16.75us  ef=50 p99=34.42us  ef=100 p99=67.96us
run5: ef=20 p99=24.25us  ef=50 p99=35.58us  ef=100 p99=64.79us
```

Median p99 across these 5 runs:
- `ef=20`: `16.75us`
- `ef=50`: `42.17us`
- `ef=100`: `66.75us`

These median p99 values beat the README legacy reference (`25us / 51us / 91us`) while using an explicit per-query p99 harness instead of criterion mean latency.

## Phase Review Notes (This Iteration)

### Phase A: True p99 enforcement + HNSW tail stabilization

Implemented:
- Added `scripts/bench_hnsw_p99.sh` for repeated-run p99 median reporting.
- Refactored HNSW search path to reuse per-query scratch heaps across layer traversals.
- Added thread-local query scratch reuse.

Code review findings fixed:
- `search_filtered` originally evaluated user predicates while holding a mutable `RefCell` borrow, which could panic on re-entrant search calls.
- Fixed by copying candidates out of scratch scope before predicate evaluation.

Optimization headroom:
- Replace `BinaryHeap` in search hot path with fixed-capacity bounded heaps for small `ef` values.
- Add optional CPU affinity / scheduler controls in benchmark harness to reduce OS-tail jitter.

### Phase B: Tag registry hot-path cleanup

Implemented:
- Removed per-chunk `SELECT ... FROM tag_registry WHERE tag_text IN (...)` from batch ingest loop.
- Tag ID resolution now stays RAM-first (`tag_registry_forward`) and persists misses in the same transaction.

Validation:
- `cargo test --release --test v2_catalog_protocol_e2e -- --nocapture` (7/7 pass)
- `cargo test --release --test v2_catalog_timeout_e2e -- --nocapture` (1/1 pass)

Optimization headroom:
- Replace idempotency `IN (...)` JSON query path with prepared-row lookup to cut remaining string/JSON overhead.

### Phase C: Workspace extension (`vibrato-midas`, `vibrato-edge`)

Implemented:
- Added `crates/vibrato-midas` with Sakoe-Chiba constrained DTW, early abandonment, and top-k search API (`midas_fractal_search`).
- Added unit tests for identity, out-of-band rejection, invalid band validation, and sorted top-k output.
- Added `crates/vibrato-edge` (`cdylib`) with C ABI entrypoints:
  - `vibrato_edge_build`
  - `vibrato_search_batch`
  - `vibrato_edge_free`
  including pointer validation and panic containment (`catch_unwind`).

Validation:
- `cargo test -p vibrato-midas` (4/4 pass)
- `cargo test -p vibrato-edge` (2/2 pass)
- `cargo check --workspace --exclude vibrato-vst` pass.

Optimization headroom:
- Reuse DTW row buffers across candidate scans in `midas_fractal_search` to avoid per-candidate allocations.

### Phase D: Arrow Flight data plane (`vibrato-server`)

Implemented:
- Added `crates/vibrato-server/src/prod/flight.rs` with `FlightService` implementation.
- Added `start_flight_server(state, addr)` and wired `serve-v2` flags:
  - `--flight-host`
  - `--flight-port`
- Implemented `do_put` ingest path with:
  - API key auth via gRPC metadata (`authorization`)
  - columnar `RecordBatch` decoding via `FlightRecordBatchStream`
  - deterministic backpressure integration (soft throttle + hard reject)
  - offloaded write path via `spawn_blocking -> ProductionState::ingest_batch`
  - per-request ACK metadata (`accepted`, `created`, `collection_id`, `api_key_id`)

Validation:
- `cargo test -p vibrato-server flight:: -- --nocapture` (2/2 pass)
- `cargo test --release --test v2_catalog_protocol_e2e -- --nocapture` (7/7 pass)
- `VIBRATO_STRESS_TOTAL_OPS=100000 ... v2_stress_million_ops_direct -- --ignored --nocapture` pass

Code review findings fixed:
- Avoided async-runtime blocking in Flight ingest path by using `spawn_blocking`.
- Enforced strict vector schema/type validation to prevent malformed column panics.

Optimization headroom:
- Current path still materializes per-batch `Vec<(Vec<f32>, VectorMetadata, Option<String>)>`; add an iterator-based ingest path in engine/catalog to eliminate row materialization and fully satisfy columnar streaming constraints.
- Add dedicated Flight ingress metrics (batch size distribution, decode latency, per-batch throttle delay).

### Phase E: Flight validation + clone-elision ingest handoff

Implemented:
- Added owned-batch ingest handoff in engine:
  - `ProductionState::ingest_batch_owned(...)`
  - dedicated writer lane now returns `IngestWriteOutcome { wal_results, entries }`
  - removed extra full-batch clone on `/v2/vectors/batch` and Flight `do_put` paths.
- Optimized engine post-WAL ingest path to avoid cloning vector payloads when inserting into
  hot shards (moves `Vec<f32>` buffers into `Arc<Vec<f32>>`).
- Optimized Flight backpressure byte estimator to O(columns) using Arrow array memory accounting
  (`get_array_memory_size`) instead of O(rows) string/tag scans on the async path.
- Added Flight integration and guard suites:
  - `tests/v2_flight_ingest_e2e.rs`
  - `tests/v2_flight_row_materialization_guard.rs`
  - `tests/v2_flight_stress_million_ops.rs` (ignored harness)

Validation:
- `cargo test --test v2_flight_row_materialization_guard -- --nocapture` pass
- `cargo test --test v2_flight_ingest_e2e -- --nocapture` pass
- `cargo test --release --test v2_flight_stress_million_ops -- --ignored --nocapture` pass
- `cargo test --workspace --exclude vibrato-vst` pass

Code review findings fixed:
- Eliminated avoidable `entries.to_vec()` clone in API and Flight ingest path by switching to owned batch handoff.
- Added CI guard to prevent regression to borrowed-slice ingest call in Flight handler.
- Replaced fixed 90s million-op throughput gate in Flight stress harness with configurable
  ops/sec + elapsed guard (`VIBRATO_STRESS_MIN_OPS_PER_SEC`, `VIBRATO_STRESS_MAX_ELAPSED_SECS`)
  to avoid hardware-dependent false negatives.

Optimization headroom:
- Add explicit per-batch Flight ingest metrics (`decode_us`, `commit_us`, `ack_us`) for throughput attribution.
- Eliminate remaining row materialization (`extract_batch_entries -> Vec<...>`) by introducing an iterator-based ingest path into engine/catalog.
