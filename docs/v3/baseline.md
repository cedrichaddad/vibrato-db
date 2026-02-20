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
cargo bench --bench simd
cargo bench --bench hnsw
```

Representative post-fix results:

```text
dot_product/dim_128          9.35 ns   (13.69 Gelem/s)
l2_distance_squared/dim_128  9.68 ns   (13.23 Gelem/s)

hnsw_search/20               22.28 us
hnsw_search/50               42.85 us
hnsw_search/100              70.34 us
```

These exceed README "Current" reference numbers for both SIMD throughput and HNSW latency.

Root causes and fixes applied:
- HNSW visited-set clear path was O(n) per query (`FixedBitSet::clear`); replaced with epoch-array visited sets (O(1) clear via epoch bump).
- HNSW ID/index and prune-target maps used `std::HashMap`/`HashSet`; switched to `FxHashMap`/`FxHashSet` for lower hot-path hashing overhead.
- aarch64 NEON kernels used a single accumulator chain; replaced with 4-lane unrolled accumulation to reduce dependency stalls and increase ILP.
