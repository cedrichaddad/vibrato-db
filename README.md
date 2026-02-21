# Vibrato-DB

A high-performance, disk-backed vector search engine written in Rust.

Vibrato-DB is a specialized vector database designed for low-latency similarity search on dense embeddings. It implements a custom Hierarchical Navigable Small World (HNSW) index from scratch, backed by a zero-copy memory-mapped storage engine.

Unlike wrapper libraries, Vibrato-DB is a standalone database server that handles persistence, concurrency, and vector math optimization down to the instruction level.

---

## Performance Metrics

Benchmarks run on a single-thread consumer CPU (Apple M2) against 128-dimensional normalized vectors.
`Current` latency values below are true p99 numbers from `benches/hnsw_p99.rs` (median of 5 runs).

| Metric | Phase 0 Baseline | Phase 4 | **Current** | Improvement (vs Phase 0) |
|--------|------------------|---------|-------------|--------------------------|
| Query Latency (P99, ef=20) | 147 µs | 65 µs | **16 µs** | **9.2x** |
| Query Latency (P99, ef=50) | 271 µs | 124 µs | **42 µs** | **6.5x** |
| Throughput | 1.8 Gelem/s | 4.4 Gelem/s | **13.9 Gelem/s** | **7.7x** |
| Recall@10 | 99.0% | 99.0% | **99.0%** | No regression |

### Detailed Benchmarks

**SIMD Distance Kernels (128D vectors)**

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Dot Product | 9.18 ns | 13.9 Gelem/s |
| L2 Distance | 9.73 ns | 13.2 Gelem/s |

**HNSW Search Latency (P99, 5000 vectors, k=10)**

| ef Parameter | Phase 0 | Phase 4 | **Current** |
|--------------|---------|---------|-------------|
| ef=20 | 147 µs | 65 µs | **16 µs** |
| ef=50 | 271 µs | 124 µs | **42 µs** |
| ef=100 | 369 µs | 203 µs | **68 µs** |

**P99 Methodology**

```bash
# true p99 harness (not criterion mean)
cargo bench -p vibrato-core --bench hnsw_p99

# robust median p99 across repeated runs
./scripts/bench_hnsw_p99.sh 5
```

- HNSW p99 values above are the median of 5 fixed-query harness runs (`20,000` queries/run).
- SIMD throughput values come from `cargo bench -p vibrato-core --bench simd` at 128D.

---

## Technical Architecture

Vibrato-DB is built on core pillars of systems engineering, evolved into a robust `v2` distributed system:

### 1. Zero-Copy Storage Engine (mmap)

Vectors are not loaded into the heap. The `.vdb` binary format is designed to be memory-mapped directly into the process's virtual address space.

- **Benefit**: Instant startup time regardless of dataset size.
- **Safety**: Uses `bytemuck` for alignment-checked casting from raw bytes to `&[f32]` slices.
- **OS Optimization**: Relies on the kernel's Page Cache to handle hot/cold data swapping.
- **Madvise Tuning**: Implements memory access pattern hints (AB tested in `v2`) to ensure page cache efficiency under random read load.

### 2. Lock-Free HNSW Indexing

Implements a multi-layered graph traversal algorithm for Approximate Nearest Neighbor (ANN) search.

- **Diversity Heuristic**: Neighbor selection logic actively prunes redundant connections.
- **Epoch Visited Pool**: Uses thread-local epoch-array visited sets to eliminate per-query clear cost.

### 3. SIMD-Accelerated Math

Distance kernels are strictly optimized for AVX2 (x86_64) and NEON (aarch64).

- **Aligned Loads**: Exploits 32-byte alignment for AVX2 `vmovaps` when data layout permits.
- **Runtime Dispatch**: Automatically selects the fastest kernel supported by the CPU.

### 4. V2 Control Plane & Storage Tiers

The `v2` architecture introduces an immutable segment lifecycle and distributed-ready semantics.

- **SQLite Catalog (`catalog.rs`)**: Centralizes metadata, segmented collections, API key authentication, and role-based access control (RBAC).
- **WAL & Multi-Version Concurrency (`recovery.rs`)**: Write-Ahead Logs provide durability against crashes by replaying uncommitted ingestion events.
- **Hierarchical Indexing**: Divides storage into a Hot/Active tier (mmap `.vdb`) and an Archive tier (`pq.rs`).
- **Product Quantization (PQ)**: L2 segments are PQ-encoded for highly memory-efficient background archive searching.
- **Roaring Bitmaps (`filter.rs`)**: Accelerates pre-filtering attributes before executing dense vector similarity scans.

### 5. Multi-Modal Ingestion & Python FFI

Vibrato-DB natively processes high-level audio and neural data.

- **Onboard Neural Pipeline (`crates/vibrato-neural`)**: Embeds ONNX runtime to do on-the-fly audio feature extraction (Spectrograms, Windowing) within the DB.
- **Native Python Bindings (`crates/vibrato-ffi`)**: High-performance bindings via PyO3 that bypass HTTP serialization overhead entirely, allowing vibrato clusters to be embedded in python workloads with zero network latency.

---

## Installation and Usage

### 1. Start the Server

Vibrato-DB runs as a standalone HTTP service.

```bash
# Clone the repository
git clone https://github.com/cedrichaddad/vibrato-db.git
cd vibrato-db

# Run the server (auto-builds graph if missing)
cargo run --release -- serve \
  --data ./data/music.vdb \
  --port 8080
```

Optional (for audio ingest with `audio_path`):

```bash
# Download model artifacts once
cargo run --release -- setup-models --model-dir ./models
```

`setup-models` now writes a local `model-manifest.json` with SHA-256 hashes and
`serve` verifies model integrity against that manifest on startup.

### 2. CLI Tools

The CLI includes built-in commands for data ingestion and search.

```bash
# Ingest vectors from JSON to .vdb
# Input format: [[0.1, ...], [0.2, ...]]
cargo run --release -- ingest --input vectors.json --output data.vdb

# Search via CLI (queries running server)
cargo run --release -- search --query "0.1,0.2,..." --k 5
```

### 3. Production v2 Control Plane

`v2` adds SQLite catalog, WAL ingest, immutable segment lifecycle, admin auth, and recovery/orphan handling.
It also includes roaring-style bitmap filter acceleration and PQ-encoded L2 archive segments for archive-tier search.

```bash
# Start production server
cargo run --release -- serve-v2 \
  --data-dir ./vibrato_data \
  --collection default \
  --dim 128 \
  --bootstrap-admin-key true

# Optional: enable Arrow Flight ingest data plane on a second port
cargo run --release -- serve-v2 \
  --data-dir ./vibrato_data \
  --collection default \
  --dim 128 \
  --port 8080 \
  --flight-port 50051

# Create/revoke API keys
cargo run --release -- key-create --data-dir ./vibrato_data --name ops --roles admin,query,ingest
cargo run --release -- key-revoke --data-dir ./vibrato_data --key-id <vbk_id>

# Snapshot / restore / replay
cargo run --release -- snapshot-create --data-dir ./vibrato_data --collection default
cargo run --release -- snapshot-restore --data-dir ./vibrato_data --snapshot-dir ./vibrato_data/snapshots/<snapshot_id>
cargo run --release -- replay-to-lsn --data-dir ./vibrato_data --collection default --target-lsn 1000
```

See `/Users/cedrichaddad/vibrato-db/docs/PRODUCTION_RUNBOOK_V21.md` for full operations guidance.

Arrow Flight ingest notes:
- Flight service is enabled only when `--flight-port` is set.
- Auth uses gRPC metadata header `authorization: Bearer vbk_<id>.<secret>`.
- `do_put` expects a `RecordBatch` with:
  - required: `vector` (`FixedSizeList<Float32>`; fixed length must equal collection `dim`)
  - optional: `metadata_json` (`Utf8`), `idempotency_key` (`Utf8`)
  - optional metadata columns: `source_file` (`Utf8`), `start_time_ms` (`UInt32`), `duration_ms` (`UInt16`), `bpm` (`Float32`), `tags` (`List<Utf8>`)

### 4. Python Bindings

Vibrato-DB exposes a high-performance Python API via PyO3.

```python
import vibrato

# Open an index (releases GIL during search)
index = vibrato.VibratoIndex("data.idx", "data.vdb")

# Search
results = index.search(query_vector, k=10)
print(results) # [(id, score), ...]
```

### 5. HTTP API

Search for similar vectors using a simple REST API.

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.12, 0.05, ...],
    "k": 5
  }'
```

Response:

```json
{
  "results": [
    { "id": 42, "score": 0.987 },
    { "id": 105, "score": 0.942 }
  ],
  "query_time_ms": 0.14
}
```

---

## End-to-End Demo

Verify the full ingestion and search pipeline using the provided Python script:

```bash
# Spins up server, ingests mock audio, and performs search
python3 demo.py
```

> **Note on Inference**: `serve` runs in search-only mode if model files are missing or fail manifest verification. Run `setup-models` first to enable `/ingest` with `audio_path`.


## File Structure

```
src/
  hnsw/           # The core graph algorithm implementation
  store.rs        # The mmap abstraction layer
  simd.rs         # Low-level distance kernel optimizations
  server.rs       # Axum HTTP API server
  main.rs         # CLI entry point

crates/
  vibrato-core    # Core engine (HNSW, SIMD, V2 Format)
  vibrato-server  # V2 control/data plane services
  vibrato-edge    # C-ABI edge embedding surface
  vibrato-midas   # Time-series constrained DTW search primitives
  vibrato-neural  # Audio pipeline (features, ONNX)
  vibrato-ffi     # Python/C bindings
```

---

## Verification

### Running Tests

```bash
cargo test --workspace
```

### Running Benchmarks

```bash
# Measure SIMD throughput
cargo bench -p vibrato-core --bench simd

# Measure Index Latency
cargo bench -p vibrato-core --bench hnsw
```

---

## License

MIT License
