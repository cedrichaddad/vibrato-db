# Vibrato-DB

A high-performance, disk-backed vector search engine written in Rust.

Vibrato-DB is a specialized vector database designed for low-latency similarity search on dense embeddings. It implements a custom Hierarchical Navigable Small World (HNSW) index from scratch, backed by a zero-copy memory-mapped storage engine.

Unlike wrapper libraries, Vibrato-DB is a standalone database server that handles persistence, concurrency, and vector math optimization down to the instruction level.

---

## Performance Metrics

Benchmarks run on a single-thread consumer CPU (Apple M2) against 128-dimensional normalized vectors.

| Metric | Phase 0 Baseline | **Phase 4 (Current)** | Improvement |
|--------|------------------|-----------------------|-------------|
| Query Latency (P99, ef=20) | 147 µs | **65 µs** | **2.26x** |
| Query Latency (P99, ef=50) | 271 µs | **124 µs** | **2.18x** |
| Throughput | 1.8 Gelem/s | **4.4 Gelem/s** | **2.44x** |
| Recall@10 | 99.0% | 99.0% | No regression |

### Detailed Benchmarks

**SIMD Distance Kernels (128D vectors)**

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Dot Product | 29 ns | 4.4 Gelem/s |
| L2 Distance | 62 ns | 2.1 Gelem/s |

**HNSW Search Latency (5000 vectors, k=10)**

| ef Parameter | Phase 0 | **Phase 4** |
|--------------|---------|-------------|
| ef=20 | 147 µs | **65 µs** |
| ef=50 | 271 µs | **124 µs** |
| ef=100 | 369 µs | **203 µs** |

---

## Technical Architecture

Vibrato-DB is built on three core pillars of systems engineering:

### 1. Zero-Copy Storage Engine (mmap)

Vectors are not loaded into the heap. The `.vdb` binary format is designed to be memory-mapped directly into the process's virtual address space.

- **Benefit**: Instant startup time regardless of dataset size.
- **Safety**: Uses `bytemuck` for alignment-checked casting from raw bytes to `&[f32]` slices.
- **OS Optimization**: Relies on the kernel's Page Cache to handle hot/cold data swapping.

### 2. Lock-Free HNSW Indexing

Implements a multi-layered graph traversal algorithm for Approximate Nearest Neighbor (ANN) search.

- **Diversity Heuristic**: Neighbor selection logic actively prunes redundant connections.
- **BitSet Visited Pool**: Uses thread-local `FixedBitSet` pools to eliminate hashing overhead.
- **Persistence**: The graph structure is serialized to a compact `.idx` binary format.

### 3. SIMD-Accelerated Math

Distance kernels are strictly optimized for AVX2 (x86_64) and NEON (aarch64).

- **Aligned Loads**: Exploits 32-byte alignment for AVX2 `vmovaps` when data layout permits.
- **Runtime Dispatch**: Automatically selects the fastest kernel supported by the CPU.

---

## Installation and Usage

### 1. Start the Server

Vibrato-DB runs as a standalone HTTP service.

```bash
# Clone the repository
git clone https://github.com/cedrichaddad/vibrato-db.git
cd vibrato-db

# Run the server (auto-builds index if missing)
cargo run --release -- serve \
  --data ./data/music.vdb \
  --index ./data/music.idx \
  --port 8080
```

### 2. CLI Tools

The CLI includes built-in commands for data ingestion and search.

```bash
# Ingest vectors from JSON to .vdb
# Input format: [[0.1, ...], [0.2, ...]]
cargo run --release -- ingest --input vectors.json --output data.vdb

# Search via CLI (queries running server)
cargo run --release -- search --query "0.1,0.2,..." --k 5
```

### 3. Python Bindings

Vibrato-DB exposes a high-performance Python API via PyO3.

```python
import vibrato

# Open an index (releases GIL during search)
index = vibrato.VibratoIndex("data.idx", "data.vdb")

# Search
results = index.search(query_vector, k=10)
print(results) # [(id, score), ...]
```

### 4. HTTP API

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

> **Note on Inference**: The neural pipeline strictly mocks the inference step due to unstable `ort` 2.x dependencies on current compilers. The system architecture is fully wired (ingest handler -> resize/decode logic -> inference engine), but the embedding vector is currently a deterministic pseudo-random signal.


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
cargo bench --bench simd

# Measure Index Latency
cargo bench --bench hnsw
```

---

## License

MIT License
