# Vibrato-DB

A high-performance, disk-backed vector search engine written in Rust.

Vibrato-DB is a specialized vector database designed for low-latency similarity search on dense embeddings. It implements a custom Hierarchical Navigable Small World (HNSW) index from scratch, backed by a zero-copy memory-mapped storage engine.

Unlike wrapper libraries, Vibrato-DB is a standalone database server that handles persistence, concurrency, and vector math optimization down to the instruction level.

---

## Performance Metrics

Benchmarks run on a single-thread consumer CPU (Apple M2) against 128-dimensional normalized vectors.

| Metric | Result | Context |
|--------|--------|---------|
| Query Latency (P99) | 147 µs | ef=20, k=10, 5000 vectors |
| Throughput | 4.4 Gelem/s | SIMD-accelerated Dot Product |
| Recall@10 | 99.0% | vs. Brute Force Ground Truth |
| Memory Overhead | Zero-Copy | Vectors read directly from OS Page Cache |

### Detailed Benchmarks

**SIMD Distance Kernels (128D vectors)**

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Dot Product | 29 ns | 4.4 Gelem/s |
| L2 Distance | 62 ns | 2.1 Gelem/s |

**HNSW Index Build (M=16, ef=100)**

| Dataset Size | Build Time |
|--------------|------------|
| 100 vectors | 50 ms |
| 500 vectors | 588 ms |
| 1000 vectors | 1.7 s |

**HNSW Search Latency (5000 vectors, k=10)**

| ef Parameter | Latency |
|--------------|---------|
| ef=20 | 147 µs |
| ef=50 | 271 µs |
| ef=100 | 369 µs |

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

- **Diversity Heuristic**: Neighbor selection logic actively prunes redundant connections to prevent "island" formation in the graph.
- **BitSet Visited Pool**: Uses thread-local `FixedBitSet` pools instead of HashSets to track visited nodes, eliminating hashing overhead in the hot path.
- **Persistence**: The graph structure is serialized to a compact `.idx` binary format using `bincode`.

### 3. SIMD-Accelerated Math

Distance kernels are written using iterator patterns that strictly compile down to AVX2 / NEON vector instructions.

- **Optimization**: L2-normalized vectors allow replacing expensive Euclidean Distance calculations with fast Dot Products.
- **Throughput**: Achieves ~4.4 Billion element-operations per second on a single core.

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

### 2. Ingest Data (Python Pipeline)

Vibrato-DB includes a Python toolkit to convert audio files into the `.vdb` binary format.

```bash
cd python
pip install -r requirements.txt

# Ingest a folder of MP3s (Extracts VGGish embeddings)
python ingest.py --input ./my_music_library/ --output ../data/music.vdb

# Or generate synthetic data for testing
python test_writer.py --output ../data/test.vdb --count 10000 --dim 128
```

### 3. Query the API

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

## File Structure

```
src/
  hnsw/           # The core graph algorithm implementation
    index.rs      # Greedy search and layer management
    visited.rs    # Thread-local memory pooling
  store.rs        # The mmap abstraction layer
  simd.rs         # Low-level distance kernel optimizations
  server.rs       # Axum HTTP API server

python/           # Data ingestion and VGGish model wrappers
  ingest.py       # Audio-to-embedding pipeline
  vdb_writer.py   # Binary format writer
```

---

## Verification

### Running Tests

The suite includes correctness tests for the HNSW graph and math kernels.

```bash
cargo test
```

### Running Benchmarks

To reproduce the performance metrics:

```bash
# Measure SIMD throughput
cargo bench --bench simd

# Measure Index Latency
cargo bench --bench hnsw
```

---

## License

MIT License
