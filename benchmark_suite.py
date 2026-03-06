#!/usr/bin/env python3
"""
Vibrato vs TDengine — Fair Benchmark Suite

Compares both ingestion throughput AND query latency.

Usage:
    # Vibrato-only (TDengine not installed)
    python3 benchmark_suite.py --devices 10 --rows-per-device 100000

    # Full comparison (requires taosBenchmark + taos CLI)
    python3 benchmark_suite.py --devices 100 --rows-per-device 1000000

    # Quick smoke test
    python3 benchmark_suite.py --devices 2 --rows-per-device 1000 --num-queries 10 --skip-tdengine
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    name: str
    total_rows: int
    duration_s: float
    throughput: float  # rows/sec

@dataclass
class QueryLatencyResult:
    name: str
    num_queries: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

# ---------------------------------------------------------------------------
# Vibrato benchmark
# ---------------------------------------------------------------------------

def run_vibrato_ingest(host, http_port, flight_port, api_key, num_devices, rows_per_device, batch_size, dim):
    from vibrato.client import VibratoClient

    total = num_devices * rows_per_device
    print(f"\n{'='*60}")
    print(f"  VIBRATO INGEST BENCHMARK")
    print(f"  {num_devices} devices × {rows_per_device:,} rows = {total:,} vectors")
    print(f"  Batch size: {batch_size:,}  |  Dimensions: {dim}")
    print(f"{'='*60}")

    client = VibratoClient(
        http_url=f"http://{host}:{http_port}",
        api_key=api_key,
        flight_url=f"grpc://{host}:{flight_port}",
        flight_chunk_rows=batch_size,  # match batch → one chunk per batch
    )

    total_batches = total // batch_size

    def batch_generator():
        vectors_sent = 0
        for i in range(total_batches):
            batch_vectors = np.random.rand(batch_size, dim).astype(np.float32)
            batch_entities = np.random.randint(0, num_devices, size=batch_size, dtype=np.uint64)
            batch_ts = np.arange(vectors_sent, vectors_sent + batch_size, dtype=np.uint64)
            yield {
                "data": batch_vectors,
                "entity_ids": batch_entities,
                "sequence_ts": batch_ts,
            }
            vectors_sent += batch_size
            if i % max(1, total_batches // 10) == 0:
                print(f"  Progress: {vectors_sent:,} / {total:,} vectors...")

    start = time.perf_counter()
    result = client.ingest_stream(batch_generator(), prefer_flight=True)
    duration = time.perf_counter() - start
    throughput = total / duration

    print(f"  ✅ Done: {result.accepted:,} accepted, {result.created:,} created")
    print(f"  ⏱  Time: {duration:.2f}s  |  Throughput: {throughput:,.0f} rec/s")

    return IngestResult(
        name="Vibrato",
        total_rows=total,
        duration_s=duration,
        throughput=throughput,
    )


def run_vibrato_queries(host, http_port, api_key, dim, num_queries, k=10, ef=50):
    print(f"\n{'='*60}")
    print(f"  VIBRATO QUERY BENCHMARK")
    print(f"  {num_queries} random k-NN queries  |  k={k}  |  ef={ef}")
    print(f"{'='*60}")

    session = requests.Session()
    url = f"http://{host}:{http_port}/v3/query"
    headers = {"authorization": f"Bearer {api_key}"}

    # Generate random query vectors
    rng = np.random.default_rng(42)
    queries = rng.random((num_queries, dim), dtype=np.float32)

    # Warmup (5 queries)
    warmup_count = min(5, num_queries)
    for i in range(warmup_count):
        body = {"vector": queries[i].tolist(), "k": k, "ef": ef, "include_metadata": False}
        session.post(url, json=body, headers=headers, timeout=30)
    print(f"  Warmup: {warmup_count} queries")

    # Timed queries
    latencies = []
    for i in range(num_queries):
        body = {"vector": queries[i].tolist(), "k": k, "ef": ef, "include_metadata": False}
        t0 = time.perf_counter()
        resp = session.post(url, json=body, headers=headers, timeout=30)
        t1 = time.perf_counter()
        if resp.status_code != 200:
            print(f"  ⚠️  Query {i} failed: {resp.status_code}")
            continue
        latencies.append((t1 - t0) * 1000)  # ms

    latencies.sort()
    n = len(latencies)
    if n == 0:
        print("  ❌ All queries failed")
        return None

    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    mean = sum(latencies) / n

    print(f"  ✅ {n} queries completed")
    print(f"  p50: {p50:.2f}ms  |  p95: {p95:.2f}ms  |  p99: {p99:.2f}ms  |  mean: {mean:.2f}ms")

    return QueryLatencyResult(
        name="Vibrato", num_queries=n,
        p50_ms=p50, p95_ms=p95, p99_ms=p99, mean_ms=mean,
    )

# ---------------------------------------------------------------------------
# TDengine benchmark
# ---------------------------------------------------------------------------

def get_tdengine_cmd_prefix():
    if shutil.which("taosBenchmark") and shutil.which("taos"):
        return []
    if shutil.which("docker"):
        # Check if tdengine container is running
        res = subprocess.run(["docker", "ps", "--filter", "name=tdengine", "--format", "{{.Names}}"], capture_output=True, text=True)
        if "tdengine" in res.stdout:
            return ["docker", "exec", "-i", "tdengine"]
    return None

def check_tdengine_available():
    return get_tdengine_cmd_prefix() is not None


def run_tdengine_ingest(num_devices, rows_per_device):
    total = num_devices * rows_per_device
    print(f"\n{'='*60}")
    print(f"  TDENGINE INGEST BENCHMARK")
    print(f"  {num_devices} devices × {rows_per_device:,} rows = {total:,} records")
    print(f"{'='*60}")

    prefix = get_tdengine_cmd_prefix()
    cmd = prefix + ["taosBenchmark", "-t", str(num_devices), "-n", str(rows_per_device), "-y"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  ❌ taosBenchmark timed out after 600s")
        return None

    output = result.stdout + result.stderr
    # Parse: "Spent X.Y (real Z.W) seconds to insert rows: N ... M records/second"
    throughput = None
    duration = None
    for line in output.splitlines():
        if "records/second" in line and "Spent" in line:
            # Extract real throughput
            import re
            m = re.search(r'real\s+([\d.]+)\)\s+seconds', line)
            if m:
                duration = float(m.group(1))
            m2 = re.search(r'real\s+([\d.]+)\)\s+records/second', line)
            if m2:
                throughput = float(m2.group(1))
            else:
                # Fallback: parse the first records/second number
                m3 = re.search(r'([\d.]+)\s+records/second', line)
                if m3:
                    throughput = float(m3.group(1))

    if throughput is None:
        print("  ⚠️  Could not parse taosBenchmark output:")
        for line in output.splitlines()[-10:]:
            print(f"    {line}")
        return None

    if duration is None:
        duration = total / throughput

    print(f"  ✅ Done")
    print(f"  ⏱  Time: {duration:.2f}s  |  Throughput: {throughput:,.0f} rec/s")

    return IngestResult(
        name="TDengine",
        total_rows=total,
        duration_s=duration,
        throughput=throughput,
    )


def run_tdengine_queries(num_queries):
    """Run equivalent SELECT queries against TDengine to measure query latency."""
    print(f"\n{'='*60}")
    print(f"  TDENGINE QUERY BENCHMARK")
    print(f"  {num_queries} random SELECT queries")
    print(f"{'='*60}")

    # TDengine doesn't have k-NN search, so we test the closest equivalent:
    # a point lookup by device + time range (what you'd actually query in IoT).
    prefix = get_tdengine_cmd_prefix()
    latencies = []
    
    import re
    
    for i in range(num_queries):
        device_id = np.random.randint(0, 100)
        table_name = f"d{device_id}"
        sql = f"SELECT * FROM test.meters WHERE tbname = '{table_name}' LIMIT 10;"
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                prefix + ["taos", "-s", sql],
                capture_output=True, text=True, timeout=10,
            )
            t1 = time.perf_counter()
            if result.returncode != 0:
                continue
            
            # Extract true query time from TDengine's internal measurement: 
            # e.g., "Query OK, 10 row(s) in set (0.003120s)"
            match = re.search(r'Query OK, \d+ row\(s\) in set \(([\d.]+)\w*\)', result.stdout)
            if match:
                latencies.append(float(match.group(1)) * 1000)
            else:
                # Fallback to process time if parsing fails
                latencies.append((t1 - t0) * 1000)
                
        except subprocess.TimeoutExpired:
            continue

    if not latencies:
        print("  ❌ All queries failed")
        return None

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    mean = sum(latencies) / n

    print(f"  ✅ {n} queries completed")
    print(f"  p50: {p50:.2f}ms  |  p95: {p95:.2f}ms  |  p99: {p99:.2f}ms  |  mean: {mean:.2f}ms")
    print(f"  ⚠️  Note: TDengine queries are row lookups, not k-NN vector search")

    return QueryLatencyResult(
        name="TDengine (row lookup)", num_queries=n,
        p50_ms=p50, p95_ms=p95, p99_ms=p99, mean_ms=mean,
    )

# ---------------------------------------------------------------------------
# Embedded C ABI benchmark
# ---------------------------------------------------------------------------

import ctypes
import sys

def load_vibrato_c_abi():
    # Detect os to find the right lib extension
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib_path = f"target/release/libvibrato_ffi.{ext}"
    if not os.path.exists(lib_path):
        return None
    
    lib = ctypes.cdll.LoadLibrary(os.path.abspath(lib_path))
    
    # vibrato_open
    lib.vibrato_open.argtypes = [ctypes.c_char_p]
    lib.vibrato_open.restype = ctypes.c_void_p
    
    # vibrato_close
    lib.vibrato_close.argtypes = [ctypes.c_void_p]
    lib.vibrato_close.restype = None
    
    # vibrato_search
    # handle, query_ptr, dim, k, ef, out_ids_ptr, out_scores_ptr
    lib.vibrato_search.argtypes = [
        ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_size_t, 
        ctypes.c_size_t, 
        ctypes.c_size_t, 
        ctypes.POINTER(ctypes.c_size_t), 
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.vibrato_search.restype = ctypes.c_int
    return lib

def run_c_abi_ingest(num_devices, rows_per_device, batch_size, dim):
    lib = load_vibrato_c_abi()
    if not lib:
        print("  ❌ Could not find target/release/libvibrato_ffi.dylib/.so. Did you run `cargo build -p vibrato-ffi --release`?")
        return None, None

    # We need vibrato python lib to use VDBWriter
    sys.path.insert(0, os.path.abspath("python"))
    from vibrato.writer import VDBWriter

    total = num_devices * rows_per_device
    print(f"\n{'='*60}")
    print(f"  EMBEDDED C ABI INGEST BENCHMARK")
    print(f"  {num_devices} devices × {rows_per_device:,} rows = {total:,} vectors")
    print(f"  Writing .vdb locally, then building HNSW via vibrato_open")
    print(f"{'='*60}")

    vdb_path = "/tmp/c_abi_bench.vdb"
    if os.path.exists(vdb_path):
        os.remove(vdb_path)

    t0 = time.perf_counter()
    
    # 1. Write the vectors using VDBWriter
    def batch_generator():
        for d in range(num_devices):
            client_id = f"dev_{d}"
            remaining = rows_per_device
            while remaining > 0:
                chunk = min(remaining, batch_size)
                # Ensure float32!
                data = np.random.rand(chunk, dim).astype(np.float32)
                yield data
                remaining -= chunk

    with VDBWriter(vdb_path, dim) as writer:
        for data in batch_generator():
            writer.write_batch(data)
            
    # 2. Open the index (which triggers rebuild_hnsw dynamically)
    print(f"  File written. Now calling vibrato_open to build the HNSW graph...")
    handle = lib.vibrato_open(vdb_path.encode('utf-8'))
    if not handle:
        print("  ❌ vibrato_open returned NULL handle.")
        return None, None

    t1 = time.perf_counter()
    duration = t1 - t0
    throughput = total / duration

    print(f"  ✅ Done: {total:,} vectors ingested + indexed")
    print(f"  ⏱  Time: {duration:.2f}s  |  Throughput: {throughput:,.0f} rec/s")

    return IngestResult(
        name="Embedded C ABI", total_rows=total, duration_s=duration, throughput=throughput
    ), handle

def run_c_abi_queries(handle, dim, num_queries, k, ef):
    lib = load_vibrato_c_abi()
    if not lib or not handle:
        return None
        
    print(f"\n{'='*60}")
    print(f"  EMBEDDED C ABI QUERY BENCHMARK")
    print(f"  {num_queries} random k-NN queries  |  k={k}  |  ef={ef}")
    print(f"{'='*60}")

    # Pre-allocate ctypes output buffers
    out_ids = (ctypes.c_size_t * k)()
    out_scores = (ctypes.c_float * k)()

    # Warmup
    print(f"  Warmup: 5 queries")
    for _ in range(5):
        query_vec = np.random.rand(dim).astype(np.float32)
        q_ptr = query_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.vibrato_search(handle, q_ptr, dim, k, ef, out_ids, out_scores)

    latencies = []
    for _ in range(num_queries):
        query_vec = np.random.rand(dim).astype(np.float32)
        q_ptr = query_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        t0 = time.perf_counter()
        rc = lib.vibrato_search(handle, q_ptr, dim, k, ef, out_ids, out_scores)
        t1 = time.perf_counter()
        
        if rc == 0:
            latencies.append((t1 - t0) * 1000)

    # Clean up handle after queries are done
    lib.vibrato_close(handle)

    if not latencies:
        print("  ❌ All C ABI queries failed")
        return None

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    mean = sum(latencies) / n

    print(f"  ✅ {n} queries completed")
    print(f"  p50: {p50:.2f}ms  |  p95: {p95:.2f}ms  |  p99: {p99:.2f}ms  |  mean: {mean:.2f}ms")

    return QueryLatencyResult(
        name="Embedded C ABI", num_queries=n,
        p50_ms=p50, p95_ms=p95, p99_ms=p99, mean_ms=mean,
    )

# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_comparison(ingest_results, query_results):
    print(f"\n{'='*70}")
    print(f"  📊  BENCHMARK RESULTS")
    print(f"{'='*70}")

    # Ingest comparison
    print(f"\n  ┌─────────────────────────── INGEST THROUGHPUT ──────────────────────┐")
    print(f"  │ {'Engine':<20} {'Rows':>12} {'Time (s)':>10} {'Throughput':>18} │")
    print(f"  ├──────────────────────────────────────────────────────────────────────┤")
    for r in ingest_results:
        print(f"  │ {r.name:<20} {r.total_rows:>12,} {r.duration_s:>10.2f} {r.throughput:>14,.0f} r/s │")
    print(f"  └──────────────────────────────────────────────────────────────────────┘")

    if len(ingest_results) == 2:
        a, b = ingest_results[0], ingest_results[1]
        ratio = a.throughput / b.throughput if b.throughput > 0 else float("inf")
        faster = a.name if ratio > 1 else b.name
        factor = ratio if ratio > 1 else (1 / ratio if ratio > 0 else float("inf"))
        print(f"\n  → {faster} is {factor:.1f}× faster at ingestion")

    # Query comparison
    if query_results:
        print(f"\n  ┌───────────────────────────── QUERY LATENCY ──────────────────────────┐")
        print(f"  │ {'Engine':<25} {'Queries':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'Mean':>8} │")
        print(f"  ├────────────────────────────────────────────────────────────────────────┤")
        for r in query_results:
            print(f"  │ {r.name:<25} {r.num_queries:>8} {r.p50_ms:>6.2f}ms {r.p95_ms:>6.2f}ms {r.p99_ms:>6.2f}ms {r.mean_ms:>6.2f}ms │")
        print(f"  └────────────────────────────────────────────────────────────────────────┘")

        if len(query_results) == 2:
            a, b = query_results[0], query_results[1]
            print(f"\n  ⚠️  Note: Vibrato does k-NN vector search; TDengine does row lookups.")
            print(f"      These are fundamentally different operations — compare with context.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vibrato vs TDengine Benchmark Suite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--flight-port", type=int, default=8081)
    parser.add_argument("--devices", type=int, default=10, help="Number of simulated devices")
    parser.add_argument("--rows-per-device", type=int, default=100_000, help="Rows per device")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Vectors per batch")
    parser.add_argument("--dim", type=int, default=10, help="Vector dimensions (matches TDengine's 10 float columns)")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of query latency samples")
    parser.add_argument("--query-ef", type=int, default=50, help="ef for HNSW search")
    parser.add_argument("--skip-tdengine", action="store_true", help="Skip TDengine benchmark")
    parser.add_argument("--skip-vibrato", action="store_true", help="Skip Vibrato Flight benchmark")
    parser.add_argument("--skip-c-abi", action="store_true", help="Skip Embedded C ABI benchmark")
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Vibrato API key token (from `vibrato-db key-create`). "
            "Falls back to VIBRATO_API_KEY env var."
        ),
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("VIBRATO_API_KEY")
    if not args.skip_vibrato and not api_key:
        print("❌ No API key provided. Create one with:")
        print()
        print("  VIBRATO_API_PEPPER=<pepper> vibrato-db key-create \\")
        print("      --data-dir <your-data-dir> \\")
        print("      --name bench --roles admin,query,ingest")
        print()
        print("  Then pass the token= value via --api-key <token> or VIBRATO_API_KEY=<token>")
        sys.exit(1)

    ingest_results = []
    query_results = []

    # --- Vibrato ---
    if not args.skip_vibrato:
        vibrato_ingest = run_vibrato_ingest(
            args.host, args.http_port, args.flight_port, api_key,
            args.devices, args.rows_per_device, args.batch_size, args.dim,
        )
        ingest_results.append(vibrato_ingest)

        vibrato_queries = run_vibrato_queries(
            args.host, args.http_port, api_key, args.dim,
            args.num_queries, k=args.query_k, ef=args.query_ef,
        )
        if vibrato_queries:
            query_results.append(vibrato_queries)

    # --- TDengine (optional) ---
    if not args.skip_tdengine:
        if not check_tdengine_available():
            print("\n  ℹ️  TDengine not installed (taosBenchmark/taos not found). Skipping.")
        else:
            tdengine_ingest = run_tdengine_ingest(args.devices, args.rows_per_device)
            if tdengine_ingest:
                ingest_results.append(tdengine_ingest)
                tdengine_queries = run_tdengine_queries(args.num_queries)
                if tdengine_queries:
                    query_results.append(tdengine_queries)

    # --- Embedded C ABI ---
    if not args.skip_c_abi:
        c_abi_ingest, handle = run_c_abi_ingest(
            args.devices, args.rows_per_device, args.batch_size, args.dim
        )
        if c_abi_ingest and handle:
            ingest_results.append(c_abi_ingest)
            c_abi_queries = run_c_abi_queries(
                handle, args.dim, args.num_queries, k=10, ef=args.query_ef
            )
            if c_abi_queries:
                query_results.append(c_abi_queries)

    # --- Final Output ---
    if ingest_results or query_results:
        print_comparison(ingest_results, query_results)


if __name__ == "__main__":
    main()
