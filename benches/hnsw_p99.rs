//! Deterministic p99 latency harness for HNSW search.
//!
//! Run with:
//! `cargo bench --bench hnsw_p99`
//!
//! Optional environment overrides:
//! - `VIBRATO_BENCH_DIM` (default: 128)
//! - `VIBRATO_BENCH_NUM_VECTORS` (default: 5000)
//! - `VIBRATO_BENCH_NUM_QUERIES` (default: 20000)
//! - `VIBRATO_BENCH_QUERY_MODE` (`fixed` or `mixed`, default: `fixed`)

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use vibrato_core::hnsw::HNSW;
use vibrato_core::simd::l2_normalized;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    l2_normalized(&raw)
}

fn percentile_ns(sorted: &[u64], p: f64) -> u64 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    let rank = ((p * n as f64).ceil() as usize).clamp(1, n);
    sorted[rank - 1]
}

fn ns_to_us(v: u64) -> f64 {
    v as f64 / 1_000.0
}

fn main() {
    let dim = env_usize("VIBRATO_BENCH_DIM", 128);
    let num_vectors = env_usize("VIBRATO_BENCH_NUM_VECTORS", 5_000);
    let num_queries = env_usize("VIBRATO_BENCH_NUM_QUERIES", 20_000);
    let mode = env_string("VIBRATO_BENCH_QUERY_MODE", "fixed");

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| random_vector(dim, &mut rng))
        .collect();

    let vectors_for_index = vectors.clone();
    let mut hnsw =
        HNSW::new_with_accessor_and_seed(16, 100, move |id, sink| sink(&vectors_for_index[id]), 42);
    for id in 0..num_vectors {
        hnsw.insert(id);
    }

    let mut query_rng = StdRng::seed_from_u64(0xBADC0DE);
    let queries_mixed: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let anchor = query_rng.gen_range(0..num_vectors);
            let mut q = vectors[anchor].clone();
            for val in q.iter_mut().take(8) {
                *val += (query_rng.gen::<f32>() - 0.5) * 0.01;
            }
            l2_normalized(&q)
        })
        .collect();

    // Warmup
    for q in queries_mixed.iter().take(2_000) {
        let _ = hnsw.search(q, 10, 50);
    }

    println!(
        "hnsw_p99 benchmark: dim={} vectors={} queries={} mode={}",
        dim, num_vectors, num_queries, mode
    );

    for ef in [20usize, 50, 100] {
        let fixed_query = {
            let mut rng = StdRng::seed_from_u64(ef as u64 + 5000);
            random_vector(dim, &mut rng)
        };
        for i in 0..2_000usize.min(num_queries) {
            let q = if mode == "mixed" {
                &queries_mixed[i]
            } else {
                &fixed_query
            };
            let _ = hnsw.search(q, 10, ef);
        }
        let mut latencies = Vec::with_capacity(num_queries);
        for i in 0..num_queries {
            let q = if mode == "mixed" {
                &queries_mixed[i]
            } else {
                &fixed_query
            };
            let start = Instant::now();
            let _ = hnsw.search(q, 10, ef);
            latencies.push(start.elapsed().as_nanos() as u64);
        }
        latencies.sort_unstable();

        let p50 = percentile_ns(&latencies, 0.50);
        let p95 = percentile_ns(&latencies, 0.95);
        let p99 = percentile_ns(&latencies, 0.99);

        println!(
            "ef={} p50={:.2}us p95={:.2}us p99={:.2}us",
            ef,
            ns_to_us(p50),
            ns_to_us(p95),
            ns_to_us(p99)
        );
    }
}
