//! Deterministic direct-engine p99 harness for filtered query paths.
//!
//! Measures end-to-end `ProductionState::query()` latency without network overhead.
//! Run with:
//! `cargo bench --bench query_filter_p99`
//!
//! Optional environment overrides:
//! - `VIBRATO_FILTER_BENCH_DIM` (default: 128)
//! - `VIBRATO_FILTER_BENCH_NUM_VECTORS` (default: 20000)
//! - `VIBRATO_FILTER_BENCH_NUM_QUERIES` (default: 20000)
//! - `VIBRATO_FILTER_BENCH_K` (default: 10)
//! - `VIBRATO_FILTER_BENCH_EF` (default: 50)
//! - `VIBRATO_FILTER_BENCH_SHARDS` (default: 8)
//! - `VIBRATO_FILTER_BENCH_QUERY_THREADS` (default: 0, auto)
//! - `VIBRATO_FILTER_BENCH_FILTER_PAR_MIN_SHARDS` (default: 8)

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::tempdir;
use vibrato_core::simd::l2_normalized;
use vibrato_db::prod::{
    bootstrap_data_dirs, CatalogOptions, IngestMetadataV3Input, ProductionConfig, ProductionState,
    SqliteCatalog,
};
use vibrato_server::prod::model::{QueryFilter, QueryRequestV2, SearchTier};

#[derive(Debug, Clone, Copy)]
struct LatencyStats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
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

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let raw = (0..dim)
        .map(|_| rng.gen::<f32>() - 0.5)
        .collect::<Vec<f32>>();
    l2_normalized(&raw)
}

fn make_requests(
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
    filter: Option<QueryFilter>,
) -> Vec<QueryRequestV2> {
    queries
        .iter()
        .map(|q| QueryRequestV2 {
            vector: q.clone(),
            k,
            ef,
            include_metadata: false,
            filter: filter.clone(),
            search_tier: SearchTier::Active,
        })
        .collect()
}

fn measure_case(state: &ProductionState, requests: &[QueryRequestV2]) -> Result<LatencyStats> {
    let warmup = requests.len().min(2_000);
    for request in requests.iter().take(warmup) {
        let _ = state.query(request)?;
    }

    let mut latencies = Vec::with_capacity(requests.len());
    for request in requests {
        let started = Instant::now();
        let _ = state.query(request)?;
        latencies.push(started.elapsed().as_nanos() as u64);
    }
    latencies.sort_unstable();

    Ok(LatencyStats {
        p50_us: ns_to_us(percentile_ns(&latencies, 0.50)),
        p95_us: ns_to_us(percentile_ns(&latencies, 0.95)),
        p99_us: ns_to_us(percentile_ns(&latencies, 0.99)),
    })
}

fn main() -> Result<()> {
    let dim = env_usize("VIBRATO_FILTER_BENCH_DIM", 128);
    let num_vectors = env_usize("VIBRATO_FILTER_BENCH_NUM_VECTORS", 20_000);
    let num_queries = env_usize("VIBRATO_FILTER_BENCH_NUM_QUERIES", 20_000);
    let k = env_usize("VIBRATO_FILTER_BENCH_K", 10);
    let ef = env_usize("VIBRATO_FILTER_BENCH_EF", 50);
    let shards = env_usize("VIBRATO_FILTER_BENCH_SHARDS", 8).max(1);
    let query_threads = std::env::var("VIBRATO_FILTER_BENCH_QUERY_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let filter_parallel_min_shards =
        env_usize("VIBRATO_FILTER_BENCH_FILTER_PAR_MIN_SHARDS", 8).max(1);

    let dir = tempdir()?;
    let mut config =
        ProductionConfig::from_data_dir(dir.path().join("filter_p99"), "bench".to_string(), dim);
    config.public_health_metrics = true;
    config.audio_colocated = false;
    config.checkpoint_interval = Duration::from_secs(3600);
    config.compaction_interval = Duration::from_secs(3600);
    config.hot_index_shards = shards;
    config.query_pool_threads = query_threads;
    config.flight_decode_pool_threads = 1;
    config.filter_parallel_min_shards = filter_parallel_min_shards;

    bootstrap_data_dirs(&config)?;
    let catalog = Arc::new(SqliteCatalog::open_with_options(
        &config.catalog_path(),
        CatalogOptions {
            read_timeout_ms: config.catalog_read_timeout_ms,
            wal_autocheckpoint_pages: config.sqlite_wal_autocheckpoint_pages,
            max_tag_registry_size: config.max_tag_registry_size,
        },
    )?);
    let state = ProductionState::initialize(config, catalog)?;

    let mut ingest_rng = StdRng::seed_from_u64(0xDEADBEEF);
    let mut batch = Vec::with_capacity(512);
    for i in 0..num_vectors {
        let vector = random_vector(dim, &mut ingest_rng);
        let metadata = IngestMetadataV3Input {
            entity_id: i as u64,
            sequence_ts: i as u64,
            tags: vec![format!("tag_{}", i % 256), "bench".to_string()],
            payload: Vec::new(),
        };
        batch.push((vector, metadata, Some(format!("bench-key-{i}"))));
        if batch.len() == 512 {
            let _ = state.ingest_batch_owned(std::mem::take(&mut batch))?;
        }
    }
    if !batch.is_empty() {
        let _ = state.ingest_batch_owned(batch)?;
    }

    let mut query_rng = StdRng::seed_from_u64(0xC0FFEE);
    let queries = (0..num_queries)
        .map(|_| random_vector(dim, &mut query_rng))
        .collect::<Vec<_>>();

    let unfiltered_reqs = make_requests(&queries, k, ef, None);
    let filtered_tag_reqs = make_requests(
        &queries,
        k,
        ef,
        Some(QueryFilter {
            tags_any: Vec::new(),
            tags_all: vec!["tag_7".to_string()],
        }),
    );
    let filtered_combo_reqs = make_requests(
        &queries,
        k,
        ef,
        Some(QueryFilter {
            tags_any: vec!["bench".to_string()],
            tags_all: vec!["tag_7".to_string()],
        }),
    );

    println!(
        "query_filter_p99 benchmark: dim={} vectors={} queries={} k={} ef={} shards={} qthreads={} filter_par_min_shards={}",
        dim,
        num_vectors,
        num_queries,
        k,
        ef,
        shards,
        query_threads,
        filter_parallel_min_shards
    );

    let unfiltered = measure_case(&state, &unfiltered_reqs)?;
    println!(
        "case=unfiltered p50={:.2}us p95={:.2}us p99={:.2}us",
        unfiltered.p50_us, unfiltered.p95_us, unfiltered.p99_us
    );

    let filtered_tag = measure_case(&state, &filtered_tag_reqs)?;
    println!(
        "case=filtered_tag_all p50={:.2}us p95={:.2}us p99={:.2}us",
        filtered_tag.p50_us, filtered_tag.p95_us, filtered_tag.p99_us
    );

    let filtered_combo = measure_case(&state, &filtered_combo_reqs)?;
    println!(
        "case=filtered_combo p50={:.2}us p95={:.2}us p99={:.2}us",
        filtered_combo.p50_us, filtered_combo.p95_us, filtered_combo.p99_us
    );

    Ok(())
}
