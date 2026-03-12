//! Deterministic direct-engine p99 harness for identify paths.
//!
//! Measures end-to-end `ProductionState::identify()` latency without network overhead.
//! Run with:
//! `cargo bench --bench identify_p99`
//!
//! Optional environment overrides:
//! - `VIBRATO_IDENTIFY_BENCH_DIM` (default: 128)
//! - `VIBRATO_IDENTIFY_BENCH_TRACKS` (default: 3)
//! - `VIBRATO_IDENTIFY_BENCH_TRACK_LEN` (default: 1536)
//! - `VIBRATO_IDENTIFY_BENCH_QUERY_LEN` (default: 40)
//! - `VIBRATO_IDENTIFY_BENCH_NUM_QUERIES` (default: 5000)
//! - `VIBRATO_IDENTIFY_BENCH_K` (default: 5)
//! - `VIBRATO_IDENTIFY_BENCH_EF` (default: 320)
//! - `VIBRATO_IDENTIFY_BENCH_INCLUDE_METADATA` (`0` or `1`, default: `0`)
//! - `VIBRATO_IDENTIFY_BENCH_NOISE_SIGMA` (default: 0.025)

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::{tempdir, TempDir};
use vibrato_core::metadata::VectorMetadataV3;
use vibrato_core::simd::l2_normalized;
use vibrato_db::prod::{
    bootstrap_data_dirs, CatalogOptions, IngestMetadataV3Input, ProductionConfig, ProductionState,
    SqliteCatalog,
};
use vibrato_server::prod::model::{IdentifyRequestV2, SearchTier};

#[derive(Debug, Clone, Copy)]
struct LatencyStats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

#[derive(Clone)]
struct TrackData {
    frames: Vec<Vec<f32>>,
    id_offset: usize,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| *v > 0.0)
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

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn deterministic_frame_for_track(dim: usize, track_seed: u64, idx: usize) -> Vec<f32> {
    let mut state =
        splitmix64(track_seed ^ (idx as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0x94D049BB133111EB);
    let mut vector = Vec::with_capacity(dim);
    for lane in 0..dim {
        state = splitmix64(state ^ ((lane as u64).wrapping_mul(0xBF58476D1CE4E5B9)));
        let unit = (state as f64 / u64::MAX as f64) as f32;
        vector.push(unit * 2.0 - 1.0);
    }
    l2_normalized(&vector)
}

fn gaussian_noise(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().clamp(1e-7, 1.0 - 1e-7);
    let u2 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    r * theta.cos()
}

fn add_gaussian_noise(base: &[f32], sigma: f32, rng: &mut StdRng) -> Vec<f32> {
    let noisy = base
        .iter()
        .map(|value| *value + sigma * gaussian_noise(rng))
        .collect::<Vec<_>>();
    l2_normalized(&noisy)
}

fn build_tracks(dim: usize, track_count: usize, track_len: usize) -> Vec<TrackData> {
    let base_seeds = [
        0xA1A1_A1A1_A1A1_A1A1u64,
        0xB2B2_B2B2_B2B2_B2B2u64,
        0xC3C3_C3C3_C3C3_C3C3u64,
        0xD4D4_D4D4_D4D4_D4D4u64,
        0xE5E5_E5E5_E5E5_E5E5u64,
        0xF6F6_F6F6_F6F6_F6F6u64,
    ];

    let mut tracks = Vec::with_capacity(track_count);
    let mut next_id = 0usize;
    for idx in 0..track_count {
        let seed = base_seeds
            .get(idx)
            .copied()
            .unwrap_or_else(|| splitmix64(0x9E37_79B9_7F4A_7C15u64 ^ idx as u64));
        let frames = (0..track_len)
            .map(|frame_idx| deterministic_frame_for_track(dim, seed, frame_idx))
            .collect::<Vec<_>>();
        tracks.push(TrackData {
            frames,
            id_offset: next_id,
        });
        next_id = next_id.saturating_add(track_len);
    }
    tracks
}

fn build_requests(
    tracks: &[TrackData],
    query_len: usize,
    num_queries: usize,
    k: usize,
    ef: usize,
    include_metadata: bool,
    noise_sigma: Option<f32>,
) -> Vec<(IdentifyRequestV2, usize)> {
    let mut rng = StdRng::seed_from_u64(0xDEC0_DED1_CAFE_BABE);
    let mut requests = Vec::with_capacity(num_queries);
    for probe in 0..num_queries {
        let track = &tracks[probe % tracks.len()];
        let max_start = track.frames.len() - query_len - 1;
        let local_start = 64 + (((probe * 197) + (probe / tracks.len()) * 31) % (max_start - 64));
        let query_vectors = track.frames[local_start..local_start + query_len]
            .iter()
            .map(|vector| match noise_sigma {
                Some(sigma) => add_gaussian_noise(vector, sigma, &mut rng),
                None => vector.clone(),
            })
            .collect::<Vec<_>>();
        requests.push((
            IdentifyRequestV2 {
                vectors: query_vectors,
                k,
                ef,
                include_metadata,
                search_tier: SearchTier::Active,
            },
            track.id_offset + local_start,
        ));
    }
    requests
}

fn build_hot_state(
    dim: usize,
    tracks: &[TrackData],
    include_metadata: bool,
) -> Result<(Arc<ProductionState>, TempDir)> {
    let dir = tempdir()?;
    let mut config =
        ProductionConfig::from_data_dir(dir.path().join("identify_hot"), "bench".to_string(), dim);
    config.public_health_metrics = false;
    config.audio_colocated = false;
    config.checkpoint_interval = Duration::from_secs(3600);
    config.compaction_interval = Duration::from_secs(3600);
    config.hot_index_shards = 64;
    config.query_pool_threads = 0;
    config.flight_decode_pool_threads = 1;

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

    for track in tracks {
        for (local_idx, vector) in track.frames.iter().enumerate() {
            let id = track.id_offset + local_idx;
            state.insert_hot_vector(
                id,
                vector.clone(),
                VectorMetadataV3 {
                    entity_id: id as u64,
                    sequence_ts: (local_idx * 20) as u64,
                    tags: Vec::new(),
                    payload: if include_metadata {
                        format!("track={}:frame={local_idx}", track.id_offset).into_bytes()
                    } else {
                        Vec::new()
                    },
                },
            );
        }
    }

    Ok((state, dir))
}

fn build_unindexed_state(
    dim: usize,
    tracks: &[TrackData],
    include_metadata: bool,
) -> Result<(Arc<ProductionState>, TempDir)> {
    let dir = tempdir()?;
    let mut config = ProductionConfig::from_data_dir(
        dir.path().join("identify_unindexed"),
        "bench".to_string(),
        dim,
    );
    config.public_health_metrics = false;
    config.audio_colocated = false;
    config.checkpoint_interval = Duration::from_secs(3600);
    config.compaction_interval = Duration::from_secs(3600);
    config.hot_index_shards = 64;
    config.query_pool_threads = 0;
    config.flight_decode_pool_threads = 1;

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

    let mut batch = Vec::new();
    for track in tracks {
        for (local_idx, vector) in track.frames.iter().enumerate() {
            let id = track.id_offset + local_idx;
            batch.push((
                vector.clone(),
                IngestMetadataV3Input {
                    entity_id: id as u64,
                    sequence_ts: (local_idx * 20) as u64,
                    tags: Vec::new(),
                    payload: if include_metadata {
                        format!("track={}:frame={local_idx}", track.id_offset).into_bytes()
                    } else {
                        Vec::new()
                    },
                },
                Some(format!("bench-identify-{id}")),
            ));
        }
    }
    let results = state.ingest_batch_owned(batch)?;
    if results.iter().any(|(_, created)| !created) {
        return Err(anyhow!("unindexed benchmark unexpectedly deduplicated inserted rows"));
    }

    Ok((state, dir))
}

fn measure_case(state: &ProductionState, requests: &[(IdentifyRequestV2, usize)]) -> Result<LatencyStats> {
    if let Some((request, expected_id)) = requests.first() {
        let response = state.identify(request)?;
        let top_id = response.results.first().map(|row| row.id);
        if top_id != Some(*expected_id) {
            return Err(anyhow!(
                "identify benchmark sanity check failed: expected top id {} got {:?}",
                expected_id,
                top_id
            ));
        }
    }

    let warmup = requests.len().min(1_000);
    for (request, _) in requests.iter().take(warmup) {
        let response = state.identify(request)?;
        black_box(response.results.len());
        if let Some(first) = response.results.first() {
            black_box(first.id);
        }
    }

    let mut latencies = Vec::with_capacity(requests.len());
    for (request, _) in requests {
        let started = Instant::now();
        let response = state.identify(request)?;
        black_box(&response.results);
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
    let dim = env_usize("VIBRATO_IDENTIFY_BENCH_DIM", 128);
    let track_count = env_usize("VIBRATO_IDENTIFY_BENCH_TRACKS", 3).max(1);
    let track_len = env_usize("VIBRATO_IDENTIFY_BENCH_TRACK_LEN", 1_536).max(256);
    let query_len = env_usize("VIBRATO_IDENTIFY_BENCH_QUERY_LEN", 40)
        .min(track_len.saturating_sub(1))
        .max(16);
    let num_queries = env_usize("VIBRATO_IDENTIFY_BENCH_NUM_QUERIES", 5_000);
    let k = env_usize("VIBRATO_IDENTIFY_BENCH_K", 5);
    let ef = env_usize("VIBRATO_IDENTIFY_BENCH_EF", 320);
    let include_metadata = env_bool("VIBRATO_IDENTIFY_BENCH_INCLUDE_METADATA", false);
    let noise_sigma = env_f32("VIBRATO_IDENTIFY_BENCH_NOISE_SIGMA", 0.025);

    let tracks = build_tracks(dim, track_count, track_len);
    let (hot_state, _hot_dir) = build_hot_state(dim, &tracks, include_metadata)?;
    let (unindexed_state, _unindexed_dir) = build_unindexed_state(dim, &tracks, include_metadata)?;

    let exact_requests = build_requests(&tracks, query_len, num_queries, k, ef, include_metadata, None);
    let noisy_requests = build_requests(
        &tracks,
        query_len,
        num_queries,
        k,
        ef,
        include_metadata,
        Some(noise_sigma),
    );

    println!(
        "identify_p99 benchmark: dim={} tracks={} track_len={} query_len={} queries={} k={} ef={} include_metadata={} noise_sigma={}",
        dim,
        track_count,
        track_len,
        query_len,
        num_queries,
        k,
        ef,
        include_metadata,
        noise_sigma
    );

    let hot_exact = measure_case(&hot_state, &exact_requests)?;
    println!(
        "case=hot_exact p50={:.2}us p95={:.2}us p99={:.2}us",
        hot_exact.p50_us, hot_exact.p95_us, hot_exact.p99_us
    );

    let hot_noisy = measure_case(&hot_state, &noisy_requests)?;
    println!(
        "case=hot_noisy p50={:.2}us p95={:.2}us p99={:.2}us",
        hot_noisy.p50_us, hot_noisy.p95_us, hot_noisy.p99_us
    );

    let unindexed_exact = measure_case(&unindexed_state, &exact_requests)?;
    println!(
        "case=unindexed_exact p50={:.2}us p95={:.2}us p99={:.2}us",
        unindexed_exact.p50_us, unindexed_exact.p95_us, unindexed_exact.p99_us
    );

    let unindexed_noisy = measure_case(&unindexed_state, &noisy_requests)?;
    println!(
        "case=unindexed_noisy p50={:.2}us p95={:.2}us p99={:.2}us",
        unindexed_noisy.p50_us, unindexed_noisy.p95_us, unindexed_noisy.p99_us
    );

    Ok(())
}
