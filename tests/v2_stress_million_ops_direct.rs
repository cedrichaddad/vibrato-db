use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::tempdir;
use vibrato_core::metadata::VectorMetadata;
use vibrato_db::prod::model::{QueryRequestV2, SearchTier};
use vibrato_db::prod::{bootstrap_data_dirs, ProductionConfig, ProductionState, SqliteCatalog};

const DIM: usize = 16;
const DEFAULT_TOTAL_OPS: usize = 1_000_000;
const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_MAX_ELAPSED_SECS: u64 = 60;
const QUERY_BANK_CAP: usize = 4096;
const BATCH_SIZE: usize = 100;
const VERIFY_SAMPLE_CAP: usize = 1024;
const WARMUP_VECTORS: usize = 128;

fn normalized_vector(seed: u64, worker_id: usize, op_idx: usize) -> Vec<f32> {
    let stream_seed =
        seed ^ ((worker_id as u64) << 32) ^ ((op_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = StdRng::seed_from_u64(stream_seed);
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        v.push(rng.gen::<f32>() * 2.0 - 1.0);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

#[derive(Default)]
struct Counters {
    read_ok: AtomicU64,
    write_ok: AtomicU64,
    write_batches_ok: AtomicU64,
    admin_skipped: AtomicU64,
    admin_ok: AtomicU64,
}

fn record_failure(first_error: &Mutex<Option<String>>, stop: &AtomicBool, msg: String) {
    let mut slot = first_error.lock();
    if slot.is_none() {
        *slot = Some(msg);
    }
    stop.store(true, Ordering::SeqCst);
}

fn flush_write_buffer_direct(
    state: &Arc<ProductionState>,
    buffer: &mut Vec<(Vec<f32>, VectorMetadata, Option<String>)>,
    query_bank: &Arc<RwLock<Vec<Vec<f32>>>>,
    verification_samples: &Arc<Mutex<Vec<(usize, Vec<f32>)>>>,
    max_seen_id: &Arc<AtomicUsize>,
    counters: &Arc<Counters>,
    rng: &mut StdRng,
    worker_id: usize,
    op_idx: usize,
    seed: u64,
) -> Result<(), String> {
    if buffer.is_empty() {
        return Ok(());
    }
    let entries = std::mem::take(buffer);
    let sent = entries.len();
    let vectors = entries
        .iter()
        .map(|(vector, _, _)| vector.clone())
        .collect::<Vec<_>>();
    let results = state.ingest_batch(&entries).map_err(|e| {
        format!(
            "direct batch ingest failed: {} worker={} op={} seed={} sent={}",
            e, worker_id, op_idx, seed, sent
        )
    })?;
    if results.len() != sent {
        return Err(format!(
            "direct batch result length mismatch: sent={} got={} worker={} op={} seed={}",
            sent,
            results.len(),
            worker_id,
            op_idx,
            seed
        ));
    }

    {
        let mut bank = query_bank.write();
        let mut sample_pool = verification_samples.lock();
        for ((id, _created), vector) in results.iter().zip(vectors.iter()) {
            max_seen_id.fetch_max(*id, Ordering::Relaxed);
            if bank.len() < QUERY_BANK_CAP {
                bank.push(vector.clone());
            } else {
                let slot = rng.gen_range(0..QUERY_BANK_CAP);
                bank[slot] = vector.clone();
            }
            if sample_pool.len() < VERIFY_SAMPLE_CAP {
                sample_pool.push((*id, vector.clone()));
            }
        }
    }

    counters.write_ok.fetch_add(sent as u64, Ordering::Relaxed);
    counters.write_batches_ok.fetch_add(1, Ordering::Relaxed);
    *buffer = Vec::with_capacity(BATCH_SIZE);
    Ok(())
}

#[test]
#[ignore = "direct no-network stress harness; run explicitly with --ignored"]
fn stress_test_million_ops_direct_engine() {
    let total_ops = env_usize("VIBRATO_STRESS_TOTAL_OPS", DEFAULT_TOTAL_OPS);
    let concurrency = env_usize("VIBRATO_STRESS_CONCURRENCY", DEFAULT_CONCURRENCY).max(1);
    let seed = env_u64("VIBRATO_STRESS_SEED", DEFAULT_SEED);
    let enable_admin_chaos = env_usize("VIBRATO_STRESS_ENABLE_ADMIN_CHAOS", 0) > 0;
    let max_elapsed_secs = env_u64("VIBRATO_STRESS_MAX_ELAPSED_SECS", DEFAULT_MAX_ELAPSED_SECS);
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: direct stress test running in debug profile; use --release for realistic behavior"
        );
    }

    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("stress_data_direct");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let mut config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), DIM);
    config.checkpoint_interval = Duration::from_secs(3600);
    config.compaction_interval = Duration::from_secs(3600);
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = Arc::new(SqliteCatalog::open(&config.catalog_path()).expect("open catalog"));
    let state = ProductionState::initialize(config, catalog).expect("initialize state");

    let query_bank = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
    for i in 0..WARMUP_VECTORS {
        let vector = normalized_vector(seed ^ 0xA5A5_5A5A, 0, i);
        let metadata = VectorMetadata {
            source_file: format!("warmup_{i}.wav"),
            start_time_ms: (i * 10) as u32,
            duration_ms: 80,
            bpm: 120.0,
            tags: vec!["stress".to_string(), "warmup".to_string()],
        };
        state
            .ingest_vector(&vector, &metadata, Some(&format!("direct-warmup-{i}")))
            .expect("warmup ingest");
        query_bank.write().push(vector);
    }

    let counters = Arc::new(Counters::default());
    let first_error = Arc::new(Mutex::new(None::<String>));
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(concurrency));
    let admin_in_flight = Arc::new(AtomicBool::new(false));
    let max_seen_id = Arc::new(AtomicUsize::new(WARMUP_VECTORS.saturating_sub(1)));
    let verification_samples = Arc::new(Mutex::new(Vec::<(usize, Vec<f32>)>::new()));
    let started = Instant::now();

    thread::scope(|scope| {
        for worker_id in 0..concurrency {
            let state = state.clone();
            let counters = counters.clone();
            let first_error = first_error.clone();
            let stop = stop.clone();
            let barrier = barrier.clone();
            let query_bank = query_bank.clone();
            let admin_in_flight = admin_in_flight.clone();
            let max_seen_id = max_seen_id.clone();
            let verification_samples = verification_samples.clone();

            let worker_ops =
                total_ops / concurrency + usize::from(worker_id < (total_ops % concurrency));
            let write_threshold = if enable_admin_chaos { 99 } else { 100 };

            scope.spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed ^ ((worker_id as u64) << 32));
                let mut buffered_writes = Vec::with_capacity(BATCH_SIZE);
                barrier.wait();

                let mut local_idx = 0usize;
                while local_idx < worker_ops {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    let op_roll = rng.gen_range(0..100usize);
                    if op_roll < 60 {
                        let query_vec = {
                            let bank = query_bank.read();
                            if bank.is_empty() {
                                normalized_vector(seed, worker_id, local_idx)
                            } else {
                                bank[rng.gen_range(0..bank.len())].clone()
                            }
                        };
                        let request = QueryRequestV2 {
                            vector: query_vec,
                            k: 10,
                            ef: 40,
                            include_metadata: false,
                            filter: None,
                            search_tier: SearchTier::Active,
                        };
                        match state.query(&request) {
                            Ok(_) => {
                                counters.read_ok.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => {
                                record_failure(
                                    &first_error,
                                    &stop,
                                    format!(
                                        "direct read failed: {} worker={} op={} seed={}",
                                        e, worker_id, local_idx, seed
                                    ),
                                );
                            }
                        }
                        local_idx += 1;
                    } else if op_roll < write_threshold {
                        let vector = normalized_vector(seed, worker_id, local_idx + 1_000_000);
                        let metadata = VectorMetadata {
                            source_file: format!("stress_w{worker_id}_o{local_idx}.wav"),
                            start_time_ms: local_idx as u32,
                            duration_ms: 64,
                            bpm: 100.0 + ((local_idx % 64) as f32),
                            tags: vec!["stress".to_string(), format!("worker-{worker_id}")],
                        };
                        buffered_writes.push((
                            vector,
                            metadata,
                            Some(format!("direct-{seed}-{worker_id}-{local_idx}")),
                        ));
                        local_idx += 1;

                        if buffered_writes.len() >= BATCH_SIZE {
                            if let Err(msg) = flush_write_buffer_direct(
                                &state,
                                &mut buffered_writes,
                                &query_bank,
                                &verification_samples,
                                &max_seen_id,
                                &counters,
                                &mut rng,
                                worker_id,
                                local_idx,
                                seed,
                            ) {
                                record_failure(&first_error, &stop, msg);
                                break;
                            }
                        }
                    } else {
                        if let Err(msg) = flush_write_buffer_direct(
                            &state,
                            &mut buffered_writes,
                            &query_bank,
                            &verification_samples,
                            &max_seen_id,
                            &counters,
                            &mut rng,
                            worker_id,
                            local_idx,
                            seed,
                        ) {
                            record_failure(&first_error, &stop, msg);
                            break;
                        }

                        if admin_in_flight
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                            .is_err()
                        {
                            counters.admin_skipped.fetch_add(1, Ordering::Relaxed);
                            local_idx += 1;
                            continue;
                        }

                        let admin_result = if rng.gen::<u8>() % 10 < 9 {
                            state.compact_once()
                        } else {
                            state.checkpoint_once()
                        };
                        admin_in_flight.store(false, Ordering::Release);
                        match admin_result {
                            Ok(_) => {
                                counters.admin_ok.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => {
                                record_failure(
                                    &first_error,
                                    &stop,
                                    format!(
                                        "direct admin failed: {} worker={} op={} seed={}",
                                        e, worker_id, local_idx, seed
                                    ),
                                );
                            }
                        }
                        local_idx += 1;
                    }
                }

                if !stop.load(Ordering::Relaxed)
                    && !buffered_writes.is_empty()
                    && first_error.lock().is_none()
                {
                    if let Err(msg) = flush_write_buffer_direct(
                        &state,
                        &mut buffered_writes,
                        &query_bank,
                        &verification_samples,
                        &max_seen_id,
                        &counters,
                        &mut rng,
                        worker_id,
                        local_idx,
                        seed,
                    ) {
                        record_failure(&first_error, &stop, msg);
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.lock().clone() {
        panic!("direct stress test failed: {}", err);
    }

    let stats = state.stats().expect("stats");
    let writes = counters.write_ok.load(Ordering::Relaxed);
    let write_batches = counters.write_batches_ok.load(Ordering::Relaxed);
    let expected_total = WARMUP_VECTORS as u64 + writes;
    assert_eq!(
        stats.total_vectors as u64, expected_total,
        "direct vector count invariant failed: total_vectors={} expected_total={} writes={} seed={} elapsed={:?}",
        stats.total_vectors,
        expected_total,
        writes,
        seed,
        started.elapsed()
    );
    let max_observed = max_seen_id.load(Ordering::Relaxed) as u64;
    assert_eq!(
        max_observed + 1,
        expected_total,
        "direct max id invariant failed: max_seen_id={} expected_total={} writes={} seed={}",
        max_observed,
        expected_total,
        writes,
        seed
    );

    let samples = verification_samples.lock().clone();
    let verify_count = if writes > 0 {
        samples.len().min(200)
    } else {
        0
    };
    if writes > 0 {
        assert!(
            verify_count > 0,
            "direct harness did not capture verification samples"
        );
        for (idx, (expected_id, vector)) in samples.iter().take(verify_count).enumerate() {
            let request = QueryRequestV2 {
                vector: vector.clone(),
                k: 20,
                ef: 256,
                include_metadata: false,
                filter: None,
                search_tier: if enable_admin_chaos {
                    SearchTier::All
                } else {
                    SearchTier::Active
                },
            };
            let response = state.query(&request).expect("direct verify query");
            let found = response.results.iter().any(|r| r.id == *expected_id);
            assert!(
                found,
                "direct verification query missing expected id: sample={} expected_id={} top_ids={:?}",
                idx,
                expected_id,
                response
                    .results
                    .iter()
                    .take(5)
                    .map(|r| r.id)
                    .collect::<Vec<_>>()
            );
        }
    }

    let elapsed = started.elapsed();
    if !cfg!(debug_assertions) {
        assert!(
            elapsed <= Duration::from_secs(max_elapsed_secs),
            "direct throughput target missed: elapsed={:?} max_elapsed_secs={} total_ops={} concurrency={}",
            elapsed,
            max_elapsed_secs,
            total_ops,
            concurrency
        );
    }

    eprintln!(
        "direct stress summary seed={} total_ops={} concurrency={} elapsed={:?} reads={} writes={} write_batches={} verify_samples={} admin_enabled={} admin_ok={} admin_skipped={}",
        seed,
        total_ops,
        concurrency,
        elapsed,
        counters.read_ok.load(Ordering::Relaxed),
        writes,
        write_batches,
        verify_count,
        enable_admin_chaos,
        counters.admin_ok.load(Ordering::Relaxed),
        counters.admin_skipped.load(Ordering::Relaxed),
    );
}
