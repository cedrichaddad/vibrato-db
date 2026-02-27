use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::tempdir;
use vibrato_db::prod::model::{QueryRequestV2, SearchTier};
use vibrato_db::prod::{
    bootstrap_data_dirs, IngestMetadataV3Input, ProductionConfig, ProductionState, SqliteCatalog,
};

const DIM: usize = 16;
const DEFAULT_TOTAL_OPS: usize = 1_000_000;
const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_MAX_ELAPSED_SECS: u64 = 600;
const QUERY_BANK_CAP: usize = 4096;
const BATCH_SIZE: usize = 256;

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

fn normalized_vector(seed: u64, worker_id: usize, op_idx: usize) -> Vec<f32> {
    let key = ((worker_id as u64) << 32) | (op_idx as u64);
    let stream_seed = seed.wrapping_add(key.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = StdRng::seed_from_u64(stream_seed);
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        v.push(rng.gen::<f32>() * 2.0 - 1.0);
    }
    let signature_u64 = key ^ key.rotate_left(17) ^ 0xD6E8_FD9B_56A9_6C37;
    let sig_a = ((signature_u64 & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
    let sig_b = (((signature_u64 >> 16) & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
    v[0] += sig_a * 0.5;
    if DIM > 1 {
        v[1] += sig_b * 0.5;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn make_metadata(worker_id: usize, op_idx: usize) -> IngestMetadataV3Input {
    IngestMetadataV3Input {
        entity_id: ((worker_id as u64) << 32) | (op_idx as u64),
        sequence_ts: op_idx as u64,
        tags: vec!["stress".to_string()],
        payload: format!("raw_w{worker_id}_o{op_idx}").into_bytes(),
    }
}

#[derive(Default)]
struct Counters {
    read_ok: AtomicU64,
    write_ok: AtomicU64,
    write_batches_ok: AtomicU64,
    admin_ok: AtomicU64,
    admin_skipped: AtomicU64,
}

fn record_failure(first_error: &Mutex<Option<String>>, stop: &AtomicBool, msg: String) {
    let mut slot = first_error.lock();
    if slot.is_none() {
        *slot = Some(msg);
    }
    stop.store(true, Ordering::SeqCst);
}

fn flush_write_buffer(
    state: &Arc<ProductionState>,
    buffer: &mut Vec<(Vec<f32>, IngestMetadataV3Input, Option<String>)>,
    query_bank: &Arc<RwLock<Vec<Vec<f32>>>>,
    counters: &Arc<Counters>,
    rng: &mut StdRng,
) -> Result<(), String> {
    if buffer.is_empty() {
        return Ok(());
    }
    let entries = std::mem::take(buffer);
    let sent = entries.len();
    let results = state
        .ingest_batch(&entries)
        .map_err(|e| format!("direct raw ingest batch failed: {e}"))?;
    if results.len() != sent {
        return Err(format!(
            "direct raw ingest batch mismatch: sent={} got={}",
            sent,
            results.len()
        ));
    }
    {
        let mut bank = query_bank.write();
        for (vector, _, _) in entries {
            if bank.len() < QUERY_BANK_CAP {
                bank.push(vector);
            } else {
                let slot = rng.gen_range(0..QUERY_BANK_CAP);
                bank[slot] = vector;
            }
        }
    }
    counters.write_ok.fetch_add(sent as u64, Ordering::Relaxed);
    counters.write_batches_ok.fetch_add(1, Ordering::Relaxed);
    *buffer = Vec::with_capacity(BATCH_SIZE);
    Ok(())
}

#[test]
#[ignore = "raw direct no-network throughput stress harness; run explicitly with --ignored"]
fn stress_test_million_ops_direct_engine_raw() {
    let total_ops = env_usize("VIBRATO_STRESS_TOTAL_OPS", DEFAULT_TOTAL_OPS);
    let concurrency = env_usize("VIBRATO_STRESS_CONCURRENCY", DEFAULT_CONCURRENCY).max(1);
    let seed = env_u64("VIBRATO_STRESS_SEED", DEFAULT_SEED);
    let max_elapsed_secs = env_u64("VIBRATO_STRESS_MAX_ELAPSED_SECS", DEFAULT_MAX_ELAPSED_SECS);
    let enable_admin_chaos = env_usize("VIBRATO_STRESS_ENABLE_ADMIN_CHAOS", 0) > 0;
    let read_percent = env_usize("VIBRATO_STRESS_RAW_READ_PERCENT", 5).min(90);
    let admin_percent = if enable_admin_chaos { 1 } else { 0 };
    let write_threshold = 100usize.saturating_sub(admin_percent);
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: direct raw stress test running in debug profile; use --release for realistic throughput"
        );
    }

    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("stress_data_direct_raw");
    std::fs::create_dir_all(&data_dir).expect("create data dir");

    let mut config = ProductionConfig::from_data_dir(data_dir, "default".to_string(), DIM);
    config.checkpoint_interval = Duration::from_secs(3600);
    config.compaction_interval = Duration::from_secs(3600);
    bootstrap_data_dirs(&config).expect("bootstrap dirs");

    let catalog = Arc::new(SqliteCatalog::open(&config.catalog_path()).expect("open catalog"));
    let state = ProductionState::initialize(config.clone(), catalog).expect("initialize state");

    let query_bank = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
    let counters = Arc::new(Counters::default());
    let first_error = Arc::new(Mutex::new(None::<String>));
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(concurrency));
    let admin_in_flight = Arc::new(AtomicBool::new(false));
    let started = Instant::now();

    thread::scope(|scope| {
        for worker_id in 0..concurrency {
            let state = state.clone();
            let query_bank = query_bank.clone();
            let counters = counters.clone();
            let first_error = first_error.clone();
            let stop = stop.clone();
            let barrier = barrier.clone();
            let admin_in_flight = admin_in_flight.clone();

            let worker_ops =
                total_ops / concurrency + usize::from(worker_id < (total_ops % concurrency));
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
                    if op_roll < read_percent {
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
                            ef: 64,
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
                                        "direct raw read failed: {} worker={} op={} seed={}",
                                        e, worker_id, local_idx, seed
                                    ),
                                );
                                break;
                            }
                        }
                        local_idx += 1;
                    } else if op_roll < write_threshold {
                        let vector = normalized_vector(seed, worker_id, local_idx + 1_000_000);
                        let metadata = make_metadata(worker_id, local_idx);
                        buffered_writes.push((
                            vector,
                            metadata,
                            Some(format!("direct-raw-{seed}-{worker_id}-{local_idx}")),
                        ));
                        local_idx += 1;
                        if buffered_writes.len() >= BATCH_SIZE {
                            if let Err(msg) = flush_write_buffer(
                                &state,
                                &mut buffered_writes,
                                &query_bank,
                                &counters,
                                &mut rng,
                            ) {
                                record_failure(&first_error, &stop, msg);
                                break;
                            }
                        }
                    } else {
                        if let Err(msg) = flush_write_buffer(
                            &state,
                            &mut buffered_writes,
                            &query_bank,
                            &counters,
                            &mut rng,
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
                                        "direct raw admin failed: {} worker={} op={} seed={}",
                                        e, worker_id, local_idx, seed
                                    ),
                                );
                                break;
                            }
                        }
                        local_idx += 1;
                    }
                }

                if !stop.load(Ordering::Relaxed) && !buffered_writes.is_empty() {
                    if let Err(msg) = flush_write_buffer(
                        &state,
                        &mut buffered_writes,
                        &query_bank,
                        &counters,
                        &mut rng,
                    ) {
                        record_failure(&first_error, &stop, msg);
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.lock().clone() {
        panic!("direct raw stress test failed: {}", err);
    }

    let elapsed = started.elapsed();
    if !cfg!(debug_assertions) {
        assert!(
            elapsed <= Duration::from_secs(max_elapsed_secs),
            "direct raw throughput target missed: elapsed={:?} max_elapsed_secs={} total_ops={} concurrency={}",
            elapsed,
            max_elapsed_secs,
            total_ops,
            concurrency
        );
    }

    let stats = state.stats().expect("stats");
    let writes = counters.write_ok.load(Ordering::Relaxed);
    assert_eq!(
        stats.total_vectors as u64, writes,
        "direct raw vector count invariant failed: total_vectors={} writes={}",
        stats.total_vectors, writes
    );

    let ops_per_sec = if elapsed.as_secs_f64() > 0.0 {
        total_ops as f64 / elapsed.as_secs_f64()
    } else {
        total_ops as f64
    };
    eprintln!(
        "direct raw stress summary seed={} total_ops={} concurrency={} elapsed={:?} ops_per_sec={:.2} reads={} writes={} write_batches={} admin_enabled={} admin_ok={} admin_skipped={}",
        seed,
        total_ops,
        concurrency,
        elapsed,
        ops_per_sec,
        counters.read_ok.load(Ordering::Relaxed),
        writes,
        counters.write_batches_ok.load(Ordering::Relaxed),
        enable_admin_chaos,
        counters.admin_ok.load(Ordering::Relaxed),
        counters.admin_skipped.load(Ordering::Relaxed),
    );
}
