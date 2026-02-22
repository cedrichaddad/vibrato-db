use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::types::Float32Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt16Array, UInt32Array,
};
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::FlightClient;
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
use serde::Deserialize;
use tempfile::tempdir;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::{Barrier, Semaphore};
use tokio::time::sleep;

const DIM: usize = 16;
const DEFAULT_TOTAL_OPS: usize = 1_000_000;
const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 30;
const DEFAULT_FLIGHT_TIMEOUT_SECS: u64 = 90;
const DEFAULT_WRITE_FLUSH_PARALLELISM: usize = 8;
const DEFAULT_WRITE_FLUSH_RETRIES: usize = 4;
const DEFAULT_MAX_ELAPSED_SECS: u64 = 300;
const DEFAULT_MIN_OPS_PER_SEC: f64 = 3500.0;
const QUERY_BANK_CAP: usize = 4096;
const BATCH_SIZE: usize = 200;
const VERIFY_SAMPLE_CAP: usize = 1024;
const WARMUP_VECTORS: usize = 128;
const READ_PERCENT: usize = 50;
const QUERY_K: usize = 8;
const QUERY_EF: usize = 24;

#[derive(Clone)]
struct FlightWriteRequest {
    vector: Vec<f32>,
    source_file: String,
    start_time_ms: u32,
    duration_ms: u16,
    bpm: f32,
    idempotency_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QueryResponseEnvelope {
    data: QueryResponseData,
}

#[derive(Debug, Deserialize)]
struct QueryResponseData {
    results: Vec<QueryResult>,
}

#[derive(Debug, Deserialize)]
struct QueryResult {
    score: f32,
}

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, http_port: u16, flight_port: u16) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v2")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(http_port.to_string())
        .arg("--flight-host")
        .arg("127.0.0.1")
        .arg("--flight-port")
        .arg(flight_port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn()
}

async fn wait_for_ready_with_child(
    child: &mut Child,
    base_url: &str,
    token: &str,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    for _ in 0..900 {
        if let Ok(Some(status)) = child.try_wait() {
            let mut stderr_text = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut stderr_text).await;
            }
            return Err(format!(
                "server exited before ready: status={} url={} stderr={}",
                status,
                ready_url,
                stderr_text.trim()
            ));
        }
        if let Ok(resp) = client.get(&ready_url).bearer_auth(token).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("server did not become ready at {}", ready_url))
}

async fn wait_for_flight_ready(endpoint: &str) -> Result<(), String> {
    for _ in 0..200 {
        let channel = tonic::transport::Endpoint::from_shared(endpoint.to_string())
            .map_err(|e| format!("invalid flight endpoint: {e}"))?;
        if channel.connect().await.is_ok() {
            return Ok(());
        }
        sleep(Duration::from_millis(50)).await;
    }
    Err(format!(
        "flight endpoint did not become ready at {}",
        endpoint
    ))
}

async fn stop_server(child: &mut Child, print_stderr_tail: bool) {
    let mut stderr_handle = child.stderr.take();
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(8), child.wait()).await;
    if print_stderr_tail {
        if let Some(mut stderr) = stderr_handle.take() {
            let mut log = String::new();
            if tokio::time::timeout(Duration::from_secs(2), stderr.read_to_string(&mut log))
                .await
                .is_ok()
                && !log.trim().is_empty()
            {
                eprintln!("\n=== SERVER STDERR (LAST 20 LINES) ===");
                let lines = log.lines().collect::<Vec<_>>();
                let start = lines.len().saturating_sub(20);
                for line in &lines[start..] {
                    eprintln!("{}", line);
                }
                eprintln!("====================================\n");
            }
        }
    }
}

async fn stop_server_silent(child: &mut Child) {
    let _ = stop_server(child, false).await;
}

async fn stop_server_with_logs(child: &mut Child) {
    let _ = stop_server(child, true).await;
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("flight-stress")
        .arg("--roles")
        .arg("admin,query,ingest")
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "key-create failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(token) = line.strip_prefix("token=") {
            return Ok(token.trim().to_string());
        }
    }
    anyhow::bail!("token not found in key-create output: {}", stdout)
}

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

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

async fn start_ready_server_with_retry(
    data_dir: &Path,
    token: &str,
    max_attempts: usize,
) -> anyhow::Result<(Child, u16, u16)> {
    let mut last_err = String::new();
    for _ in 0..max_attempts.max(1) {
        let Some(http_port) = reserve_local_port() else {
            last_err = "localhost http bind unavailable".to_string();
            continue;
        };
        let Some(flight_port) = reserve_local_port() else {
            last_err = "localhost flight bind unavailable".to_string();
            continue;
        };

        let base_url = format!("http://127.0.0.1:{}", http_port);
        let flight_endpoint = format!("http://127.0.0.1:{}", flight_port);
        let mut child = start_server(data_dir, http_port, flight_port)
            .await
            .map_err(|e| anyhow::anyhow!("spawn failed on ports {http_port}/{flight_port}: {e}"))?;

        match wait_for_ready_with_child(&mut child, &base_url, token).await {
            Ok(()) => match wait_for_flight_ready(&flight_endpoint).await {
                Ok(()) => return Ok((child, http_port, flight_port)),
                Err(e) => {
                    last_err = format!("ports {http_port}/{flight_port}: {e}");
                    stop_server_silent(&mut child).await;
                }
            },
            Err(e) => {
                last_err = format!("ports {http_port}/{flight_port}: {e}");
                stop_server_silent(&mut child).await;
            }
        }
    }
    anyhow::bail!(
        "failed to start ready server after {} attempts: {}",
        max_attempts.max(1),
        last_err
    )
}

#[derive(Default)]
struct Counters {
    read_ok: AtomicU64,
    write_ok: AtomicU64,
    write_batches_ok: AtomicU64,
    admin_skipped: AtomicU64,
    admin_timeout: AtomicU64,
    admin_ok: AtomicU64,
}

fn record_failure(first_error: &Mutex<Option<String>>, stop: &AtomicBool, msg: String) {
    let mut slot = first_error.lock();
    if slot.is_none() {
        *slot = Some(msg);
    }
    stop.store(true, Ordering::SeqCst);
}

fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn build_flight_batch(buffer: &[FlightWriteRequest]) -> RecordBatch {
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        buffer
            .iter()
            .map(|row| Some(row.vector.iter().copied().map(Some).collect::<Vec<_>>())),
        DIM as i32,
    );
    let source_file = StringArray::from(
        buffer
            .iter()
            .map(|row| Some(row.source_file.as_str()))
            .collect::<Vec<_>>(),
    );
    let start_time_ms = UInt32Array::from(
        buffer
            .iter()
            .map(|row| row.start_time_ms)
            .collect::<Vec<_>>(),
    );
    let duration_ms =
        UInt16Array::from(buffer.iter().map(|row| row.duration_ms).collect::<Vec<_>>());
    let bpm = Float32Array::from(buffer.iter().map(|row| row.bpm).collect::<Vec<_>>());
    let idempotency_key = StringArray::from(
        buffer
            .iter()
            .map(|row| row.idempotency_key.as_deref())
            .collect::<Vec<_>>(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
        Field::new("source_file", DataType::Utf8, true),
        Field::new("start_time_ms", DataType::UInt32, true),
        Field::new("duration_ms", DataType::UInt16, true),
        Field::new("bpm", DataType::Float32, true),
        Field::new("idempotency_key", DataType::Utf8, true),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(vectors),
            Arc::new(source_file),
            Arc::new(start_time_ms),
            Arc::new(duration_ms),
            Arc::new(bpm),
            Arc::new(idempotency_key),
        ],
    )
    .expect("record batch")
}

async fn send_flight_batch(
    client: &mut FlightClient,
    batch: RecordBatch,
) -> Result<(usize, usize), String> {
    let stream = FlightDataEncoderBuilder::new().build(futures::stream::iter(vec![Ok(batch)]));
    let put_stream = client
        .do_put(stream)
        .await
        .map_err(|e| format!("flight do_put failed: {e}"))?;
    let responses = put_stream
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| format!("flight put result stream failed: {e}"))?;
    if responses.is_empty() {
        return Err("flight do_put returned empty ack stream".to_string());
    }
    let ack: serde_json::Value = serde_json::from_slice(responses[0].app_metadata.as_ref())
        .map_err(|e| format!("invalid flight ack metadata: {e}"))?;

    let accepted = ack["accepted"].as_u64().unwrap_or(0) as usize;
    let created = ack["created"].as_u64().unwrap_or(0) as usize;
    Ok((accepted, created))
}

async fn connect_flight_client(flight_endpoint: &str, token: &str) -> Result<FlightClient, String> {
    let channel = tonic::transport::Endpoint::from_shared(flight_endpoint.to_string())
        .map_err(|e| format!("invalid flight endpoint: {e}"))?
        .connect()
        .await
        .map_err(|e| format!("connect flight endpoint failed: {e}"))?;
    let mut client = FlightClient::new(channel);
    client
        .add_header("authorization", &format!("Bearer {token}"))
        .map_err(|e| format!("set flight auth metadata failed: {e}"))?;
    Ok(client)
}

async fn flush_write_buffer_flight(
    flight_client: &mut FlightClient,
    flight_endpoint: &str,
    token: &str,
    buffer: &mut Vec<FlightWriteRequest>,
    write_flush_sem: &Arc<Semaphore>,
    max_retries: usize,
    query_bank: &Arc<RwLock<Vec<Vec<f32>>>>,
    verification_samples: &Arc<Mutex<Vec<Vec<f32>>>>,
    counters: &Arc<Counters>,
    rng: &mut StdRng,
    worker_id: usize,
    op_idx: usize,
    seed: u64,
) -> Result<(), String> {
    if buffer.is_empty() {
        return Ok(());
    }
    let _permit = write_flush_sem
        .acquire()
        .await
        .map_err(|_| "write flush semaphore closed".to_string())?;

    let payload = std::mem::take(buffer);
    let sent = payload.len();
    let vectors = payload
        .iter()
        .map(|row| row.vector.clone())
        .collect::<Vec<_>>();
    let batch = build_flight_batch(&payload);

    let mut accepted = 0usize;
    let mut created = 0usize;
    for attempt in 0..=max_retries {
        match send_flight_batch(flight_client, batch.clone()).await {
            Ok((acc, cre)) => {
                accepted = acc;
                created = cre;
                break;
            }
            Err(e) => {
                if attempt < max_retries {
                    *flight_client = connect_flight_client(flight_endpoint, token)
                        .await
                        .map_err(|reconnect| {
                            format!("flight reconnect failed after error='{}': {}", e, reconnect)
                        })?;
                    sleep(Duration::from_millis((attempt as u64 + 1) * 125)).await;
                    continue;
                }
                return Err(format!(
                    "flight batch write request error: {} worker={} op={} seed={} sent={} attempt={}",
                    e,
                    worker_id,
                    op_idx,
                    seed,
                    sent,
                    attempt + 1
                ));
            }
        }
    }

    if accepted != sent {
        return Err(format!(
            "flight ack mismatch: accepted={} sent={} worker={} op={} seed={}",
            accepted, sent, worker_id, op_idx, seed
        ));
    }

    {
        let mut bank = query_bank.write();
        let mut sample_pool = verification_samples.lock();
        for vector in &vectors {
            if bank.len() < QUERY_BANK_CAP {
                bank.push(vector.clone());
            } else {
                let slot = rng.gen_range(0..QUERY_BANK_CAP);
                bank[slot] = vector.clone();
            }
            if sample_pool.len() < VERIFY_SAMPLE_CAP {
                sample_pool.push(vector.clone());
            }
        }
    }

    counters
        .write_ok
        .fetch_add(created as u64, Ordering::Relaxed);
    counters.write_batches_ok.fetch_add(1, Ordering::Relaxed);
    *buffer = Vec::with_capacity(BATCH_SIZE);

    Ok(())
}

#[tokio::test]
#[ignore = "network + flight stress harness; run explicitly with --ignored"]
async fn stress_test_million_ops_flight_mixed() {
    let total_ops = env_usize("VIBRATO_STRESS_TOTAL_OPS", DEFAULT_TOTAL_OPS);
    let concurrency = env_usize("VIBRATO_STRESS_CONCURRENCY", DEFAULT_CONCURRENCY).max(1);
    let seed = env_u64("VIBRATO_STRESS_SEED", DEFAULT_SEED);
    let enable_admin_chaos = env_usize("VIBRATO_STRESS_ENABLE_ADMIN_CHAOS", 0) > 0;
    let http_timeout = Duration::from_secs(env_u64(
        "VIBRATO_STRESS_HTTP_TIMEOUT_SECS",
        DEFAULT_HTTP_TIMEOUT_SECS,
    ));
    let _flight_timeout = Duration::from_secs(env_u64(
        "VIBRATO_STRESS_FLIGHT_TIMEOUT_SECS",
        DEFAULT_FLIGHT_TIMEOUT_SECS,
    ));
    let write_flush_parallelism = env_usize(
        "VIBRATO_STRESS_WRITE_FLUSH_PARALLELISM",
        DEFAULT_WRITE_FLUSH_PARALLELISM,
    )
    .max(1);
    let write_flush_retries = env_usize(
        "VIBRATO_STRESS_WRITE_FLUSH_RETRIES",
        DEFAULT_WRITE_FLUSH_RETRIES,
    );
    let max_elapsed_secs = env_u64("VIBRATO_STRESS_MAX_ELAPSED_SECS", DEFAULT_MAX_ELAPSED_SECS);
    let min_ops_per_sec = env_f64("VIBRATO_STRESS_MIN_OPS_PER_SEC", DEFAULT_MIN_OPS_PER_SEC);

    if cfg!(debug_assertions) {
        eprintln!(
            "warning: flight stress test running in debug profile; use --release for realistic behavior"
        );
    }

    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("flight_stress_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create api key");

    let (mut server, http_port, flight_port) =
        match start_ready_server_with_retry(&data_dir, &token, 4).await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skipping flight stress test: {}", e);
                return;
            }
        };

    let base_url = format!("http://127.0.0.1:{}", http_port);
    let flight_endpoint = format!("http://127.0.0.1:{}", flight_port);
    let client = reqwest::Client::new();

    for i in 0..WARMUP_VECTORS {
        let vector = normalized_vector(seed ^ 0xCAFE_BABE, 0, i);
        let body = serde_json::json!({
            "vector": vector,
            "metadata": {
                "source_file": format!("warmup_{i}.wav"),
                "start_time_ms": i * 10,
                "duration_ms": 80,
                "bpm": 120.0,
                "tags": ["flight-stress", "warmup"]
            },
            "idempotency_key": format!("flight-warmup-{i}")
        });
        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(&token)
            .timeout(http_timeout)
            .json(&body)
            .send()
            .await
            .expect("warmup ingest");
        assert_eq!(resp.status(), StatusCode::CREATED, "warmup ingest failed");
    }

    let query_bank = Arc::new(RwLock::new(
        (0..WARMUP_VECTORS)
            .map(|i| normalized_vector(seed ^ 0xCAFE_BABE, 0, i))
            .collect::<Vec<_>>(),
    ));
    let verification_samples = Arc::new(Mutex::new(Vec::<Vec<f32>>::new()));
    let counters = Arc::new(Counters::default());
    let first_error = Arc::new(Mutex::new(None::<String>));
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(concurrency));
    let write_flush_sem = Arc::new(Semaphore::new(write_flush_parallelism));
    let admin_in_flight = Arc::new(AtomicBool::new(false));
    let started = Instant::now();

    let mut tasks = Vec::with_capacity(concurrency);
    for worker_id in 0..concurrency {
        let client = client.clone();
        let base_url = base_url.clone();
        let flight_endpoint = flight_endpoint.clone();
        let token = token.clone();
        let counters = counters.clone();
        let first_error = first_error.clone();
        let stop = stop.clone();
        let barrier = barrier.clone();
        let query_bank = query_bank.clone();
        let write_flush_sem = write_flush_sem.clone();
        let admin_in_flight = admin_in_flight.clone();
        let verification_samples = verification_samples.clone();

        let worker_ops =
            total_ops / concurrency + usize::from(worker_id < (total_ops % concurrency));
        let write_threshold = if enable_admin_chaos { 99 } else { 100 };

        tasks.push(tokio::spawn(async move {
            let mut rng = StdRng::seed_from_u64(seed ^ ((worker_id as u64) << 32));
            let mut flight_client = match connect_flight_client(&flight_endpoint, &token).await {
                Ok(client) => client,
                Err(e) => {
                    record_failure(
                        &first_error,
                        &stop,
                        format!(
                            "flight client init failed: {} worker={} seed={}",
                            e, worker_id, seed
                        ),
                    );
                    return;
                }
            };
            let mut buffered_writes = Vec::with_capacity(BATCH_SIZE);
            barrier.wait().await;

            let mut local_idx = 0usize;
            while local_idx < worker_ops {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let op_roll = rng.gen_range(0..100usize);
                if op_roll < READ_PERCENT {
                    let query_vec = {
                        let bank = query_bank.read();
                        if bank.is_empty() {
                            normalized_vector(seed, worker_id, local_idx)
                        } else {
                            bank[rng.gen_range(0..bank.len())].clone()
                        }
                    };
                    let body = serde_json::json!({
                        "vector": query_vec,
                        "k": QUERY_K,
                        "ef": QUERY_EF,
                        "include_metadata": false,
                        "search_tier": "active"
                    });
                    match client
                        .post(format!("{}/v2/query", base_url))
                        .bearer_auth(&token)
                        .timeout(http_timeout)
                        .json(&body)
                        .send()
                        .await
                    {
                        Ok(resp) if resp.status() == StatusCode::OK => {
                            counters.read_ok.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(resp) => {
                            record_failure(
                                &first_error,
                                &stop,
                                format!(
                                    "read failed: status={} worker={} op={} seed={}",
                                    resp.status(),
                                    worker_id,
                                    local_idx,
                                    seed
                                ),
                            );
                        }
                        Err(e) => {
                            record_failure(
                                &first_error,
                                &stop,
                                format!(
                                    "read request error: {} worker={} op={} seed={}",
                                    e, worker_id, local_idx, seed
                                ),
                            );
                        }
                    }
                    local_idx += 1;
                } else if op_roll < write_threshold {
                    let vector = normalized_vector(seed, worker_id, local_idx + 1_000_000);
                    buffered_writes.push(FlightWriteRequest {
                        vector,
                        source_file: format!("flight_stress_w{worker_id}_o{local_idx}.wav"),
                        start_time_ms: local_idx as u32,
                        duration_ms: 64,
                        bpm: 100.0 + ((local_idx % 64) as f32),
                        idempotency_key: Some(format!("flight-{seed}-{worker_id}-{local_idx}")),
                    });
                    local_idx += 1;

                    if buffered_writes.len() >= BATCH_SIZE {
                        if let Err(msg) = flush_write_buffer_flight(
                            &mut flight_client,
                            &flight_endpoint,
                            &token,
                            &mut buffered_writes,
                            &write_flush_sem,
                            write_flush_retries,
                            &query_bank,
                            &verification_samples,
                            &counters,
                            &mut rng,
                            worker_id,
                            local_idx,
                            seed,
                        )
                        .await
                        {
                            record_failure(&first_error, &stop, msg);
                            break;
                        }
                    }
                } else {
                    if let Err(msg) = flush_write_buffer_flight(
                        &mut flight_client,
                        &flight_endpoint,
                        &token,
                        &mut buffered_writes,
                        &write_flush_sem,
                        write_flush_retries,
                        &query_bank,
                        &verification_samples,
                        &counters,
                        &mut rng,
                        worker_id,
                        local_idx,
                        seed,
                    )
                    .await
                    {
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

                    let admin_path = if rng.gen::<u8>() % 10 < 9 {
                        "/v2/admin/compact"
                    } else {
                        "/v2/admin/checkpoint"
                    };
                    let admin_resp = tokio::time::timeout(
                        http_timeout,
                        client
                            .post(format!("{}{}", base_url, admin_path))
                            .bearer_auth(&token)
                            .json(&serde_json::json!({}))
                            .send(),
                    )
                    .await;

                    admin_in_flight.store(false, Ordering::Release);

                    match admin_resp {
                        Ok(request) => match request {
                            Ok(resp)
                                if resp.status() == StatusCode::OK
                                    || resp.status() == StatusCode::ACCEPTED =>
                            {
                                counters.admin_ok.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(resp) => {
                                if is_retryable_status(resp.status()) {
                                    counters.admin_timeout.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    record_failure(
                                        &first_error,
                                        &stop,
                                        format!(
                                            "admin failed: status={} worker={} op={} seed={}",
                                            resp.status(),
                                            worker_id,
                                            local_idx,
                                            seed
                                        ),
                                    );
                                }
                            }
                            Err(e) => {
                                counters.admin_timeout.fetch_add(1, Ordering::Relaxed);
                                tracing::debug!("admin request error: {}", e);
                            }
                        },
                        Err(_) => {
                            counters.admin_timeout.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    local_idx += 1;
                }
            }

            if let Err(msg) = flush_write_buffer_flight(
                &mut flight_client,
                &flight_endpoint,
                &token,
                &mut buffered_writes,
                &write_flush_sem,
                write_flush_retries,
                &query_bank,
                &verification_samples,
                &counters,
                &mut rng,
                worker_id,
                local_idx,
                seed,
            )
            .await
            {
                record_failure(&first_error, &stop, msg);
            }
        }));
    }

    for task in tasks {
        if let Err(e) = task.await {
            record_failure(
                &first_error,
                &stop,
                format!("worker join error: {} seed={}", e, seed),
            );
        }
    }

    if let Some(err) = first_error.lock().clone() {
        stop_server_with_logs(&mut server).await;
        panic!("flight stress test failed: {}", err);
    }

    let ready = client
        .get(format!("{}/v2/health/ready", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("ready check after stress");
    assert_eq!(
        ready.status(),
        StatusCode::OK,
        "server became unready during flight stress run"
    );

    let stats_resp = client
        .get(format!("{}/v2/admin/stats", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("stats request");
    assert_eq!(stats_resp.status(), StatusCode::OK);
    let payload: serde_json::Value = stats_resp.json().await.expect("stats payload");
    let total_vectors = payload["data"]["total_vectors"].as_u64().unwrap_or(0);

    let writes = counters.write_ok.load(Ordering::Relaxed);
    let write_batches = counters.write_batches_ok.load(Ordering::Relaxed);
    let admin_ok = counters.admin_ok.load(Ordering::Relaxed);
    let admin_timeout = counters.admin_timeout.load(Ordering::Relaxed);
    let expected_total = WARMUP_VECTORS as u64 + writes;

    assert_eq!(
        total_vectors,
        expected_total,
        "vector count invariant failed: total_vectors={} expected_total={} writes={} seed={} elapsed={:?}",
        total_vectors,
        expected_total,
        writes,
        seed,
        started.elapsed()
    );

    if enable_admin_chaos {
        assert!(
            admin_ok > 0 || admin_timeout > 0,
            "admin chaos path did not execute (ok={}, timeout={})",
            admin_ok,
            admin_timeout
        );
    }

    let samples = verification_samples.lock().clone();
    let verify_count = if writes > 0 {
        samples.len().min(200)
    } else {
        0
    };
    if writes > 0 {
        assert!(
            verify_count > 0,
            "no verification samples were captured from flight writes"
        );

        for (idx, vector) in samples.iter().take(verify_count).enumerate() {
            let body = serde_json::json!({
                "vector": vector,
                "k": 20,
                "ef": 256,
                "include_metadata": false,
                "search_tier": if enable_admin_chaos { "all" } else { "active" }
            });
            let resp = client
                .post(format!("{}/v2/query", base_url))
                .bearer_auth(&token)
                .json(&body)
                .send()
                .await
                .expect("verification query request");
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "verification query failed for sample={}",
                idx
            );
            let parsed: QueryResponseEnvelope =
                resp.json().await.expect("verification query payload");
            assert!(
                !parsed.data.results.is_empty(),
                "verification query returned empty result set for sample={}",
                idx
            );
            let best_score = parsed.data.results[0].score;
            assert!(
                best_score > 0.75,
                "verification query score too low for sample={} score={}",
                idx,
                best_score
            );
        }
    }

    let elapsed = started.elapsed();
    let achieved_ops_per_sec = if elapsed.is_zero() {
        0.0
    } else {
        total_ops as f64 / elapsed.as_secs_f64()
    };

    eprintln!(
        "flight stress summary seed={} total_ops={} concurrency={} elapsed={:?} ops_per_sec={:.2} reads={} writes={} write_batches={} verify_samples={} admin_enabled={} admin_ok={} admin_timeout={} admin_skipped={}",
        seed,
        total_ops,
        concurrency,
        elapsed,
        achieved_ops_per_sec,
        counters.read_ok.load(Ordering::Relaxed),
        writes,
        write_batches,
        verify_count,
        enable_admin_chaos,
        admin_ok,
        admin_timeout,
        counters.admin_skipped.load(Ordering::Relaxed),
    );

    if !cfg!(debug_assertions) {
        assert!(
            achieved_ops_per_sec >= min_ops_per_sec,
            "throughput target missed: ops_per_sec={:.2} min_ops_per_sec={:.2} elapsed={:?} total_ops={} concurrency={} reads={} writes={} write_batches={}",
            achieved_ops_per_sec,
            min_ops_per_sec,
            elapsed,
            total_ops,
            concurrency,
            counters.read_ok.load(Ordering::Relaxed),
            writes,
            write_batches
        );
        assert!(
            elapsed <= Duration::from_secs(max_elapsed_secs),
            "elapsed target missed: elapsed={:?} max_elapsed_secs={} total_ops={} concurrency={} ops_per_sec={:.2} reads={} writes={} write_batches={}",
            elapsed,
            max_elapsed_secs,
            total_ops,
            concurrency,
            achieved_ops_per_sec,
            counters.read_ok.load(Ordering::Relaxed),
            writes,
            write_batches
        );
    }

    stop_server_silent(&mut server).await;
}
