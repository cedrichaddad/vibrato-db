use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Barrier;
use tokio::time::sleep;

const DIM: usize = 16;
const DEFAULT_TOTAL_OPS: usize = 1_000_000;
const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 30;
const QUERY_BANK_CAP: usize = 4096;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, port: u16) -> std::io::Result<Child> {
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
        .arg(port.to_string())
        .arg("--checkpoint-interval-secs")
        .arg("3600")
        .arg("--compaction-interval-secs")
        .arg("3600")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn()
}

async fn wait_for_ready_with_child(child: &mut Child, base_url: &str) -> Result<(), String> {
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
        if let Ok(resp) = client.get(&ready_url).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("server did not become ready at {}", ready_url))
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
        .arg("stress")
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

async fn start_ready_server_with_retry(
    data_dir: &Path,
    max_attempts: usize,
) -> anyhow::Result<(Child, u16)> {
    let mut last_err = String::new();
    for _ in 0..max_attempts.max(1) {
        let Some(port) = reserve_local_port() else {
            last_err = "localhost bind unavailable".to_string();
            continue;
        };
        let base_url = format!("http://127.0.0.1:{}", port);
        let mut child = start_server(data_dir, port)
            .await
            .map_err(|e| anyhow::anyhow!("spawn failed on port {}: {}", port, e))?;

        match wait_for_ready_with_child(&mut child, &base_url).await {
            Ok(()) => return Ok((child, port)),
            Err(e) => {
                last_err = format!("port {}: {}", port, e);
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
    delete_attempted: AtomicU64,
    delete_ok: AtomicU64,
    delete_unsupported: AtomicU64,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "high-velocity stress harness; run explicitly with --ignored"]
async fn stress_test_million_ops_mixed() {
    let total_ops = env_usize("VIBRATO_STRESS_TOTAL_OPS", DEFAULT_TOTAL_OPS);
    let concurrency = env_usize("VIBRATO_STRESS_CONCURRENCY", DEFAULT_CONCURRENCY).max(1);
    let seed = env_u64("VIBRATO_STRESS_SEED", DEFAULT_SEED);
    let http_timeout_secs = env_u64(
        "VIBRATO_STRESS_HTTP_TIMEOUT_SECS",
        DEFAULT_HTTP_TIMEOUT_SECS,
    )
    .max(1);
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: stress test running in debug profile; use --release for realistic contention/latency behavior"
        );
    }

    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("stress_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create api key");

    let startup_retries = env_usize("VIBRATO_STRESS_STARTUP_RETRIES", 4);
    let (mut server, port) = match start_ready_server_with_retry(&data_dir, startup_retries).await {
        Ok(v) => v,
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("localhost bind unavailable") {
                eprintln!("skipping stress test: {}", msg);
                return;
            }
            panic!("start stress server: {}", msg);
        }
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(http_timeout_secs))
        .pool_max_idle_per_host(concurrency.saturating_mul(4))
        .tcp_nodelay(true)
        .build()
        .expect("build reqwest client");

    // Seed initial vectors so read pressure starts immediately.
    let query_bank = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
    for i in 0..128usize {
        let vec = normalized_vector(seed ^ 0xA5A5_5A5A, 0, i);
        let body = serde_json::json!({
            "vector": vec.clone(),
            "metadata": {
                "source_file": format!("warmup_{i}.wav"),
                "start_time_ms": i * 10,
                "duration_ms": 80,
                "bpm": 120.0,
                "tags": ["stress", "warmup"]
            },
            "idempotency_key": format!("stress-warmup-{i}")
        });
        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(&token)
            .json(&body)
            .send()
            .await
            .expect("warmup ingest");
        assert_eq!(resp.status(), StatusCode::CREATED, "warmup ingest failed");
        query_bank.write().push(vec);
    }

    let counters = Arc::new(Counters::default());
    let first_error = Arc::new(Mutex::new(None::<String>));
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(concurrency));
    let delete_support = Arc::new(AtomicI8::new(-1)); // -1 unknown, 0 unsupported, 1 supported
    let admin_in_flight = Arc::new(AtomicBool::new(false));
    let max_seen_id = Arc::new(AtomicUsize::new(127));
    let started = Instant::now();

    let mut tasks = Vec::with_capacity(concurrency);
    for worker_id in 0..concurrency {
        let client = client.clone();
        let base_url = base_url.clone();
        let token = token.clone();
        let counters = counters.clone();
        let first_error = first_error.clone();
        let stop = stop.clone();
        let barrier = barrier.clone();
        let query_bank = query_bank.clone();
        let delete_support = delete_support.clone();
        let admin_in_flight = admin_in_flight.clone();
        let max_seen_id = max_seen_id.clone();

        let worker_ops =
            total_ops / concurrency + usize::from(worker_id < (total_ops % concurrency));

        tasks.push(tokio::spawn(async move {
            let mut rng = StdRng::seed_from_u64(seed ^ ((worker_id as u64) << 32));
            barrier.wait().await;

            for local_idx in 0..worker_ops {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let op_roll = rng.gen_range(0..100usize);
                if op_roll < 60 {
                    // Read
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
                        "k": 10,
                        "ef": 40,
                        "include_metadata": false
                    });
                    match client
                        .post(format!("{}/v2/query", base_url))
                        .bearer_auth(&token)
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
                } else if op_roll < 90 {
                    // Write
                    let vec = normalized_vector(seed, worker_id, local_idx + 1_000_000);
                    let idempotency_key = format!("stress-{seed}-{worker_id}-{local_idx}");
                    let body = serde_json::json!({
                        "vector": vec,
                        "metadata": {
                            "source_file": format!("stress_w{worker_id}_o{local_idx}.wav"),
                            "start_time_ms": local_idx as u32,
                            "duration_ms": 64,
                            "bpm": 100.0 + ((local_idx % 64) as f32),
                            "tags": ["stress", format!("worker-{worker_id}")]
                        },
                        "idempotency_key": idempotency_key
                    });
                    match client
                        .post(format!("{}/v2/vectors", base_url))
                        .bearer_auth(&token)
                        .json(&body)
                        .send()
                        .await
                    {
                        Ok(resp) if resp.status() == StatusCode::CREATED => {
                            let payload: serde_json::Value = match resp.json().await {
                                Ok(v) => v,
                                Err(e) => {
                                    record_failure(
                                        &first_error,
                                        &stop,
                                        format!(
                                            "write payload parse failed: {} worker={} op={} seed={}",
                                            e, worker_id, local_idx, seed
                                        ),
                                    );
                                    continue;
                                }
                            };
                            if let Some(id) = payload["data"]["id"].as_u64() {
                                max_seen_id.fetch_max(id as usize, Ordering::Relaxed);
                            }
                            counters.write_ok.fetch_add(1, Ordering::Relaxed);
                            let mut bank = query_bank.write();
                            if bank.len() < QUERY_BANK_CAP {
                                bank.push(vec);
                            } else {
                                let slot = rng.gen_range(0..QUERY_BANK_CAP);
                                bank[slot] = vec;
                            }
                        }
                        Ok(resp) => {
                            record_failure(
                                &first_error,
                                &stop,
                                format!(
                                    "write failed: status={} worker={} op={} seed={}",
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
                                    "write request error: {} worker={} op={} seed={}",
                                    e, worker_id, local_idx, seed
                                ),
                            );
                        }
                    }
                } else if op_roll < 99 {
                    // Delete if supported, else fallback to read to keep pressure high.
                    counters.delete_attempted.fetch_add(1, Ordering::Relaxed);
                    if delete_support.load(Ordering::Relaxed) == 0 {
                        let fallback = normalized_vector(seed ^ 0xDEAD_BEEF, worker_id, local_idx);
                        let body = serde_json::json!({
                            "vector": fallback,
                            "k": 5,
                            "ef": 32,
                            "include_metadata": false
                        });
                        match client
                            .post(format!("{}/v2/query", base_url))
                            .bearer_auth(&token)
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
                                        "delete-fallback-read failed: status={} worker={} op={} seed={}",
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
                                        "delete-fallback-read request error: {} worker={} op={} seed={}",
                                        e, worker_id, local_idx, seed
                                    ),
                                );
                            }
                        }
                        continue;
                    }

                    let target = rng.gen_range(0..=max_seen_id.load(Ordering::Relaxed).max(1));
                    match client
                        .delete(format!("{}/v2/vectors/{}", base_url, target))
                        .bearer_auth(&token)
                        .send()
                        .await
                    {
                        Ok(resp)
                            if resp.status() == StatusCode::NOT_FOUND
                                || resp.status() == StatusCode::METHOD_NOT_ALLOWED =>
                        {
                            delete_support.store(0, Ordering::SeqCst);
                            counters.delete_unsupported.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(resp)
                            if resp.status() == StatusCode::OK
                                || resp.status() == StatusCode::NO_CONTENT
                                || resp.status() == StatusCode::ACCEPTED =>
                        {
                            delete_support.store(1, Ordering::SeqCst);
                            counters.delete_ok.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(resp) => {
                            record_failure(
                                &first_error,
                                &stop,
                                format!(
                                    "delete failed: status={} worker={} op={} seed={}",
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
                                    "delete request error: {} worker={} op={} seed={}",
                                    e, worker_id, local_idx, seed
                                ),
                            );
                        }
                    }
                } else {
                    // Admin chaos path.
                    if admin_in_flight
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                        .is_err()
                    {
                        counters.admin_skipped.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    let path = if rng.gen::<u8>() % 10 < 9 {
                        "v2/admin/compact"
                    } else {
                        "v2/admin/checkpoint"
                    };
                    let result = client
                        .post(format!("{}/{}", base_url, path))
                        .bearer_auth(&token)
                        .timeout(Duration::from_secs(http_timeout_secs.max(20)))
                        .send()
                        .await;
                    admin_in_flight.store(false, Ordering::Release);
                    match result {
                        Ok(resp) if resp.status() == StatusCode::OK => {
                            counters.admin_ok.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) if e.is_timeout() => {
                            counters.admin_timeout.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(resp) => {
                            record_failure(
                                &first_error,
                                &stop,
                                format!(
                                    "admin failed: path={} status={} worker={} op={} seed={}",
                                    path,
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
                                    "admin request error: {} path={} worker={} op={} seed={}",
                                    e, path, worker_id, local_idx, seed
                                ),
                            );
                        }
                    }
                }
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
        panic!("stress test failed: {}", err);
    }

    let ready = client
        .get(format!("{}/v2/health/ready", base_url))
        .send()
        .await
        .expect("ready check after stress");
    assert_eq!(
        ready.status(),
        StatusCode::OK,
        "server became unready during stress run"
    );

    let stats_resp = client
        .get(format!("{}/v2/admin/stats", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("stats request");
    assert_eq!(stats_resp.status(), StatusCode::OK, "stats endpoint failed");
    let payload: serde_json::Value = stats_resp.json().await.expect("stats payload");
    let total_vectors = payload["data"]["total_vectors"].as_u64().unwrap_or(0);

    let writes = counters.write_ok.load(Ordering::Relaxed);
    let deletes = counters.delete_ok.load(Ordering::Relaxed);
    let admin_ok = counters.admin_ok.load(Ordering::Relaxed);
    let admin_timeout = counters.admin_timeout.load(Ordering::Relaxed);
    let expected_min = (128_u64 + writes).saturating_sub(deletes);
    assert!(
        total_vectors >= expected_min,
        "vector count invariant failed: total_vectors={} expected_min={} writes={} deletes={} seed={} elapsed={:?}",
        total_vectors,
        expected_min,
        writes,
        deletes,
        seed,
        started.elapsed()
    );
    assert!(
        admin_ok > 0 || admin_timeout > 0,
        "admin chaos path did not execute (ok={}, timeout={})",
        admin_ok,
        admin_timeout
    );

    eprintln!(
        "stress summary seed={} total_ops={} concurrency={} elapsed={:?} reads={} writes={} delete_attempted={} delete_ok={} delete_unsupported={} admin_ok={} admin_timeout={} admin_skipped={}",
        seed,
        total_ops,
        concurrency,
        started.elapsed(),
        counters.read_ok.load(Ordering::Relaxed),
        writes,
        counters.delete_attempted.load(Ordering::Relaxed),
        deletes,
        counters.delete_unsupported.load(Ordering::Relaxed),
        admin_ok,
        admin_timeout,
        counters.admin_skipped.load(Ordering::Relaxed),
    );

    stop_server_silent(&mut server).await;
}
