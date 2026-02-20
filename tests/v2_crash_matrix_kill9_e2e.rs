use std::net::TcpListener;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;
use vibrato_db::prod::{CatalogStore, ProductionConfig, SqliteCatalog};

const DIM: usize = 16;

#[derive(Clone, Debug)]
struct AckRecord {
    vector_id: usize,
    idempotency_key: String,
    vector: Vec<f32>,
    source_file: String,
    start_time_ms: u32,
    duration_ms: u16,
    bpm: f32,
    tags: Vec<String>,
}

#[derive(Clone, Debug)]
struct ReplayRecord {
    idempotency_key: String,
    vector: Vec<f32>,
    source_file: String,
    start_time_ms: u32,
    duration_ms: u16,
    bpm: f32,
    tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct IngestRequest {
    vector: Vec<f32>,
    metadata: serde_json::Value,
    idempotency_key: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct IngestBatchRequest {
    vectors: Vec<IngestRequest>,
}

#[derive(Clone, Debug, Deserialize)]
struct IngestBatchResponseEnvelope {
    data: IngestBatchResponseData,
}

#[derive(Clone, Debug, Deserialize)]
struct IngestBatchResponseData {
    results: Vec<IngestBatchResult>,
}

#[derive(Clone, Debug, Deserialize)]
struct IngestBatchResult {
    id: usize,
    created: bool,
}

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
        .arg("--public-health-metrics")
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    for _ in 0..200 {
        if let Ok(resp) = client.get(&ready_url).send().await {
            if resp.status() == StatusCode::OK {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("server did not become ready at {}", ready_url))
}

async fn stop_server(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

async fn kill_server(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("crash-matrix")
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

fn normalized_vector(seed: u64, idx: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed ^ ((idx as u64).wrapping_mul(0x9E3779B97F4A7C15)));
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

async fn ingest_one(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    seed: u64,
    idx: usize,
) -> Option<AckRecord> {
    let vector = normalized_vector(seed, idx);
    let source_file = format!("seed{seed}_sample_{idx}.wav");
    let start_time_ms = (idx as u32).saturating_mul(10);
    let duration_ms = 50u16;
    let bpm = 90.0 + ((idx % 80) as f32 * 0.5);
    let tags = vec!["drums".to_string(), format!("seed-{seed}")];
    let idempotency_key = format!("seed-{seed}-vec-{idx}");
    let body = serde_json::json!({
        "vector": vector,
        "metadata": {
            "source_file": source_file,
            "start_time_ms": start_time_ms,
            "duration_ms": duration_ms,
            "bpm": bpm,
            "tags": tags
        },
        "idempotency_key": idempotency_key
    });

    let resp = client
        .post(format!("{}/v2/vectors", base_url))
        .bearer_auth(token)
        .json(&body)
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(_) => return None,
    };
    if resp.status() != StatusCode::CREATED {
        return None;
    }
    let payload: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return None,
    };
    let vector_id = payload["data"]["id"].as_u64()? as usize;
    let created = payload["data"]["created"].as_bool().unwrap_or(true);
    if !created {
        return Some(AckRecord {
            vector_id,
            idempotency_key: format!("seed-{seed}-vec-{idx}"),
            vector,
            source_file: format!("seed{seed}_sample_{idx}.wav"),
            start_time_ms,
            duration_ms,
            bpm,
            tags: vec!["drums".to_string(), format!("seed-{seed}")],
        });
    }
    Some(AckRecord {
        vector_id,
        idempotency_key: format!("seed-{seed}-vec-{idx}"),
        vector,
        source_file: format!("seed{seed}_sample_{idx}.wav"),
        start_time_ms,
        duration_ms,
        bpm,
        tags: vec!["drums".to_string(), format!("seed-{seed}")],
    })
}

async fn ingest_batch(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    seed: u64,
    worker_id: usize,
    start_idx: usize,
    count: usize,
) -> (Vec<AckRecord>, Vec<ReplayRecord>) {
    let mut requests = Vec::with_capacity(count);
    let mut template = Vec::with_capacity(count);
    for i in 0..count {
        let idx = start_idx + i;
        let vector = normalized_vector(seed ^ ((worker_id as u64) << 32), idx);
        let source_file = format!("jepsen_seed{seed}_w{worker_id}_{idx}.wav");
        let start_time_ms = idx as u32;
        let duration_ms = 48u16;
        let bpm = 95.0 + ((idx % 100) as f32) * 0.25;
        let tags = vec![
            "drums".to_string(),
            "crash".to_string(),
            format!("worker-{worker_id}"),
        ];
        let idempotency_key = format!("jepsen-{seed}-{worker_id}-{idx}");
        requests.push(IngestRequest {
            vector: vector.clone(),
            metadata: serde_json::json!({
                "source_file": source_file,
                "start_time_ms": start_time_ms,
                "duration_ms": duration_ms,
                "bpm": bpm,
                "tags": tags
            }),
            idempotency_key: Some(idempotency_key.clone()),
        });
        template.push(ReplayRecord {
            idempotency_key,
            vector,
            source_file: format!("jepsen_seed{seed}_w{worker_id}_{idx}.wav"),
            start_time_ms,
            duration_ms,
            bpm,
            tags: vec![
                "drums".to_string(),
                "crash".to_string(),
                format!("worker-{worker_id}"),
            ],
        });
    }

    let payload = IngestBatchRequest { vectors: requests };
    let response = match client
        .post(format!("{}/v2/vectors/batch", base_url))
        .bearer_auth(token)
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return (Vec::new(), template),
    };
    if response.status() != StatusCode::CREATED {
        return (Vec::new(), template);
    }
    let parsed: IngestBatchResponseEnvelope = match response.json().await {
        Ok(v) => v,
        Err(_) => return (Vec::new(), template),
    };
    if parsed.data.results.len() != template.len() {
        return (Vec::new(), template);
    }

    let mut out = Vec::with_capacity(template.len());
    for (rec, result) in template.iter().zip(parsed.data.results.into_iter()) {
        if !result.created {
            continue;
        }
        out.push(AckRecord {
            vector_id: result.id,
            idempotency_key: rec.idempotency_key.clone(),
            vector: rec.vector.clone(),
            source_file: rec.source_file.clone(),
            start_time_ms: rec.start_time_ms,
            duration_ms: rec.duration_ms,
            bpm: rec.bpm,
            tags: rec.tags.clone(),
        });
    }
    (out, template)
}

async fn admin_post(client: &reqwest::Client, base_url: &str, token: &str, path: &str) {
    let _ = client
        .post(format!("{}/{}", base_url, path))
        .bearer_auth(token)
        .send()
        .await;
}

async fn verify_ack_records(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    records: &[AckRecord],
) {
    for rec in records {
        let body = serde_json::json!({
            "vector": rec.vector,
            "metadata": {
                "source_file": rec.source_file,
                "start_time_ms": rec.start_time_ms,
                "duration_ms": rec.duration_ms,
                "bpm": rec.bpm,
                "tags": rec.tags
            },
            "idempotency_key": rec.idempotency_key
        });
        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("reingest acknowledged record");
        assert_eq!(
            resp.status(),
            StatusCode::CREATED,
            "reingest should succeed for {}",
            rec.idempotency_key
        );
        let payload: serde_json::Value = resp.json().await.expect("reingest json");
        assert_eq!(
            payload["data"]["created"].as_bool(),
            Some(false),
            "acknowledged record must already exist after crash recovery ({})",
            rec.idempotency_key
        );
        assert_eq!(
            payload["data"]["id"].as_u64(),
            Some(rec.vector_id as u64),
            "vector id should be stable for idempotency key {}",
            rec.idempotency_key
        );
    }
}

async fn replay_attempted_records(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    records: &[ReplayRecord],
) -> anyhow::Result<(usize, usize)> {
    let mut existing = 0usize;
    let mut inserted = 0usize;
    for rec in records {
        let body = serde_json::json!({
            "vector": rec.vector,
            "metadata": {
                "source_file": rec.source_file,
                "start_time_ms": rec.start_time_ms,
                "duration_ms": rec.duration_ms,
                "bpm": rec.bpm,
                "tags": rec.tags
            },
            "idempotency_key": rec.idempotency_key
        });

        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await?;
        anyhow::ensure!(
            resp.status() == StatusCode::CREATED,
            "replay ingest failed for {}: status={}",
            rec.idempotency_key,
            resp.status()
        );
        let payload: serde_json::Value = resp.json().await?;
        if payload["data"]["created"].as_bool() == Some(false) {
            existing += 1;
        } else {
            inserted += 1;
        }
    }
    Ok((existing, inserted))
}

fn verify_metadata_integrity(data_dir: &Path, records: &[AckRecord]) -> anyhow::Result<()> {
    let cfg = ProductionConfig::from_data_dir(data_dir.to_path_buf(), "default".to_string(), DIM);
    let catalog = SqliteCatalog::open(&cfg.catalog_path())?;
    let collection = catalog.ensure_collection("default", DIM)?;
    let ids = records.iter().map(|r| r.vector_id).collect::<Vec<_>>();
    let map = catalog.fetch_metadata(&collection.id, &ids)?;

    for rec in records {
        let meta = map
            .get(&rec.vector_id)
            .ok_or_else(|| anyhow::anyhow!("metadata missing for vector_id {}", rec.vector_id))?;
        anyhow::ensure!(
            meta.source_file == rec.source_file,
            "metadata source_file mismatch for {}",
            rec.idempotency_key
        );
        anyhow::ensure!(
            meta.start_time_ms == rec.start_time_ms,
            "metadata start_time mismatch for {}",
            rec.idempotency_key
        );
        anyhow::ensure!(
            meta.duration_ms == rec.duration_ms,
            "metadata duration mismatch for {}",
            rec.idempotency_key
        );
    }
    Ok(())
}

async fn run_seed(seed: u64) -> anyhow::Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join(format!("crash_seed_{seed}"));
    std::fs::create_dir_all(&data_dir)?;
    let token = create_api_key(&data_dir)?;

    let Some(port) = reserve_local_port() else {
        return Ok(());
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()?;

    let mut server = start_server(&data_dir, port).await?;
    wait_for_ready(&base_url)
        .await
        .map_err(anyhow::Error::msg)?;

    let mut acked: Vec<AckRecord> = Vec::new();
    let mut next_idx = 0usize;
    for _ in 0..24 {
        if let Some(rec) = ingest_one(&client, &base_url, &token, seed, next_idx).await {
            acked.push(rec);
        }
        next_idx += 1;
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A_D3C4_B2E1);
    match (seed % 3) as u8 {
        0 => {
            for _ in 0..80 {
                if let Some(rec) = ingest_one(&client, &base_url, &token, seed, next_idx).await {
                    acked.push(rec);
                }
                next_idx += 1;
                if rng.gen::<f32>() < 0.20 {
                    sleep(Duration::from_millis(2)).await;
                }
                if next_idx % 17 == 0 {
                    // Kill mid-ingest boundary.
                    kill_server(&mut server).await;
                    break;
                }
            }
        }
        1 => {
            for _ in 0..48 {
                if let Some(rec) = ingest_one(&client, &base_url, &token, seed, next_idx).await {
                    acked.push(rec);
                }
                next_idx += 1;
            }
            let cp = admin_post(&client, &base_url, &token, "v2/admin/checkpoint");
            let jitter_ms = 5 + (rng.gen::<u64>() % 40);
            tokio::pin!(cp);
            tokio::select! {
                _ = sleep(Duration::from_millis(jitter_ms)) => {
                    kill_server(&mut server).await;
                }
                _ = &mut cp => {
                    kill_server(&mut server).await;
                }
            }
        }
        _ => {
            for _ in 0..2 {
                for _ in 0..36 {
                    if let Some(rec) = ingest_one(&client, &base_url, &token, seed, next_idx).await
                    {
                        acked.push(rec);
                    }
                    next_idx += 1;
                }
                admin_post(&client, &base_url, &token, "v2/admin/checkpoint").await;
            }
            let compact = admin_post(&client, &base_url, &token, "v2/admin/compact");
            let jitter_ms = 5 + (rng.gen::<u64>() % 50);
            tokio::pin!(compact);
            tokio::select! {
                _ = sleep(Duration::from_millis(jitter_ms)) => {
                    kill_server(&mut server).await;
                }
                _ = &mut compact => {
                    kill_server(&mut server).await;
                }
            }
        }
    }

    let mut restarted = start_server(&data_dir, port).await?;
    wait_for_ready(&base_url)
        .await
        .map_err(anyhow::Error::msg)?;
    verify_ack_records(&client, &base_url, &token, &acked).await;
    verify_metadata_integrity(&data_dir, &acked)?;

    stop_server(&mut restarted).await;
    Ok(())
}

async fn run_jepsen_seed(seed: u64) -> anyhow::Result<()> {
    let dir = tempdir()?;
    let data_dir = dir.path().join(format!("jepsen_seed_{seed}"));
    std::fs::create_dir_all(&data_dir)?;
    let token = create_api_key(&data_dir)?;

    let Some(port) = reserve_local_port() else {
        return Ok(());
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    let mut server = start_server(&data_dir, port).await?;
    wait_for_ready(&base_url)
        .await
        .map_err(anyhow::Error::msg)?;

    let writer_concurrency = std::env::var("VIBRATO_JEPSEN_WRITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    let batch_size = std::env::var("VIBRATO_JEPSEN_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1);
    let kill_min_ms = std::env::var("VIBRATO_JEPSEN_KILL_MIN_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(80);
    let kill_max_ms = std::env::var("VIBRATO_JEPSEN_KILL_MAX_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1200)
        .max(kill_min_ms);

    let stop = Arc::new(AtomicBool::new(false));
    let next_idx = Arc::new(AtomicUsize::new(0));
    let acked = Arc::new(Mutex::new(Vec::<AckRecord>::new()));
    let attempted = Arc::new(Mutex::new(Vec::<ReplayRecord>::new()));

    let mut writers = Vec::with_capacity(writer_concurrency);
    for worker_id in 0..writer_concurrency {
        let client = client.clone();
        let token = token.clone();
        let base_url = base_url.clone();
        let stop = stop.clone();
        let next_idx = next_idx.clone();
        let acked = acked.clone();
        let attempted = attempted.clone();
        writers.push(tokio::spawn(async move {
            while !stop.load(Ordering::Relaxed) {
                let start = next_idx.fetch_add(batch_size, Ordering::Relaxed);
                let (batch, attempted_batch) = ingest_batch(
                    &client, &base_url, &token, seed, worker_id, start, batch_size,
                )
                .await;
                if !attempted_batch.is_empty() {
                    attempted.lock().extend(attempted_batch);
                }
                if !batch.is_empty() {
                    acked.lock().extend(batch);
                } else if stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }));
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0x6EAF_D11D_55AA_9021);
    let kill_delay_ms = if kill_max_ms == kill_min_ms {
        kill_min_ms
    } else {
        rng.gen_range(kill_min_ms..=kill_max_ms)
    };
    sleep(Duration::from_millis(kill_delay_ms)).await;
    kill_server(&mut server).await;
    stop.store(true, Ordering::Relaxed);

    for task in writers {
        let _ = tokio::time::timeout(Duration::from_secs(3), task).await;
    }
    let acked_records = acked.lock().clone();
    let acked_count = acked_records.len();
    let attempted_records = attempted.lock().clone();
    let attempted_unique = {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for rec in attempted_records {
            if seen.insert(rec.idempotency_key.clone()) {
                out.push(rec);
            }
        }
        out
    };
    let attempted_unique_count = attempted_unique.len();

    let Some(restart_port) = reserve_local_port() else {
        return Ok(());
    };
    let restart_url = format!("http://127.0.0.1:{}", restart_port);
    let mut restarted = start_server(&data_dir, restart_port).await?;
    wait_for_ready(&restart_url)
        .await
        .map_err(anyhow::Error::msg)?;

    let verify_cap = std::env::var("VIBRATO_JEPSEN_VERIFY_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(500);
    let verify_records = if acked_records.len() > verify_cap {
        let stride = (acked_records.len() / verify_cap).max(1);
        acked_records
            .iter()
            .step_by(stride)
            .take(verify_cap)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        acked_records.clone()
    };

    verify_ack_records(&client, &restart_url, &token, &verify_records).await;
    verify_metadata_integrity(&data_dir, &verify_records)?;

    let stats_before_replay_resp = client
        .get(format!("{}/v2/admin/stats", restart_url))
        .bearer_auth(&token)
        .send()
        .await?;
    anyhow::ensure!(
        stats_before_replay_resp.status() == StatusCode::OK,
        "stats endpoint failed after restart: status={}",
        stats_before_replay_resp.status()
    );
    let stats_before_replay: serde_json::Value = stats_before_replay_resp.json().await?;
    let total_vectors_before_replay = stats_before_replay["data"]["total_vectors"]
        .as_u64()
        .unwrap_or(0) as usize;

    let (replay_existing, replay_inserted) =
        replay_attempted_records(&client, &restart_url, &token, &attempted_unique).await?;

    anyhow::ensure!(
        replay_existing == total_vectors_before_replay,
        "jepsen mismatch before replay: seed={} kill_delay_ms={} total_vectors_before_replay={} replay_existing={}",
        seed,
        kill_delay_ms,
        total_vectors_before_replay,
        replay_existing
    );
    anyhow::ensure!(
        replay_existing >= acked_count,
        "acked durability violated: seed={} kill_delay_ms={} acked_count={} replay_existing={}",
        seed,
        kill_delay_ms,
        acked_count,
        replay_existing
    );
    anyhow::ensure!(
        replay_existing + replay_inserted == attempted_unique_count,
        "attempt ledger mismatch: seed={} attempted_unique={} replay_existing={} replay_inserted={}",
        seed,
        attempted_unique_count,
        replay_existing,
        replay_inserted
    );

    let stats_after_replay_resp = client
        .get(format!("{}/v2/admin/stats", restart_url))
        .bearer_auth(&token)
        .send()
        .await?;
    anyhow::ensure!(
        stats_after_replay_resp.status() == StatusCode::OK,
        "stats endpoint failed after replay: status={}",
        stats_after_replay_resp.status()
    );
    let stats_after_replay: serde_json::Value = stats_after_replay_resp.json().await?;
    let total_vectors_after_replay = stats_after_replay["data"]["total_vectors"]
        .as_u64()
        .unwrap_or(0) as usize;
    anyhow::ensure!(
        total_vectors_after_replay == attempted_unique_count,
        "replay final count mismatch: seed={} expected={} actual={}",
        seed,
        attempted_unique_count,
        total_vectors_after_replay
    );

    stop_server(&mut restarted).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "long-running crash-matrix suite; run in CI nightly or explicitly"]
async fn crash_matrix_kill9_100_seed_integrity() {
    let seeds = std::env::var("VIBRATO_CRASH_SEEDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100);

    for seed in 0..seeds {
        run_seed(seed)
            .await
            .unwrap_or_else(|e| panic!("seed {} failed: {}", seed, e));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "jepsen-style kill-9 during active batch ingest; run in CI/nightly or explicitly"]
async fn crash_matrix_kill9_random_ms_exact_ack_recovery() {
    let seeds = std::env::var("VIBRATO_JEPSEN_CRASH_SEEDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(24);
    for seed in 0..seeds {
        run_jepsen_seed(seed)
            .await
            .unwrap_or_else(|e| panic!("jepsen seed {} failed: {}", seed, e));
    }
}
