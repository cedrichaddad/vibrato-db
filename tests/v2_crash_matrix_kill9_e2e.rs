use std::net::TcpListener;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
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
