use std::f32::consts::TAU;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;

const DIM: usize = 128;
const TOTAL_VECTORS: usize = 10_000;
const QUERY_START: usize = 5_000;
const QUERY_LEN: usize = 50;

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
        .stderr(Stdio::null());
    cmd.spawn()
}

async fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    let ready_url = format!("{}/v2/health/ready", base_url);
    for _ in 0..180 {
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
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
}

fn create_api_key(data_dir: &Path) -> anyhow::Result<String> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vibrato-db"))
        .arg("key-create")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("identify-protocol")
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

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for lane in &mut vector {
        *lane /= norm;
    }
    vector
}

fn deterministic_frame(idx: usize) -> Vec<f32> {
    let mut state = splitmix64((idx as u64) ^ 0xD1B54A32D192ED03);
    let mut vector = Vec::with_capacity(DIM);
    for lane in 0..DIM {
        state = splitmix64(state ^ lane as u64);
        let unit = (state as f64 / u64::MAX as f64) as f32;
        vector.push(unit * 2.0 - 1.0);
    }
    normalize(vector)
}

fn gaussian_noise(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().clamp(1e-7, 1.0 - 1e-7);
    let u2 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = TAU * u2;
    r * theta.cos()
}

fn add_gaussian_noise(base: &[f32], sigma: f32, rng: &mut StdRng) -> Vec<f32> {
    let noisy = base
        .iter()
        .map(|value| *value + sigma * gaussian_noise(rng))
        .collect::<Vec<_>>();
    normalize(noisy)
}

async fn ingest_frame(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    idx: usize,
    vector: &[f32],
) -> usize {
    let body = serde_json::json!({
        "vector": vector,
        "metadata": {
            "source_file": "known_track.wav",
            "start_time_ms": idx * 20,
            "duration_ms": 20,
            "bpm": 128.0,
            "tags": ["identify", "protocol"]
        },
        "idempotency_key": format!("known-track-{idx}")
    });
    let resp = client
        .post(format!("{}/v2/vectors", base_url))
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .expect("ingest request");
    assert_eq!(resp.status(), StatusCode::CREATED);
    let payload: serde_json::Value = resp.json().await.expect("ingest json");
    payload["data"]["id"].as_u64().expect("ingest id") as usize
}

async fn identify(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    vectors: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> serde_json::Value {
    let resp = client
        .post(format!("{}/v2/identify", base_url))
        .bearer_auth(token)
        .json(&serde_json::json!({
            "vectors": vectors,
            "k": k,
            "ef": ef,
            "include_metadata": true
        }))
        .send()
        .await
        .expect("identify request");
    assert_eq!(resp.status(), StatusCode::OK);
    resp.json().await.expect("identify payload")
}

#[tokio::test]
async fn identify_protocol_perfect_noisy_and_silent_anchor() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("identify_protocol_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping identify protocol test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping identify protocol test: failed to spawn server: {}",
                e
            );
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready");

    let dataset: Vec<Vec<f32>> = (0..TOTAL_VECTORS).map(deterministic_frame).collect();
    for (idx, vector) in dataset.iter().enumerate() {
        let assigned_id = ingest_frame(&client, &base_url, &token, idx, vector).await;
        assert_eq!(
            assigned_id, idx,
            "vector ids must stay monotonic and contiguous for identify sequence semantics"
        );
    }

    let perfect_query = dataset[QUERY_START..QUERY_START + QUERY_LEN].to_vec();
    let perfect_payload = identify(&client, &base_url, &token, &perfect_query, 5, 400).await;
    let perfect_results = perfect_payload["data"]["results"]
        .as_array()
        .expect("perfect results array");
    assert!(!perfect_results.is_empty(), "perfect query should match");
    assert_eq!(
        perfect_results[0]["id"].as_u64(),
        Some(QUERY_START as u64),
        "perfect identify should return exact sequence start id"
    );
    let perfect_score = perfect_results[0]["score"].as_f64().expect("perfect score");
    assert!(
        perfect_score >= 0.995,
        "perfect identify score should be near 1.0, got {perfect_score}"
    );

    let mut rng = StdRng::seed_from_u64(0xA11D10BEEFu64);
    let noisy_query = perfect_query
        .iter()
        .map(|vector| add_gaussian_noise(vector, 0.03, &mut rng))
        .collect::<Vec<_>>();
    let noisy_payload = identify(&client, &base_url, &token, &noisy_query, 8, 512).await;
    let noisy_results = noisy_payload["data"]["results"]
        .as_array()
        .expect("noisy results array");
    assert!(
        !noisy_results.is_empty(),
        "noisy query should still produce candidates"
    );
    let noisy_rank = noisy_results
        .iter()
        .position(|row| row["id"].as_u64() == Some(QUERY_START as u64))
        .expect("noisy identify should still include exact start id");
    assert!(
        noisy_rank <= 2,
        "expected exact start id in top-3 for noisy query, got rank {}",
        noisy_rank + 1
    );
    let noisy_score = noisy_results[noisy_rank]["score"]
        .as_f64()
        .expect("noisy score");
    assert!(
        noisy_score >= 0.90,
        "noisy identify score should remain high, got {noisy_score}"
    );

    let mut silent_query = vec![vec![0.0f32; DIM]; QUERY_LEN];
    let mut artifact = deterministic_frame(42_424_242);
    for lane in &mut artifact {
        *lane *= 5.0;
    }
    silent_query[QUERY_LEN / 2] = artifact;
    let silent_payload = identify(&client, &base_url, &token, &silent_query, 5, 512).await;
    let silent_results = silent_payload["data"]["results"]
        .as_array()
        .expect("silent results array");
    assert!(
        silent_results
            .iter()
            .all(|row| row["score"].as_f64().unwrap_or(0.0) < 0.20),
        "silent-anchor query should not return high-confidence false positives"
    );

    stop_server(&mut server).await;
}
