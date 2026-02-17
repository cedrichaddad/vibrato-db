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

const DIM: usize = 64;
const INGEST_COUNT: usize = 2500;
const QUERY_COUNT: usize = 250;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, port: u16, madvise_mode: &str) -> std::io::Result<Child> {
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
        .arg("--vector-madvise-mode")
        .arg(madvise_mode)
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
    for _ in 0..160 {
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
        .arg("madvise-ab")
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

fn percentile_micros(values_us: &mut [f64], p: f64) -> f64 {
    if values_us.is_empty() {
        return 0.0;
    }
    values_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((values_us.len() - 1) as f64 * p).round() as usize;
    values_us[rank]
}

async fn ingest_dataset(client: &reqwest::Client, base_url: &str, token: &str) {
    for i in 0..INGEST_COUNT {
        let body = serde_json::json!({
            "vector": normalized_vector(12345, i),
            "metadata": {
                "source_file": format!("madvise_{i}.wav"),
                "start_time_ms": i * 5,
                "duration_ms": 20,
                "bpm": 128.0,
                "tags": ["perf", "madvise"]
            },
            "idempotency_key": format!("madvise-{i}")
        });
        let resp = client
            .post(format!("{}/v2/vectors", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("ingest");
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    let checkpoint = client
        .post(format!("{}/v2/admin/checkpoint", base_url))
        .bearer_auth(token)
        .send()
        .await
        .expect("checkpoint");
    assert_eq!(checkpoint.status(), StatusCode::OK);
}

async fn measure_query_p99_us(
    client: &reqwest::Client,
    base_url: &str,
    token: &str,
    seed: u64,
) -> f64 {
    let mut times_us = Vec::with_capacity(QUERY_COUNT);
    for i in 0..QUERY_COUNT {
        let body = serde_json::json!({
            "vector": normalized_vector(seed, i * 7 + 11),
            "k": 10,
            "ef": 50,
            "include_metadata": false,
            "search_tier": "active"
        });
        let resp = client
            .post(format!("{}/v2/query", base_url))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .expect("query");
        assert_eq!(resp.status(), StatusCode::OK);
        let payload: serde_json::Value = resp.json().await.expect("query payload");
        let query_ms = payload["data"]["query_time_ms"]
            .as_f64()
            .expect("query_time_ms");
        times_us.push(query_ms * 1000.0);
    }
    percentile_micros(&mut times_us, 0.99)
}

#[tokio::test]
#[ignore = "perf harness (run on dedicated runner/reference hardware)"]
async fn madvise_ab_active_tier_query_latency() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("madvise_ab_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping madvise A/B test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut normal = match start_server(&data_dir, port, "normal").await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping madvise A/B test: failed to spawn normal server: {}",
                e
            );
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready normal");
    ingest_dataset(&client, &base_url, &token).await;
    let normal_p99_us = measure_query_p99_us(&client, &base_url, &token, 9001).await;
    stop_server(&mut normal).await;

    let mut random = start_server(&data_dir, port, "random")
        .await
        .expect("start random server");
    wait_for_ready(&base_url).await.expect("ready random");
    let random_p99_us = measure_query_p99_us(&client, &base_url, &token, 9001).await;
    stop_server(&mut random).await;

    eprintln!(
        "madvise_ab: normal_p99_us={:.2} random_p99_us={:.2}",
        normal_p99_us, random_p99_us
    );

    if std::env::var("VIBRATO_ENFORCE_MADVISE_GATE")
        .ok()
        .as_deref()
        == Some("1")
    {
        let max_allowed = normal_p99_us * 1.30;
        assert!(
            random_p99_us <= max_allowed,
            "madvise random regressed: {:.2}us > {:.2}us (normal {:.2}us)",
            random_p99_us,
            max_allowed,
            normal_p99_us
        );
    }
}
