use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::StatusCode;
use tempfile::tempdir;
use tokio::process::{Child, Command};
use tokio::time::sleep;

fn reserve_local_port() -> Option<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let port = listener.local_addr().ok()?.port();
    drop(listener);
    Some(port)
}

async fn start_server(data_dir: &Path, port: u16) -> std::io::Result<Child> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vibrato-db"));
    cmd.arg("serve-v3")
        .env("VIBRATO_API_PEPPER", "test-pepper")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--collection")
        .arg("default")
        .arg("--dim")
        .arg("2")
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
    let ready_url = format!("{}/v3/health/ready", base_url);
    for _ in 0..80 {
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
        .env("VIBRATO_API_PEPPER", "test-pepper")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--name")
        .arg("identify-validation")
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

fn metric_value(metrics: &str, name: &str) -> Option<u64> {
    let prefix = format!("{name} ");
    metrics.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}

fn metric_value_f64(metrics: &str, name: &str) -> Option<f64> {
    let prefix = format!("{name} ");
    metrics.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .and_then(|v| v.trim().parse::<f64>().ok())
    })
}

fn histogram_bucket(metrics: &str, name: &str, le: &str) -> Option<u64> {
    let prefix = format!("{name}{{le=\"{le}\"}} ");
    metrics.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}

#[tokio::test]
async fn identify_endpoint_auth_validation_and_metrics() {
    let dir = tempdir().expect("tempdir");
    let data_dir: PathBuf = dir.path().join("identify_validation_data");
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let token = create_api_key(&data_dir).expect("create key");

    let Some(port) = reserve_local_port() else {
        eprintln!("skipping identify validation test: localhost bind unavailable");
        return;
    };
    let base_url = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    let mut server = match start_server(&data_dir, port).await {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "skipping identify validation test: failed to spawn server: {}",
                e
            );
            return;
        }
    };
    wait_for_ready(&base_url).await.expect("ready");

    let unauth = client
        .post(format!("{}/v3/identify", base_url))
        .json(&serde_json::json!({
            "vectors": [[1.0, 0.0]],
            "k": 1,
            "ef": 50
        }))
        .send()
        .await
        .expect("unauth identify");
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let empty_vectors = client
        .post(format!("{}/v3/identify", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vectors": [],
            "k": 1,
            "ef": 50
        }))
        .send()
        .await
        .expect("empty vectors identify");
    assert_eq!(empty_vectors.status(), StatusCode::BAD_REQUEST);

    let bad_dim = client
        .post(format!("{}/v3/identify", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vectors": [[1.0, 0.0, 0.0]],
            "k": 1,
            "ef": 50
        }))
        .send()
        .await
        .expect("bad dim identify");
    assert_eq!(bad_dim.status(), StatusCode::BAD_REQUEST);

    let ingest = client
        .post(format!("{}/v3/vectors", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vector": [1.0, 0.0],
            "metadata": {
                "source_file": "one.wav",
                "start_time_ms": 0,
                "duration_ms": 100,
                "bpm": 120.0,
                "tags": ["identify"]
            },
            "idempotency_key": "identify-validation-1"
        }))
        .send()
        .await
        .expect("ingest");
    assert_eq!(ingest.status(), StatusCode::CREATED);

    let k_zero = client
        .post(format!("{}/v3/identify", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vectors": [[1.0, 0.0]],
            "k": 0,
            "ef": 50
        }))
        .send()
        .await
        .expect("k=0 identify");
    assert_eq!(k_zero.status(), StatusCode::OK);
    let k_zero_payload: serde_json::Value = k_zero.json().await.expect("k=0 payload");
    assert_eq!(
        k_zero_payload["data"]["results"]
            .as_array()
            .map(|a| a.len()),
        Some(0)
    );

    let ok_identify = client
        .post(format!("{}/v3/identify", base_url))
        .bearer_auth(&token)
        .json(&serde_json::json!({
            "vectors": [[1.0, 0.0]],
            "k": 1,
            "ef": 50
        }))
        .send()
        .await
        .expect("ok identify");
    assert_eq!(ok_identify.status(), StatusCode::OK);

    let metrics = client
        .get(format!("{}/v3/metrics", base_url))
        .bearer_auth(&token)
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("metrics text");
    let identify_total = metric_value(&metrics, "vibrato_identify_requests_total")
        .expect("identify metric should be present");
    assert_eq!(identify_total, 1);

    let h_count = metric_value(&metrics, "vibrato_query_latency_seconds_count")
        .expect("histogram count should be present");
    let h_sum = metric_value_f64(&metrics, "vibrato_query_latency_seconds_sum")
        .expect("histogram sum present");
    assert!(
        h_count >= 1,
        "latency histogram should observe at least one query"
    );
    assert!(h_sum > 0.0, "latency histogram sum should be non-zero");

    let b10 = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "0.00001")
        .expect("bucket le=0.00001 missing");
    let b25 = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "0.000025")
        .expect("bucket le=0.000025 missing");
    let b50 = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "0.00005")
        .expect("bucket le=0.00005 missing");
    let b100 = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "0.0001")
        .expect("bucket le=0.0001 missing");
    let b500 = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "0.0005")
        .expect("bucket le=0.0005 missing");
    let b1000 = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "0.001")
        .expect("bucket le=0.001 missing");
    let b_inf = histogram_bucket(&metrics, "vibrato_query_latency_seconds_bucket", "+Inf")
        .expect("bucket le=+Inf missing");

    assert!(b10 <= b25);
    assert!(b25 <= b50);
    assert!(b50 <= b100);
    assert!(b100 <= b500);
    assert!(b500 <= b1000);
    assert!(b1000 <= b_inf);
    assert_eq!(b_inf, h_count);

    stop_server(&mut server).await;
}
